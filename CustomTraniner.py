from transformers import Trainer
from torch import nn
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import torch
from transformers.utils import is_sagemaker_mp_enabled, is_apex_available

class CustomTrainer(Trainer):
    def __init__(self, group_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.group_weights = group_weights
        
    def compute_loss(self, model, inputs, return_outputs=False , idx = 0):
        labels = inputs.pop(f"labels{idx+1}")
        outputs = model(**inputs)
        logits = outputs[0][idx]
        # global_score_loss = torch.nn.functional.cross_entropy(logits, labels)
        cause_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        
        # loss = self.group_weights[0] * global_score_loss +self.group_weights[1] * cause_loss
        return (cause_loss, outputs) if return_outputs else cause_loss
    

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
       
        model.train()
        inputs = self._prepare_inputs(inputs)
        li=[]
        with self.compute_loss_context_manager():
            loss1 = self.compute_loss(model, inputs, idx = 0)

        if self.args.n_gpu > 1:
            loss1 = loss1.mean()  # mean() to average on multi-gpu parallel training

        self.accelerator.backward(loss1)

        with self.compute_loss_context_manager():
            loss2 = self.compute_loss(model, inputs, idx = 1)

        if self.args.n_gpu > 1:
            loss2 = loss2.mean()  # mean() to average on multi-gpu parallel training
        self.accelerator.backward(loss2)

        with self.compute_loss_context_manager():
            loss3 = self.compute_loss(model, inputs, idx = 2)

        if self.args.n_gpu > 1:
            loss3 = loss3.mean()  # mean() to average on multi-gpu parallel training
        self.accelerator.backward(loss3)

        loss = loss1.detach() + loss2.detach() + loss3.detach()

        return loss / self.args.gradient_accumulation_steps
    
class CustomTrainer_add_loss(Trainer):
    def __init__(self, group_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.group_weights = group_weights
        
    def compute_loss(self, model, inputs, return_outputs=False ):
        labels_emotion = inputs.pop(f"labels1")
        labels_tempo = inputs.pop(f"labels2")
        labels_genre = inputs.pop(f"labels3")
        outputs = model(**inputs)
        logits_emotion = outputs[0][0]
        logits_tempo = outputs[0][1]
        logits_genre = outputs[0][2]
        # logits = nn.Softmax(logits)
        # global_score_loss = torch.nn.functional.cross_entropy(logits, labels)
        loss_emotion = torch.nn.functional.binary_cross_entropy_with_logits(logits_emotion, labels_emotion)
        loss_tempo = torch.nn.functional.binary_cross_entropy_with_logits(logits_tempo, labels_tempo)
        loss_genre = torch.nn.functional.binary_cross_entropy_with_logits(logits_genre, labels_genre)
        loss = loss_emotion + loss_tempo + loss_genre
        return (loss, outputs) if return_outputs else loss
    

     
class CustomTrainer_cross_entropy(Trainer):
    def __init__(self, group_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.group_weights = group_weights
        
    def compute_loss(self, model, inputs, return_outputs=False , idx = 0):
        labels = inputs.pop(f"labels{idx+1}")
        outputs = model(**inputs)
        logits = outputs[0][idx]
        # logits = nn.Softmax(logits)
        # global_score_loss = torch.nn.functional.cross_entropy(logits, labels)
        cause_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        
        # loss = self.group_weights[0] * global_score_loss +self.group_weights[1] * cause_loss
        return (cause_loss, outputs) if return_outputs else cause_loss
    

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
       
        model.train()
        inputs = self._prepare_inputs(inputs)
        li=[]
        with self.compute_loss_context_manager():
            loss1 = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss1 = loss.mean()  # mean() to average on multi-gpu parallel training


        with self.compute_loss_context_manager():
            loss2 = self.compute_loss(model, inputs, idx = 1)

        if self.args.n_gpu > 1:
            loss2 = loss2.mean()  # mean() to average on multi-gpu parallel training
 

        with self.compute_loss_context_manager():
            loss3 = self.compute_loss(model, inputs, idx = 2)

        if self.args.n_gpu > 1:
            loss3 = loss3.mean()  # mean() to average on multi-gpu parallel training

        loss =loss1+loss2+loss3
        self.accelerator.backward(loss)

        return loss.detach() / self.args.gradient_accumulation_steps

class BingModelCustomTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False ):
        labels_emotion = inputs.pop(f"labels1")
        labels_tempo = inputs.pop(f"labels2")
        labels_genre = inputs.pop(f"labels3")
        outputs = model(**inputs)
        logits_emotion = outputs[0][0]
        logits_tempo = outputs[0][1]
        logits_genre = outputs[0][2]
        # logits = nn.Softmax(logits)
        # global_score_loss = torch.nn.functional.cross_entropy(logits, labels)
        
        # print(logits_emotion,logits_tempo, logits_genre)
        loss_emotion = torch.nn.functional.binary_cross_entropy_with_logits(logits_emotion, labels_emotion)
        loss_tempo = torch.nn.functional.binary_cross_entropy_with_logits(logits_tempo, labels_tempo)
        loss_genre = torch.nn.functional.binary_cross_entropy_with_logits(logits_genre, labels_genre)
        
        # if self.args.n_gpu > 1:
        #     print('여기 들어가냐?')
        #     loss_emotion = loss_emotion.mean() 
        #     loss_tempo = loss_tempo.mean()
        #     loss_genre = loss_genre.mean()
        # print('loss_emotion : ',loss_emotion)
        # print('loss_tempo : ',loss_tempo)
        # print('loss_genre : ',loss_genre)

        # self.accelerator.backward(loss_emotion)
        # self.accelerator.backward(loss_tempo)
        # self.accelerator.backward(loss_genre)
        # loss = loss_emotion + loss_tempo +loss_genre
        # # print('loss total : ',loss)
        loss =loss_emotion+ loss_tempo +loss_genre
        return (loss, outputs) if return_outputs else loss
    
    

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
       
        model.train()
        inputs = self._prepare_inputs(inputs)
        li=[]
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        # loss_emotion = loss[0]
        # loss_tempo = loss[1]
        # loss_genre = loss[2]
        
        if self.args.n_gpu > 1:
            loss = loss.mean() 
            # loss_tempo = loss_tempo.mean()
            # loss_genre = loss_genre.mean()
        
        
        
        self.accelerator.backward(loss)
        # self.accelerator.backward(loss_tempo)
        # self.accelerator.backward(loss_genre)
        
        # loss =loss_emotion+loss_tempo+loss_genre
        # print('loss여기: ',loss)
            
        # self.accelerator.backward(loss)

        return loss.detach() / self.args.gradient_accumulation_steps