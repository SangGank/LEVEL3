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
        cause_loss = torch.nn.functional.cross_entropy(logits, labels)
        
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
