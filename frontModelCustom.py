from tqdm import tqdm
from transformers import (BertForSequenceClassification, BertModel, BertPreTrainedModel, AutoTokenizer,
                          AutoConfig, RobertaPreTrainedModel, RobertaModel, ElectraPreTrainedModel,
                          ElectraModel, GPT2PreTrainedModel, GPT2Model)
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from typing import List, Optional, Tuple, Union, Dict, Any
from transformers.modeling_outputs import SequenceClassifierOutput, SequenceClassifierOutputWithPast
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead
from transformers.activations import get_activation
import torch
import pandas as pd
from transformers import Trainer
from torch import nn
from transformers.utils import is_sagemaker_mp_enabled, is_apex_available
import pickle


def get_weighted_loss(loss_fct, inputs, labels, weights):
    loss = 0.0
    for i in range(weights.shape[0]):
        loss += (weights[i] + 1.0) * loss_fct(inputs[i:i + 1], labels[i:i + 1])

    return loss / (sum(weights) + weights.shape[0])


class customBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, num_labels1 = None, num_labels2 = None, num_labels3 = None ):
        super().__init__(config)
        self.num_labels1 = config.num_labels1
        self.num_labels2 = config.num_labels2
        self.num_labels3 = config.num_labels3
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier1 = nn.Linear(config.hidden_size, self.num_labels1)
        self.classifier2 = nn.Linear(config.hidden_size, self.num_labels2)
        self.classifier3 = nn.Linear(config.hidden_size, self.num_labels3)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels1: Optional[torch.Tensor] = None,
        labels2: Optional[torch.Tensor] = None,
        labels3: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits1 = self.classifier1(pooled_output)
        logits2 = self.classifier2(pooled_output)
        logits3 = self.classifier3(pooled_output)

        loss1 = None
        loss2 = None
        loss3 = None
        
        loss =None
        if loss1 and loss2 and loss3:
            loss = loss1 + loss2 + loss3
        if not return_dict:
            output = (logits1,) + (logits2,) + (logits3,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        return SequenceClassifierOutput(
            loss=loss,
            logits=(logits1,) + (logits2,) + (logits3,),
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    


class customRobertaForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, num_labels1 = None, num_labels2 = None, num_labels3 = None ):
        super().__init__(config)
        self.num_labels1 = config.num_labels1
        self.num_labels2 = config.num_labels2
        self.num_labels3 = config.num_labels3
        
        self.config = config
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier1 = nn.Linear(config.hidden_size, self.num_labels1)
        self.classifier2 = nn.Linear(config.hidden_size, self.num_labels2)
        self.classifier3 = nn.Linear(config.hidden_size, self.num_labels3)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels1: Optional[torch.Tensor] = None,
        labels2: Optional[torch.Tensor] = None,
        labels3: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[0]
        pooled_output = pooled_output[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        output1 = self.dense(pooled_output)
        output1 = torch.tanh(output1)
        output1 = self.dropout(output1)

        output2 = self.dense(pooled_output)
        output2 = torch.tanh(output2)
        output2 = self.dropout(output2)


        output3 = self.dense(pooled_output)
        output3 = torch.tanh(output3)
        output3 = self.dropout(output3)

        logits1 = self.classifier1(output1)
        logits2 = self.classifier2(output2)
        logits3 = self.classifier3(output3)

        loss1 = None
        loss2 = None
        loss3 = None
        
        loss =None
        if loss1 and loss2 and loss3:
            loss = loss1 + loss2 + loss3
        if not return_dict:
            output = (logits1,) + (logits2,) + (logits3,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        return SequenceClassifierOutput(
            loss=loss,
            logits=(logits1,) + (logits2,) + (logits3,),
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    

class customElectraForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, num_labels1 = None, num_labels2 = None, num_labels3 = None ):
        super().__init__(config)
        self.num_labels1 = config.num_labels1
        self.num_labels2 = config.num_labels2
        self.num_labels3 = config.num_labels3
        self.config = config

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.electra = ElectraModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.activation = get_activation("gelu")
        self.classifier1 = nn.Linear(config.hidden_size, self.num_labels1)
        self.classifier2 = nn.Linear(config.hidden_size, self.num_labels2)
        self.classifier3 = nn.Linear(config.hidden_size, self.num_labels3)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels1: Optional[torch.Tensor] = None,
        labels2: Optional[torch.Tensor] = None,
        labels3: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.electra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        pooled_output = outputs[0]
        pooled_output = pooled_output[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        output1 = self.dense(pooled_output)
        output1 = self.activation(output1)
        output1 = self.dropout(output1)

        output2 = self.dense(pooled_output)
        output2 = self.activation(output2)
        output2 = self.dropout(output2)

        output3 = self.dense(pooled_output)
        output3 = self.activation(output3)
        output3 = self.dropout(output3)




        logits1 = self.classifier1(output1)
        logits2 = self.classifier2(output2)
        logits3 = self.classifier3(output3)

        loss1 = None
        loss2 = None
        loss3 = None
        
        loss =None
        if loss1 and loss2 and loss3:
            loss = loss1 + loss2 + loss3
        if not return_dict:
            output = (logits1,) + (logits2,) + (logits3,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        return SequenceClassifierOutput(
            loss=loss,
            logits=(logits1,) + (logits2,) + (logits3,),
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
# 아직 안됨 
class customGPT2ForSequenceClassification(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels1 = config.num_labels1
        self.num_labels2 = config.num_labels2
        self.num_labels3 = config.num_labels3
        self.config = config
        self.GPT2 = GPT2Model(config)
        self.classifier1 = nn.Linear(config.hidden_size, self.num_labels1)
        self.classifier2 = nn.Linear(config.hidden_size, self.num_labels2)
        self.classifier3 = nn.Linear(config.hidden_size, self.num_labels3)
        
        # Model parallel
        self.model_parallel = False
        self.device_map = None


        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels1: Optional[torch.Tensor] = None,
        labels2: Optional[torch.Tensor] = None,
        labels3: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.GPT2(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        
        logits1 = self.classifier1(hidden_states)
        logits2 = self.classifier2(hidden_states)
        logits3 = self.classifier3(hidden_states)


        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape[:2]
        else:
            batch_size, sequence_length = inputs_embeds.shape[:2]
            
        assert (
            self.config.pad_token_id is not None or batch_size == 1
        ), "Cannot handle batch sizes > 1 if no padding token is defined."
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits1.device)
            else:
                sequence_lengths = -1

        pooled_logits1 = logits1[torch.arange(batch_size, device=logits1.device), sequence_lengths]
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits2.device)
            else:
                sequence_lengths = -1
        pooled_logits2 = logits2[torch.arange(batch_size, device=logits2.device), sequence_lengths]
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits3.device)
            else:
                sequence_lengths = -1
        pooled_logits3 = logits3[torch.arange(batch_size, device=logits3.device), sequence_lengths]

        
        loss1 = None
        loss2 = None
        loss3 = None
        
        loss =None
        if loss1 and loss2 and loss3:
            loss = loss1 + loss2 + loss3
            
        if not return_dict:
            output = (logits1,) + (logits2,) + (logits3,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=(pooled_logits1,) + (pooled_logits2,) + (pooled_logits3,),
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


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
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
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


# def data_labels(label_data_path ='./labels.pkl'):
#     with open(label_data_path,'rb') as f:
#         emotion_labels=pickle.load(f)
#         tempo_labels=pickle.load(f)
#         genre_labels=pickle.load(f)
    
#     return emotion_labels, tempo_labels, genre_labels



def data_labels(label_data_path ='./labels.pkl'):
    with open(label_data_path,'rb') as f:
        labels = pickle.load(f)
    
    return labels['emotion_labels'], labels['tempo_labels'], labels['genre_labels']


def id2labelData_labels(label_data_path ='./labels.pkl'):
    # with open(label_data_path,'rb') as f:
    #     emotion_labels=pickle.load(f)
    #     tempo_labels=pickle.load(f)
    #     genre_labels=pickle.load(f)
    emotion_labels,tempo_labels,genre_labels = data_labels(label_data_path = label_data_path)
    id2label_emotion = {k:l for k, l in enumerate(emotion_labels)}
    id2label_tempo = {k:l for k, l in enumerate(tempo_labels)}
    id2label_genre = {k:l for k, l in enumerate(genre_labels)}
    return id2label_emotion, id2label_tempo, id2label_genre
    

class frontModelDataset:
    def __init__(self, data, tokenizer, label_data_path ='./labels.pkl'):

        emotion_labels, tempo_labels, genre_labels= data_labels(label_data_path)
        
        id2label_emotion = {k:l for k, l in enumerate(emotion_labels)}
        label2id_emotion = {l:k for k, l in enumerate(emotion_labels)}
        id2label_tempo = {k:l for k, l in enumerate(tempo_labels)}
        label2id_tempo = {l:k for k, l in enumerate(tempo_labels)}
        id2label_genre = {k:l for k, l in enumerate(genre_labels)}
        label2id_genre = {l:k for k, l in enumerate(genre_labels)}

        self.tokenizer = tokenizer
        self.dataset = []
        datas = []
        self.labels1 = []
        self.labels2 = []
        self.labels3 = []
        for idx, df in tqdm(data.iterrows()):
            label1 = [0. for _ in range(len(id2label_emotion))]
            label2 = [0. for _ in range(len(id2label_tempo))]
            label3 = [0. for _ in range(len(id2label_genre))]
            datas.append(df.caption)
            label1[label2id_emotion[df.emotion]] = 1.
            label2[label2id_tempo[df['tempo(category)']]] = 1.
            label3[label2id_genre[df['genre']]] = 1.
            self.labels1.append(label1)
            self.labels2.append(label2)
            self.labels3.append(label3)
        
        self.dataset =  tokenizer(datas,padding=True, truncation=True,max_length=512 ,return_tensors="pt").to('cuda')
        self.labels1= torch.tensor(self.labels1)
        self.labels2= torch.tensor(self.labels2)
        self.labels3= torch.tensor(self.labels3)

    def __len__(self):
        return len(self.labels1)
    
    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.dataset.items()}
        item['labels1'] = self.labels1[idx].clone().detach()
        item['labels2'] = self.labels2[idx].clone().detach()
        item['labels3'] = self.labels3[idx].clone().detach()
        return item
    
class frontModelDataset_not_stopword:
    def __init__(self, data, tokenizer, label_data_path ='./labels.pkl'):

        emotion_labels, tempo_labels, genre_labels= data_labels(label_data_path)
        
        id2label_emotion = {k:l for k, l in enumerate(emotion_labels)}
        label2id_emotion = {l:k for k, l in enumerate(emotion_labels)}
        id2label_tempo = {k:l for k, l in enumerate(tempo_labels)}
        label2id_tempo = {l:k for k, l in enumerate(tempo_labels)}
        id2label_genre = {k:l for k, l in enumerate(genre_labels)}
        label2id_genre = {l:k for k, l in enumerate(genre_labels)}

        self.tokenizer = tokenizer
        self.dataset = []
        datas = []
        self.labels1 = []
        self.labels2 = []
        self.labels3 = []
        for idx, df in tqdm(data.iterrows()):
            label1 = [0. for _ in range(len(id2label_emotion))]
            label2 = [0. for _ in range(len(id2label_tempo))]
            label3 = [0. for _ in range(len(id2label_genre))]
            
            datas.append(remove_stop_word(df.caption))
            label1[label2id_emotion[df.emotion]] = 1.
            label2[label2id_tempo[df['tempo(category)']]] = 1.
            label3[label2id_genre[df['genre']]] = 1.
            self.labels1.append(label1)
            self.labels2.append(label2)
            self.labels3.append(label3)
        
        self.dataset =  tokenizer(datas,padding=True, truncation=True,max_length=512 ,return_tensors="pt").to('cuda')
        self.labels1= torch.tensor(self.labels1)
        self.labels2= torch.tensor(self.labels2)
        self.labels3= torch.tensor(self.labels3)

    def __len__(self):
        return len(self.labels1)
    
    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.dataset.items()}
        item['labels1'] = self.labels1[idx].clone().detach()
        item['labels2'] = self.labels2[idx].clone().detach()
        item['labels3'] = self.labels3[idx].clone().detach()
        return item
    
def remove_stop_word(string):
    stop_words_list = ['and','a','the','The','of','with','in','for']
    string = string.split()
    result = []
    for word in string: 
        if word not in stop_words_list: 
            result.append(word) 
    result = ' '.join(result)
    return result 

class frontModelDataset_front_sentence:
    def __init__(self, data, tokenizer, label_data_path ='./labels.pkl'):

        emotion_labels, tempo_labels, genre_labels= data_labels(label_data_path)
        
        id2label_emotion = {k:l for k, l in enumerate(emotion_labels)}
        label2id_emotion = {l:k for k, l in enumerate(emotion_labels)}
        id2label_tempo = {k:l for k, l in enumerate(tempo_labels)}
        label2id_tempo = {l:k for k, l in enumerate(tempo_labels)}
        id2label_genre = {k:l for k, l in enumerate(genre_labels)}
        label2id_genre = {l:k for k, l in enumerate(genre_labels)}

        self.tokenizer = tokenizer
        self.dataset = []
        datas = []
        self.labels1 = []
        self.labels2 = []
        self.labels3 = []
        sentences = ['Find the mood, genre, and tempo of the music']*len(data)
        for idx, df in tqdm(data.iterrows()):
            label1 = [0. for _ in range(len(id2label_emotion))]
            label2 = [0. for _ in range(len(id2label_tempo))]
            label3 = [0. for _ in range(len(id2label_genre))]
            datas.append(df.caption)
            label1[label2id_emotion[df.emotion]] = 1.
            label2[label2id_tempo[df['tempo(category)']]] = 1.
            label3[label2id_genre[df['genre']]] = 1.
            self.labels1.append(label1)
            self.labels2.append(label2)
            self.labels3.append(label3)
        
        self.dataset =  tokenizer(sentences, datas,padding=True, truncation=True,max_length=512 ,return_tensors="pt").to('cuda')
        self.labels1= torch.tensor(self.labels1)
        self.labels2= torch.tensor(self.labels2)
        self.labels3= torch.tensor(self.labels3)

    def __len__(self):
        return len(self.labels1)
    
    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.dataset.items()}
        item['labels1'] = self.labels1[idx].clone().detach()
        item['labels2'] = self.labels2[idx].clone().detach()
        item['labels3'] = self.labels3[idx].clone().detach()
        return item


if __name__ == '__main__':
    pass