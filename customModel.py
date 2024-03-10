from tqdm import tqdm
from transformers import (BertForSequenceClassification, BertModel, BertPreTrainedModel, AutoTokenizer,
                          AutoConfig, RobertaPreTrainedModel, RobertaModel, ElectraPreTrainedModel,
                          ElectraModel, GPT2PreTrainedModel, GPT2Model)
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from typing import List, Optional, Tuple, Union
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
        if labels1 is not None:
            if self.config.problem_type is None:
                if self.num_labels1 == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels1 > 1 and (labels1.dtype == torch.long or labels1.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels1 == 1:
                    loss1 = loss_fct(logits1.squeeze(), labels1.squeeze())
                else:
                    loss1 = loss_fct(logits1, labels1)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss1 = loss_fct(logits1.view(-1, self.num_labels1), labels1.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss1 = loss_fct(logits1, labels1)

        if labels2 is not None:
            if self.config.problem_type is None:
                if self.num_labels2 == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels2 > 1 and (labels2.dtype == torch.long or labels2.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels2 == 1:
                    loss2 = loss_fct(logits2.squeeze(), labels2.squeeze())
                else:
                    loss2 = loss_fct(logits2, labels2)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss2 = loss_fct(logits2.view(-1, self.num_labels2), labels2.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss2 = loss_fct(logits2, labels2)
        
        if labels3 is not None:
            if self.config.problem_type is None:
                if self.num_labels3 == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels3 > 1 and (labels3.dtype == torch.long or labels3.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels3 == 1:
                    loss3 = loss_fct(logits3.squeeze(), labels3.squeeze())
                else:
                    loss3 = loss_fct(logits3, labels3)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss3 = loss_fct(logits3.view(-1, self.num_labels3), labels3.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss3 = loss_fct(logits3, labels3)
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
        if labels1 is not None:
            if self.config.problem_type is None:
                if self.num_labels1 == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels1 > 1 and (labels1.dtype == torch.long or labels1.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels1 == 1:
                    loss1 = loss_fct(logits1.squeeze(), labels1.squeeze())
                else:
                    loss1 = loss_fct(logits1, labels1)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss1 = loss_fct(logits1.view(-1, self.num_labels1), labels1.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss1 = loss_fct(logits1, labels1)

        if labels2 is not None:
            if self.config.problem_type is None:
                if self.num_labels2 == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels2 > 1 and (labels2.dtype == torch.long or labels2.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels2 == 1:
                    loss2 = loss_fct(logits2.squeeze(), labels2.squeeze())
                else:
                    loss2 = loss_fct(logits2, labels2)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss2 = loss_fct(logits2.view(-1, self.num_labels2), labels2.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss2 = loss_fct(logits2, labels2)
        
        if labels3 is not None:
            if self.config.problem_type is None:
                if self.num_labels3 == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels3 > 1 and (labels3.dtype == torch.long or labels3.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels3 == 1:
                    loss3 = loss_fct(logits3.squeeze(), labels3.squeeze())
                else:
                    loss3 = loss_fct(logits3, labels3)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss3 = loss_fct(logits3.view(-1, self.num_labels3), labels3.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss3 = loss_fct(logits3, labels3)
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
        if labels1 is not None:
            if self.config.problem_type is None:
                if self.num_labels1 == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels1 > 1 and (labels1.dtype == torch.long or labels1.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels1 == 1:
                    loss1 = loss_fct(logits1.squeeze(), labels1.squeeze())
                else:
                    loss1 = loss_fct(logits1, labels1)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss1 = loss_fct(logits1.view(-1, self.num_labels1), labels1.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss1 = loss_fct(logits1, labels1)

        if labels2 is not None:
            if self.config.problem_type is None:
                if self.num_labels2 == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels2 > 1 and (labels2.dtype == torch.long or labels2.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels2 == 1:
                    loss2 = loss_fct(logits2.squeeze(), labels2.squeeze())
                else:
                    loss2 = loss_fct(logits2, labels2)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss2 = loss_fct(logits2.view(-1, self.num_labels2), labels2.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss2 = loss_fct(logits2, labels2)
        
        if labels3 is not None:
            if self.config.problem_type is None:
                if self.num_labels3 == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels3 > 1 and (labels3.dtype == torch.long or labels3.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels3 == 1:
                    loss3 = loss_fct(logits3.squeeze(), labels3.squeeze())
                else:
                    loss3 = loss_fct(logits3, labels3)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss3 = loss_fct(logits3.view(-1, self.num_labels3), labels3.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss3 = loss_fct(logits3, labels3)
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
        labels1: Optional[torch.LongTensor] = None,
        labels2: Optional[torch.LongTensor] = None,
        labels3: Optional[torch.LongTensor] = None,
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
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1
                # logger.warning(
                #     f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                #     "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                # )

        pooled_logits = logits1[torch.arange(batch_size, device=logits1.device), sequence_lengths]
        pooled_logits = logits2[torch.arange(batch_size, device=logits2.device), sequence_lengths]
        pooled_logits = logits3[torch.arange(batch_size, device=logits3.device), sequence_lengths]

        
        loss1 = None
        loss2 = None
        loss3 = None
        if labels1 is not None:
            if self.config.problem_type is None:
                if self.num_labels1 == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels1 > 1 and (labels1.dtype == torch.long or labels1.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels1 == 1:
                    loss1 = loss_fct(logits1.squeeze(), labels1.squeeze())
                else:
                    loss1 = loss_fct(logits1, labels1)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss1 = loss_fct(logits1.view(-1, self.num_labels1), labels1.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss1 = loss_fct(logits1, labels1)

        if labels2 is not None:
            if self.config.problem_type is None:
                if self.num_labels2 == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels2 > 1 and (labels2.dtype == torch.long or labels2.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels2 == 1:
                    loss2 = loss_fct(logits2.squeeze(), labels2.squeeze())
                else:
                    loss2 = loss_fct(logits2, labels2)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss2 = loss_fct(logits2.view(-1, self.num_labels2), labels2.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss2 = loss_fct(logits2, labels2)
        
        if labels3 is not None:
            if self.config.problem_type is None:
                if self.num_labels3 == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels3 > 1 and (labels3.dtype == torch.long or labels3.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels3 == 1:
                    loss3 = loss_fct(logits3.squeeze(), labels3.squeeze())
                else:
                    loss3 = loss_fct(logits3, labels3)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss3 = loss_fct(logits3.view(-1, self.num_labels3), labels3.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss3 = loss_fct(logits3, labels3)
        loss =None
        if loss1 and loss2 and loss3:
            loss = loss1 + loss2 + loss3
            
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )






if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = pd.read_csv('./genre_seung.csv')

    id2label_emotion = {k:l for k, l in enumerate(data.emotion.unique())}
    label2id_emotion = {l:k for k, l in enumerate(data.emotion.unique())}
    id2label_tempo = {k:l for k, l in enumerate(data['tempo(category)'].unique())}
    label2id_tempo = {l:k for k, l in enumerate(data['tempo(category)'].unique())}
    id2label_genre = {k:l for k, l in enumerate(data['genre'].unique())}
    label2id_genre = {l:k for k, l in enumerate(data['genre'].unique())}
    BASE_MODEL = 'bert-base-uncased'

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    config = AutoConfig.from_pretrained(BASE_MODEL)
    config.num_labels1 = len(id2label_emotion)
    config.num_labels2 = len(id2label_tempo)
    config.num_labels3 = len(id2label_genre)
    model = customBertForSequenceClassification.from_pretrained(BASE_MODEL, config= config).to(device)
