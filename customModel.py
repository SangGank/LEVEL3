from tqdm import tqdm
from transformers import BertForSequenceClassification, BertModel, BertPreTrainedModel
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import SequenceClassifierOutput
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
import torch


def get_weighted_loss(loss_fct, inputs, labels, weights):
    loss = 0.0
    for i in range(weights.shape[0]):
        loss += (weights[i] + 1.0) * loss_fct(inputs[i:i + 1], labels[i:i + 1])

    return loss / (sum(weights) + weights.shape[0])


# class CustomBertForSequenceClassification(BertPreTrainedModel):

#     def __init__(self, config):
#         super(CustomBertForSequenceClassification, self).__init__(config)
#         self.num_labels1 = config.num_labels1
#         self.num_labels2 = config.num_labels2
#         self.num_labels3 = config.num_labels3

#         self.bert = BertModel(config)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.layer_1 = nn.Linear(config.hidden_size, self.config.num_labels1)
#         self.layer_2 = nn.Linear(config.hidden_size, self.config.num_labels2)
#         # self.layer_3 = nn.Linear(config.hidden_size, self.config.num_labels3)

#         self.init_weights()

#     def forward(self, input_ids, attention_mask=None, token_type_ids=None,
#                 position_ids=None, head_mask=None,label1=None , label2=None , label3=None, weights=None):

#         outputs = self.bert(input_ids,
#                             attention_mask=attention_mask,
#                             token_type_ids=token_type_ids,
#                             position_ids=position_ids,
#                             head_mask=head_mask)

#         pooled_output = outputs[1]

#         pooled_output = self.dropout(pooled_output)

        
#         logits1 = self.layer_1(pooled_output)
#         logits2 = self.layer_2(pooled_output)
#         # logits3 = self.layer_3(pooled_output)
        

#         # outputs1 = (logits1,) + outputs[2:]  # add hidden states and attention if they are here
#         # outputs2 = (logits2,) + outputs[2:]  # add hidden states and attention if they are here
#         # outputs3 = (logits3,) + outputs[2:]  # add hidden states and attention if they are here

#         # if (label1 is not None) & (label2 is not None) &(label3 is not None) :
#         if (label1 is not None) & (label2 is not None):
#             if self.num_labels == 1:
#                 #  We are doing regression
#                 loss_fct = MSELoss()
#                 loss1 = loss_fct(logits1.view(-1), label1.view(-1))
#                 loss2 = loss_fct(logits2.view(-1), label2.view(-1))
#                 # loss3 = loss_fct(logits3.view(-1), label3.view(-1))
#             else:
#                 loss_fct = CrossEntropyLoss()
#                 if weights is None:
#                     loss1 = loss_fct(logits1.view(-1), label1.view(-1))
#                     loss2 = loss_fct(logits2.view(-1), label2.view(-1))
#                     # loss3 = loss_fct(logits3.view(-1), label3.view(-1))
#                 else:
#                     loss1 = get_weighted_loss(loss_fct,
#                                              logits1.view(-1, self.num_labels1),
#                                              label1.view(-1), weights)
#                     loss2 = get_weighted_loss(loss_fct,
#                                              logits2.view(-1, self.num_labels2),
#                                              label2.view(-1), weights)
#                     # loss3 = get_weighted_loss(loss_fct,
#                     #                          logits3.view(-1, self.num_labels3),
#                     #                          label3.view(-1), weights)
#             loss = loss1 + loss2  #+ loss3
#             # outputs = (loss,) + (logits1,) + (logits2,) + (logits3,) + outputs[2:]
#             outputs = (loss,) + (logits1,) + (logits2,) + outputs[2:]

#         return outputs  # (loss), logits, (hidden_states), (attentions)
    
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

    # @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # @add_code_sample_docstrings(
    #     checkpoint=_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION,
    #     output_type=SequenceClassifierOutput,
    #     config_class=_CONFIG_FOR_DOC,
    #     expected_output=_SEQ_CLASS_EXPECTED_OUTPUT,
    #     expected_loss=_SEQ_CLASS_EXPECTED_LOSS,
    # )
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

