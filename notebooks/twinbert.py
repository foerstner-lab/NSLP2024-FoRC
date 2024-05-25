"""
SPDX-FileCopyrightText: 2024 ZB MED - Information Centre for Life Sciences
SPDX-FileCopyrightText: 2024 Benjamin Wolff

SPDX-License-Identifier: MIT
"""

import torch
from torch import nn
from transformers.models.bert import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import SequenceClassifierOutput
from typing import Optional, Union, Tuple


class TwinBertForSequenceClassification(BertPreTrainedModel):
    """Custom TwinBERT Model

    Bert Model that utilizes two Models, one for title+abstract, another for Enrichment
    Embeddings are concatenated and fed to the dense-layer
    """

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.bert_1 = BertModel(config)  # Titel+Abstract
        self.bert_2 = BertModel(config)  # Collected enrichments
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.intermediate = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids_1: Optional[torch.Tensor] = None,
        attention_mask_1: Optional[torch.Tensor] = None,
        token_type_ids_1: Optional[torch.Tensor] = None,
        input_ids_2: Optional[torch.Tensor] = None,
        attention_mask_2: Optional[torch.Tensor] = None,
        token_type_ids_2: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        # Bert 1 (titel + text)
        outputs_1 = self.bert_1(
            input_ids_1,
            attention_mask=attention_mask_1,
            token_type_ids=token_type_ids_1,
        )
        pooled_output_1 = outputs_1[1]

        # Bert 2 (additional data like concepts, categories, research fields,...)
        outputs_2 = self.bert_2(
            input_ids_2,
            attention_mask=attention_mask_2,
            token_type_ids=token_type_ids_2,
        )
        pooled_output_2 = outputs_2[1]

        # Concatenate pooled output
        pooled_output = torch.cat((pooled_output_1, pooled_output_2), dim=1)

        # Dropout
        pooled_output = self.dropout(pooled_output)

        # Reduce Dimensionality
        pooled_output_intermediate = self.intermediate(pooled_output)

        # Get logits
        logits = self.classifier(pooled_output_intermediate)

        loss = None
        if labels is not None:
            # Set loss function and calculate loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )
