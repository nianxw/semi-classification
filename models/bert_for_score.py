import torch.nn as nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel
from transformers import BertModel


class BertScore(BertPreTrainedModel):
    def __init__(self, config):
        super(BertScore, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.fc = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self, batch):
        sequence_output = self.bert(batch[0], batch[1], batch[3], batch[2])[1]
        sequence_output = self.dropout(sequence_output)
        logits = self.fc(sequence_output)
        probs = F.sigmoid(logits)
        preds = probs > 0.5

        return probs, preds.long()