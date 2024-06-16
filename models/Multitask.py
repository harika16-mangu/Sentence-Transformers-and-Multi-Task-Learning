from transformers import AutoModel
from torch import nn


class MultitaskModel(nn.Module):
    def __init__(self, model_name):
        super(MultitaskModel, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.classifier_hate_speech = nn.Linear(self.bert.config.hidden_size, 2)  # Assuming binary classification
        self.classifier_sentiment = nn.Linear(self.bert.config.hidden_size, 3)   # Assuming three classes for sentiment

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None, task=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs[1]

        if task[0] == 0:  # Hate Speech
            logits = self.classifier_hate_speech(pooled_output)
        elif task[0] == 1:  # Sentiment Analysis
            logits = self.classifier_sentiment(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return {'loss': loss, 'logits': logits}