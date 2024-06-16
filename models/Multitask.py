from transformers import AutoModel
from torch import nn

class MultitaskModel(nn.Module):
    # Each batch can *only* have one task 
    
    def __init__(self, model_name):
        super(MultitaskModel, self).__init__()
        self.sentence_transformer = AutoModel.from_pretrained(model_name)
        self.classifier_hate_speech = nn.Linear(self.sentence_transformer.config.hidden_size, 2)  # Assuming binary classification
        self.classifier_sentiment = nn.Linear(self.sentence_transformer.config.hidden_size, 3)   # Assuming three classes for sentiment

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None, task=None):
        outputs = self.sentence_transformer(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
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
    
if __name__ == "__main__":
    from transformers import AutoTokenizer
    import torch
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    model = MultitaskModel(model_name)
    
    # Dummy inputs
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sentences = [
        "I hate this product, it's terrible.",
        "This movie is really good!",
        "This book is average."
    ]
    # 2 -> 2 passes 
    labels_hate_speech = torch.tensor([1, 0, 0])  # Binary labels for hate speech
    labels_sentiment = torch.tensor([0, 2, 1])    # Multi-class labels for sentiment analysis
    
    # Tokenize sentences
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    
    # Assuming the same inputs for both tasks
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    # Prepare task labels
    task = [0]
    
    # Forward pass
    outputs_hate = model(input_ids, attention_mask, task=task, labels = labels_hate_speech)
    
    
    task = [1]
    outputs_sentiment = model(input_ids, attention_mask, task=task, labels = labels_sentiment)
    
    # Compute loss and logits
    print("FOR HATE SPEECH")
    loss = outputs_hate['loss']
    logits = outputs_hate['logits']
    
    print(f"Loss: {loss}")
    print(f"Logits: {logits}")


    print("FOR SENTIMENT ANALYSIS")

    loss = outputs_sentiment['loss']
    logits = outputs_sentiment['logits']
    
    print(f"Loss: {loss}")
    print(f"Logits: {logits}")

    