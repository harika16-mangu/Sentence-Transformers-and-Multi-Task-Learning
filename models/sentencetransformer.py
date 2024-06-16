# models/sentence_transformer.py

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

class SentenceTransformer():
    def __init__(self, model_name='bert-base-uncased',max_length=12):
        """
        Initialize the SentenceTransformer with a pre-trained DistilBERT model.
        
        Args:
            model_name (str): The name of the pre-trained DistilBERT model to use.
        """
        # Load the tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.max_length = max_length
        
    def encode(self, sentences):
        """
        Encode the input sentences into fixed-length embeddings.
        Args:
            sentences (list of str): The input sentences to encode.
            max_length (int): The maximum length of the tokenized input sequences.
        Returns:
            torch.Tensor: The embeddings for the input sentences.
        """
        # Tokenize the input sentences
        inputs = self.tokenizer(sentences, return_tensors='pt', truncation=True, padding=True, max_length=self.max_length)
        
        # Disable gradient calculation for inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Extract the embeddings for the [CLS] token
        embeddings = outputs.last_hidden_state[:, 0, :]  # DistilBERT does not have [CLS], use the first token's embedding
        return embeddings
    