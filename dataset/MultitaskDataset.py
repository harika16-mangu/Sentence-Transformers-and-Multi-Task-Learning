import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from sklearn.utils import resample

# data collator 

class BaseDataset(Dataset):
    def __init__(self, csv_file, model_name, text_column, label_column, task_id, max_samples_per_dataset=500):
        self.data = pd.read_csv(csv_file, encoding='unicode_escape')
        self.data.dropna(inplace=True)


        if max_samples_per_dataset:
            self.data = self._sample_data(self.data, label_column, max_samples_per_dataset)
        
        # try:
        #     print(self.data['Label'].value_counts())
        # except:
        #     print(self.data['sentiment'].value_counts())
        
        self.texts = self.data[text_column].tolist()
        self.labels = self.data[label_column].tolist()
        
        if label_column == 'sentiment':
            label_encoder = {'neutral': 0, 'negative': 1, 'positive': 2}
            self.labels = [label_encoder[label] for label in self.labels]

        self.task_id = task_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _sample_data(self, data, label_column, max_samples_per_dataset):
        grouped = data.groupby(label_column, group_keys=False)
        sampled_data = grouped.apply(lambda x: resample(x, n_samples=max_samples_per_dataset, random_state=42))
        return sampled_data.reset_index(drop=True)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        task = self.task_id

        inputs = self.tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors="pt")
        
        item = {key: val.squeeze(0) for key, val in inputs.items()}
        item['labels'] = torch.tensor(label, dtype=torch.long)
        item['task'] = torch.tensor(task, dtype=torch.long)
        
        return item
    

class CombinedDataset(Dataset):
    def __init__(self, sentiment_dataset, hate_speech_dataset):
        self.sentiment_dataset = sentiment_dataset
        self.hate_speech_dataset = hate_speech_dataset
        self.datasets = [self.sentiment_dataset, self.hate_speech_dataset]
        self.lengths = [len(d) for d in self.datasets]

    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, idx):
        if idx < self.lengths[0]:
            return self.sentiment_dataset[idx]
        else:
            return self.hate_speech_dataset[idx - self.lengths[0]]
        

if __name__ == "__main__":
    hate_speech_csv = "data/HateSpeechDatasetBalanced.csv"
    sentiment_csv = "data/train.csv"
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    hate_speech_dataset = BaseDataset(hate_speech_csv, model_name, "Content", "Label", task_id=0)
    sentiment_dataset = BaseDataset(sentiment_csv, model_name, "text", "sentiment", task_id=1)
    
    combined_dataset = CombinedDataset(sentiment_dataset, hate_speech_dataset)
    print(len(hate_speech_dataset), len(sentiment_dataset), len(combined_dataset))