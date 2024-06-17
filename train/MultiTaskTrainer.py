from transformers import TrainingArguments, Trainer
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from models import MultitaskModel
from dataset import BaseDataset, CombinedDataset
import torch
def multi_task_train():
    hate_speech_train_csv = "data/hate_speech_train.csv"
    hate_speech_test_csv = "data/hate_speech_test.csv"
    sentiment_train_csv = "data/train.csv"
    sentiment_test_csv = "data/test.csv"
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    output_directory = "output"
    
    # Initialize datasets
    hate_speech_train_dataset = BaseDataset(hate_speech_train_csv, model_name, "Content", "Label", task_id=0)
    sentiment_train_dataset = BaseDataset(sentiment_train_csv, model_name, "text", "sentiment", task_id=1)
    
    combined_train_dataset = CombinedDataset(hate_speech_train_dataset, sentiment_train_dataset)
    
    hate_speech_test_dataset = BaseDataset(hate_speech_test_csv, model_name, "Content", "Label", task_id=0)
    sentiment_test_dataset = BaseDataset(sentiment_test_csv, model_name, "text", "sentiment", task_id=1)
    
    combined_test_dataset = CombinedDataset(hate_speech_test_dataset, sentiment_test_dataset)
    # Initialize model
    model = MultitaskModel(model_name)

    # Training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=1,
        learning_rate=0.0001,
        num_train_epochs=3,
        logging_dir='./logs',
        logging_steps=100,
        save_steps=500,
        evaluation_strategy="epoch",
        remove_unused_columns=False,
        output_dir=output_directory
    )
    
    # Function to compute metrics
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc}
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=combined_train_dataset,
        eval_dataset=combined_test_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Start training
    trainer.train()
    torch.save(model.state_dict(), "output/pytorch_model.bin")


if __name__ == "__main__":
    multi_task_train()
