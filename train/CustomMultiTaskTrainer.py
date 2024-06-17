from transformers import Trainer, AdamW
from torch.optim import AdamW
from transformers import TrainingArguments
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from models import MultitaskModel
from dataset import BaseDataset, CombinedDataset
import torch

class CustomTrainer(Trainer):
    def __init__(self, *args, base_lr=1e-5, lr_decay=0.95, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_lr = base_lr
        self.lr_decay = lr_decay

    def create_optimizer(self):
        optimizer_grouped_parameters = self.get_layerwise_learning_rates(self.model, self.base_lr, self.lr_decay)
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.base_lr)
        return self.optimizer

    def get_layerwise_learning_rates(self, model, base_lr, lr_decay):
        # Change Learning rate for each layer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = []
        layers = [model.sentence_transformer.embeddings, *model.sentence_transformer.encoder.layer]

        for layer_num, layer in enumerate(layers):
            layer_lr = base_lr * (lr_decay ** layer_num)
            for name, param in layer.named_parameters():
                if any(nd in name for nd in no_decay):
                    optimizer_grouped_parameters.append({
                        "params": param,
                        "lr": layer_lr,
                        "weight_decay": 0.0,
                    })
                else:
                    optimizer_grouped_parameters.append({
                        "params": param,
                        "lr": layer_lr,
                        "weight_decay": 0.01,
                    })

        classifier_params = [model.classifier_hate_speech, model.classifier_sentiment]
        for classifier in classifier_params:
            for name, param in classifier.named_parameters():
                optimizer_grouped_parameters.append({
                    "params": param,
                    "lr": base_lr,
                    "weight_decay": 0.01,
                })

        return optimizer_grouped_parameters
    
def custom_multi_task_train():
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

    # Initialize CustomTrainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=combined_train_dataset,
        eval_dataset=combined_test_dataset,
        compute_metrics=compute_metrics,
        base_lr=1e-5,
        lr_decay=0.95,
    )
    
    # Start training
    trainer.train()

    torch.save(model.state_dict(), "output/pytorch_model.bin")

if __name__ == "__main__":
   custom_multi_task_train()
