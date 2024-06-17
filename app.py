from flask import Flask, request, jsonify
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score
from models import MultitaskModel
from train import multi_task_train  # Assuming CustomTrainer is defined in a separate file as shown earlier
import torch
import os

app = Flask(__name__)


output_directory = "output"

if not os.path.isfile(f"{output_directory}/pytorch_model.bin"):
    print("Training!")
    multi_task_train()
# Getting model ready
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
model = MultitaskModel(model_name)
model.load_state_dict(torch.load(f"{output_directory}/pytorch_model.bin"))
model.eval()

# Get tokenizer ready
tokenizer = AutoTokenizer.from_pretrained(model_name)

@app.route('/train', methods=['POST'])
def train_model():
    multi_task_train()
    return jsonify({"message": "Training completed and model saved."})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    sentences = data["sentences"]
    task = data["task"]  # 0 for hate speech, 1 for sentiment analysis

    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask, task=[task])
        logits = outputs['logits']
        predictions = torch.argmax(logits, dim=-1).tolist()

    return jsonify({"predictions": predictions})


if __name__ == "__main__":
    # docker run -p 5500:5000 multitask-flask-app

    app.run(host='0.0.0.0', port=5500)