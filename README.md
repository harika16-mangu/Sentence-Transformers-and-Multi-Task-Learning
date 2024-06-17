# Sentence-Transformers-and-Multi-Task-Learning

## Table of Contents
1. [Introduction](#1-introduction)
2. [Project Structure](#2-project-structure)
3. [Datasets](#3-datasets)
4. [Implementation](#4-implementation)
5. [Algorithms and Evaluation Metrics](#5-algorithms-and-evaluation-metrics)
6. [Results](#6-results)
7. [Future Scope](#7-future-scope)
8. [Conclusion](#8-conclusion)

### Questions:
1. [Task 3](#-tasks-3)

2. [Task 4](#-tasks-4)

## Setup and Installation (Docker):
1. Download the dataset from Kaggle datasource -[Hate speech](https://www.kaggle.com/datasets/waalbannyantudre/hate-speech-detection-curated-dataset), [Sentiment](https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset) and put it in the data folder.
2. Run the script to create Hate speech train and test.
```python script/split_train_test.py```
3. Build Docker image: ```docker build -t multitask-flask-app```
4. Run the Docker image: ```docker run -p 5000:5000 multitask-flask-app```
5. You can use post http://localhost:5000/train to retrain the model http://localhost:5000/predict with json body ```{ "sentences": ["sentence1","sentence2"], "task":1}``` to predict using the model.

## Setup and Installation (Flask):
1. Download the dataset from Kaggle datasource -[Hate speech](https://www.kaggle.com/datasets/waalbannyantudre/hate-speech-detection-curated-dataset), [Sentiment](https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset) and put it in the data folder.
2. Create the requirements ``` pip install -r requirements.txt ``` 
3. Run the script to create Hate speech train and test.
```python script/split_train_test.py```
4. Run ```python app.py```
5. You can use post http://localhost:5000/train to retrain the model http://localhost:5000/predict with json body ```{ "sentences": ["sentence1","sentence2"], "task":1}``` to predict using the model.

## 1. Introduction:
This project aims to implement,train,and optmize a neural network architecture particularly focusing on sentence transformers using the model sentence-transformer(all-MiniLM-L12-v2) from Hugging Face and multi task extensions which included text classification and Sentiment Analysis.


## 2.Project Structure
- `data`: Directory containing the dataset for training and testing.
- `dataset` 
    
    ----`MultitaskDataset.py`:- `BaseDataset`&`CombinedDataset` classes:Implemented Custom PyTorch Dataset for processing & sampling sentiment and hate speech data, facilitating easy tokenization and dataset handling for multi-task learning.
-  `main`
      
    ----`test_model.py`: Script for testing the model to perform sentence transformation using pre-trained transformers from Hugging Face, including visualization and cosine similarity computation.
- `models`

    ----`Multitask.py`: Defines a multi-task neural network model using a pre-trained transformer backbone, designed to handle both hate speech detection and sentiment analysis within a unified architecture.
- `output`: Directory for saving model outputs and visualizations.
- `script`

    ----`split_train_test.py`:- Split the hate speech dataset into training and test sets, ensuring a balanced distribution, and saved them as separate CSV files.
- `train`: Directory for training the Multitask and Custom multitask model

    ----`Multitasktrainer`:- Implemented and trained a multi-task model for hate speech detection and sentiment analysis using the Hugging Face Trainer API with combined datasets.

    ----`CustomMultitasktrainer`:- Implemented a custom multi-task training loop using a `CustomTrainer` class that applies layer-wise learning rates for better optimization of combined hate speech and sentiment analysis tasks.
- `venv`: - Created a virtual environment (`venv`) to manage dependencies and ensure a consistent and isolated Python environment for the multi-task training project.
- `app.py`: - Implemented a Flask API to serve predictions and retrain the multitask model on demand, using routes for model training and inference based on task type (hate speech detection or sentiment analysis).
- `Dockerfile`:- Set up a Docker container with Python, installed required dependencies, exposed port 5000, configured Flask environment variables, and launched the Flask application.


## 3.Datasets 
[Hate speech dataset](https://www.kaggle.com/datasets/waalbannyantudre/hate-speech-detection-curated-dataset) for Text Classification (Task-1) and [Twitter sentiment dataset](https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset) analyis for Sentiment analysis (Task 2)


## 4.Implementation
- Implemented BaseDataset and CombinedDataset classes using pandas for data loading and preprocessing.Balanced and sampled datasets for hate speech and sentiment analysis using sklearn.
- Developed a MultitaskModel utilizing Hugging Face's AutoModel for sentence embeddings.Incorporated separate classifiers for hate speech detection and sentiment analysis tasks.
- Utilized PyTorch for model training and evaluation.Implemented training pipeline using Trainer from Hugging Face's transformers.Customized training with CustomTrainer to adjust layer-wise learning rates and optimize performance.
- Set up a Flask web application for model deployment.Created endpoints for model training and prediction (/train and /predict).
- Utilized Docker for containerization, ensuring portability and scalability of the application.

## 5.Algorithms and Evaluation metrics:
Multitask Learning with Transformers: Specifically, utilized the sentence-transformers library which implements transformer-based architectures fine-tuned for sentence embeddings.
 
Evaluation Metrics Used: Accuracy Score, Cross-Entropy Loss

## 6.Results: 
Project effectively addresses the challenges of multitask learning in natural language processing, ensuring robust performance in hate speech detection and sentiment analysis tasks through careful model architecture design and evaluation strategies.

## 7. Future Scope:
1. Grid Search for Hyperparameter Tuning: Grid Search for Hyperparameter Tuning involves systematically exploring combinations of parameters using tools like GridSearchCV from sklearn or custom scripts to optimize factors such as learning rate, batch size, and transformer model configurations, ultimately enhancing model performance by fine-tuning hyperparameters to improve accuracy and convergence speed across multitask learning objectives.
2. Scalability and Computational Resources: To enhance training efficiency for handling larger datasets and increasing model capacity, we can implement distributed training across multiple GPUs or leverage cloud services (e.g., AWS, Google Cloud) for scalable compute resources.
3. Model Interpretability and Explainability: To enhance understanding and trust in model predictions for hate speech and sentiment analysis, we can employ attention mechanisms to visualize which parts of input sentences influence predictions and conduct feature importance analysis to interpret model decisions, crucial for informed decision-making
4. Parameter efficient fine tunning 

## 8.Conclusion
By focusing on these enhancements, the project aims to push the boundaries of multitask learning in natural language processing, leveraging advanced techniques to improve performance, scalability, and interpretability for hate speech detection and sentiment analysis tasks in diverse and dynamic environments.



# Tasks 3:
## Training Considerations
Freezing Strategies:
1. If the entire network should be frozen.
Freezing the Entire Network:

 **Answer**

 No parameters are updated during training, making the model unable to adapt to new data.

Advantage: Useful when using a pre-trained model that performs well on similar tasks, reducing computational resources.

Rationale: Suitable if the pre-trained model's weights are highly relevant and should not be altered.


2. Freezing Only the Transformer Backbone:

**Answer**

 Only the lower-level features learned by the transformer are frozen, allowing task-specific heads to adapt to new data.

Advantage: Retains flexibility for fine-tuning task-specific layers while leveraging pre-trained representations.Catastrophic forgetting is also avoided by freezing the transfomer layer.

Rationale: Beneficial when the lower-level features are generalizable and task-specific nuances need adaptation.

3. If only one of the task-specific heads (either for Task A or Task B) should be frozen.
Freezing One Task-Specific Head:

**Answer** 

Freezes either the hate speech or sentiment analysis head while allowing the other to adapt.

Advantage: Maintains specialized learning for one task while still benefiting from transfer learning on the other.

Rationale: Appropriate when one task has more available data or is more critical, optimizing resource allocation.


## Transfer Learning Approach:
Choice of Pre-trained Model:

Model Selection: Use "sentence-transformers/all-MiniLM-L6-v2" due to its balanced performance across various NLP tasks and speed/ storage efficiency in multitask learning.
![image info](Images\Model_performance.png)


Layers to Freeze/Unfreeze:

1. Freeze: Initially freeze all layers to leverage the pre-trained weights and prevent overfitting on limited data.
2. Unfreeze: Gradually unfreeze layers starting from the top of the transformer backbone and task-specific heads during fine-tuning.
3. The Rationale Behind Choices:
Freezing Strategy: Initially freeze to retain general features and gradually unfreeze to adapt to specific task requirements, balancing between utilizing pre-trained knowledge and adapting to new data.

Transfer Learning Benefits: Facilitates faster convergence, improved accuracy, and better generalization by leveraging knowledge from large-scale pre-training while adapting to task-specific nuances.

By implementing these strategies, the model can efficiently leverage transfer learning to enhance performance on hate speech and sentiment analysis tasks while optimizing computational resources and training efficiency.


# Task 4: Layer-wise Learning Rate Implementation (BONUS)

Explain the rationale for the specific learning rates you've set for each layer.

Describe the potential benefits of using layer-wise learning rates for training deep neural networks. Does the multi-task setting play into that?

#### Layer-wise Learning Rates:

The leaning rate set for the training model of multitask transformer is 0.0001 the base learning rate for the first layer is 1e-5 and it has a subsequent decay rate of 0.95. These are the specific reasons:
   - Stability: Deeper layers which are more closer to the input layer will capture more generic features that are broadly applicable across tasks. By assigning them lower learning rates, we ensure these valuable representations are not disrupted significantly.
   - Flexibility: Higher learning rates for upper layers allow them to specialize more quickly for the specific tasks at hand, such as hate speech and sentiment analysis.


#### Benefits of Layer-wise Learning Rates in Multi-task Setting Considerations:

1. Balanced Adaptation:
   - In a multi-task learning scenario, different tasks may require different adaptations. Layer-wise learning rates provide a mechanism to balance the stability of shared layers while allowing sufficient learning capacity for task-specific layers.

2. Task-specific Requirements:
   - The upper layers, being closer to the task-specific outputs, need to be more flexible to capture the nuances of each task (hate speech and sentiment analysis). Therefore, applying the base learning rate with less decay helps them adapt effectively.

3. Stabilized Training:
   - Prevents large updates to the lower layers, maintaining the integrity of the pre-trained model’s capabilities while avoiding catastrophic forgetting.







