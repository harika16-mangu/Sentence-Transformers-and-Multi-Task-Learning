import pandas as pd 
from sklearn.model_selection import train_test_split
if __name__=="__main__":
    hate_speech = pd.read_csv('data/HateSpeechDatasetBalanced.csv')
    hate_speech_train, hate_speech_test = train_test_split(hate_speech, test_size=.3, random_state=42)
    hate_speech_train.to_csv('data/hate_speech_train.csv')
    hate_speech_test.to_csv('data/hate_speech_test.csv')