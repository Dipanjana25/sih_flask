from flair.data import Sentence
from flair.models import SequenceTagger
import torch, flair

flair.device="cpu"
torch.device("cpu")

def load_classifier_model():
    classifier = SequenceTagger.load('data/final-model.pt')
    # classifier = SequenceTagger.load('ner')
    return classifier

def load_lstm_model():
    lstm = SequenceTagger.load("data/lstm.pt")
    return lstm

classifier = load_classifier_model()
lstm = load_lstm_model()

def tokenise_loc(sentence_rec):
    query_string = sentence_rec
    sentence1 = Sentence(query_string)
    classifier.predict(sentence1)

    print(sentence1.to_tagged_string())
    df=[]
    for label in sentence1.get_labels():
        df.append((label.data_point.text, label.value))
    
    return df

def tokenise_loc_2(sentence_rec):
    query_string = sentence_rec
    sentence2 = Sentence(query_string)
    lstm.predict(sentence2)
    print(sentence2.to_tagged_string())
    df=[]
    for label in sentence2.get_labels():
        df.append((label.data_point.text, label.value))
    
    return df