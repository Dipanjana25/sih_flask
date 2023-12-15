from flair.data import Sentence
from flair.models import SequenceTagger
import torch, flair

flair.device="cpu"
torch.device("cpu")

def load_classifier_model():
    classifier = SequenceTagger.load('data/xlr-tuned.pt')
    # classifier = SequenceTagger.load('ner')
    return classifier

classifier = load_classifier_model()

def tokenise_loc(sentence_rec):
    query_string = sentence_rec
    sentence = Sentence(query_string)
    classifier.predict(sentence)
    # print(sentence)
    df=[]
    for label in sentence.get_labels():
        df.append((label.data_point.text, label.value))
    
    return df
