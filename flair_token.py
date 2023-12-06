import numpy as np
import pandas as pd
from flair.data import Sentence
from flair.models import SequenceTagger

def load_classifier_model():
    classifier = SequenceTagger.load('ner-fast')
    return classifier

def load_tagger_model():
    tagger = SequenceTagger.load('upos-fast')
    return tagger

classifier = load_classifier_model()
tagger = load_tagger_model()

async def tokenise_loc(sentence_rec):
    query_string = sentence_rec
    sentence = Sentence(query_string)
    loss = classifier.predict((sentence), return_loss=True)

    df=[]
    
    for label in sentence.get_labels():
        df.append([label.data_point.text, label.value])
    
    return df

async def get_possible_loc_tokens(sentence_rec):
    query_string = sentence_rec
    sentence = Sentence(query_string)
    loss = tagger.predict((sentence), return_loss=True)

    df=[]

    for label in sentence.get_labels():
        if len(label.data_point.text)>=3 and (label.value=="PROPN" or label.value=="X"):
            df.append([label.data_point.text, label.value])

    return df