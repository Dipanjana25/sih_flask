import numpy as np
import pandas as pd
from flair.data import Sentence
from flair.models import SequenceTagger

def load_flair_model():
    classifier = SequenceTagger.load('ner-fast')
    return classifier

classifier = load_flair_model()

def handle_click(sentence_rec):
    query_string = sentence_rec
    sentence = Sentence(query_string)
    loss = classifier.predict((sentence), return_loss=True)
    processed_string = sentence.to_tagged_string()
    ner_dict = sentence.to_dict(tag_type='ner')
    ner_df = pd.DataFrame(ner_dict['entities'])
    text = ner_df['text'].values
    python_list =text.tolist()

    return processed_string


if __name__ == "__main__":
    handle_click()