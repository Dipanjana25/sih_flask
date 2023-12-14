import numpy as np
import pandas as pd
from flair.data import Sentence
from flair.models import SequenceTagger
import torch, flair

flair.device="cpu"
torch.device("cpu")

def load_classifier_model():
    classifier = SequenceTagger.load('data/final-model.pt')
    # classifier = SequenceTagger.load('ner')
    return classifier

classifier = load_classifier_model()

def tokenise_loc(sentence_rec):
    query_string = sentence_rec
    sentence = Sentence(query_string)
    classifier.predict(sentence)
    print(sentence)
    # ner_dict=sentence.to_dict(tag_type='ner')
    # ner_df = pd.DataFrame(ner_dict)
    # print(ner_df)
    df=[]
    for label in sentence.get_labels():
        df.append((label.data_point.text, label.value))
    
    return df

# async def get_possible_loc_tokens(sentence_rec):
#     query_string = sentence_rec
#     sentence = Sentence(query_string)
#     loss = tagger.predict((sentence), return_loss=True)

#     doc = nlp(sentence_rec)

#     df=[]
#     for token in doc:
#         # print(token.text, token.pos_)
#         if len(token.text)>=4 and (token.pos_=="PROPN" or token.pos_=="NOUN" or token.pos_=="X"):
#             df.append((token.text, token.pos_))

#     # for label in sentence.get_labels():
#     #     if len(label.data_point.text)>=3 and (label.value=="PROPN" or label.value=="X"):
#     #         df.append((label.data_point.text, label.value))

#     return df