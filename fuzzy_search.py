import numpy as np
import pandas as pd
from annoy import AnnoyIndex
from flair.embeddings import TransformerWordEmbeddings
from flair.data import Sentence
from rapidfuzz import fuzz
from pyphonetics import RefinedSoundex

df=pd.read_csv("data-set/cities_r2.csv")
df["city"]=df["city"].str.lower()
df=df["city"]
df=df.drop_duplicates()
df=df.dropna()
df=df.to_frame(name="city")
df1=df

# USE roberta-base-cased for final deployed model
glove_embedding = TransformerWordEmbeddings('bert-base-cased')

INDEX_SHAPE = 768
NUM_NEIGHBORS = 100
NUM_FINAL = 3

search_index = AnnoyIndex(INDEX_SHAPE, 'angular')
search_index.load('model.ann')


async def closest(query):
    QUERY = query 
    tmp=[Sentence(str(QUERY))]
    # embed a sentence using glove.
    glove_embedding.embed(tmp)
    tmp[0][0].embedding

    similar_item_ids = search_index.get_nns_by_vector(tmp[0][0].embedding.cpu().numpy(), NUM_NEIGHBORS, include_distances=True)

    results = pd.DataFrame(
        data={
            'texts': df1.iloc[similar_item_ids[0]]['city'],
            'distance': similar_item_ids[1],
        }
    )
    annoy=results["texts"].tolist()
    
    # universe=list(df["city"])
    universe=results["texts"].tolist()
    fuzzy=[]
    for word in universe:
        fuzzy.append([(fuzz.ratio(QUERY,word)+fuzz.partial_ratio(QUERY,word))/2,word])
    fuzzy.sort(reverse=True)
    fuzzy=fuzzy[:NUM_FINAL]

    rs = RefinedSoundex()
    sdx=[]
    for word in universe:
        sdx.append([rs.distance(QUERY,word),word])
    sdx.sort()
    sdx=sdx[:NUM_FINAL]

    return [fuzzy,sdx]