import re
import umap
import cohere
import numpy as np
import pandas as pd
import altair as alt
from annoy import AnnoyIndex
from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz

API_KEY = "EjhSxeGpKbt7vT85tbcLhqWoPXJegTi44PUmY6lT"  # add your API key here.
co_client = cohere.Client(API_KEY)

# # Load the dataset
df=pd.read_csv("data-set/cities_r2.csv")
df=df["ascii_name"]
df=df.drop_duplicates()
df=df.dropna()
df=df.to_frame(name="ascii_name")



embeds = co_client.embed(
    texts=list(df['ascii_name']),
    model='large',
    truncate='RIGHT'
).embeddings

embeds = np.array(embeds)


search_index = AnnoyIndex(embeds.shape[1], 'angular')
for i in range(len(embeds)):
    search_index.add_item(i, embeds[i])

search_index.build(10)
search_index.save('test.ann')



EXAMPLE_ID = 69  # Nice
NUM_NEIGHBORS = 30
QUERY = "Ahmadbad"   #  Write your own query here(Eg: What is the oldest profession?).

query_embed = co_client.embed(
    texts = [QUERY],
    model = 'large',
    truncate='RIGHT'
).embeddings

similar_item_ids = search_index.get_nns_by_vector(
    query_embed[0], NUM_NEIGHBORS, include_distances=True)

results = pd.DataFrame(
    data={
        'texts': df.iloc[similar_item_ids[0]]['ascii_name'],
        'distance': similar_item_ids[1],
    }
)
print(f"The Question is : {QUERY}")
print('The nearest neighbors are : ')
print(results)

final=[]
for word in results["texts"]:
    final.append([(fuzz.ratio(QUERY,word)+fuzz.partial_ratio(QUERY,word))/2,word])
final.sort(reverse=True)
print(final)

from pyphonetics import RefinedSoundex
rs = RefinedSoundex()
final=[]
for word in results["texts"]:
    final.append([rs.distance(QUERY,word),word])
final.sort()
print(final)

