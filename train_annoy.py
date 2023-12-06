import numpy as np
import pandas as pd
import flair, torch
from annoy import AnnoyIndex
from flair.embeddings import TransformerWordEmbeddings
from flair.data import Sentence

flair.device = torch.device('cpu')

df=pd.read_csv("data-set/indian_cities_database.csv")
df=df["ascii_name"]
df=df.drop_duplicates()
df=df.dropna()
df=df.to_frame(name="ascii_name")

# init embedding

# USE roberta-base-cased for final deployed model
glove_embedding = TransformerWordEmbeddings('bert-base-cased')

# create sentence.
tmp=[]
for i,j in df.iterrows():
    tmp.append(Sentence(str(j["ascii_name"])))
    
# embed a sentence using glove.
glove_embedding.embed(tmp)

embeds=[]
for i in tmp:
    embeds.append(i[0].embedding)

tmp_embeds=[]
for i in embeds:
    tmp_embeds.append(np.array(i.cpu()))

tmp_embeds=np.array(tmp_embeds)
print(type(embeds[0]))
print(tmp_embeds.shape)

search_index = AnnoyIndex(tmp_embeds.shape[1], 'angular')
for i in range(len(embeds)):
    search_index.add_item(i, tmp_embeds[i])

search_index.build(10)
search_index.save('model.ann')