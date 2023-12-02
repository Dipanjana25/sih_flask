import re
import umap
import cohere
import numpy as np
import pandas as pd
import altair as alt
from annoy import AnnoyIndex
from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_similarity


API_KEY = "SCVgg88N5i8qDCGimVmrsnY76sNyQvDidECqkh9B"  # add your API key here.
co_client = cohere.Client(API_KEY)