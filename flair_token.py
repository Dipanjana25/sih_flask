import numpy as np
import streamlit as st
import pandas as pd
from flair.data import Sentence
from flair.nn import Classifier
from flair.models import SequenceTagger

# cache the model so it doesn't have to be loaded every time
@st.cache_resource
def load_flair_model():
    # use ner-large for better results
    classifier = SequenceTagger.load('ner-large')
    return classifier

classifier = load_flair_model()
st.session_state.query_string = ""
st.session_state.processed_string = ""
st.session_state.recent_queries = []

def handle_click():
    st.session_state.query_string = input_box
    st.session_state.recent_queries.append(st.session_state.query_string)
    sentence = Sentence(st.session_state.query_string)
    loss = classifier.predict((sentence), return_loss=True)
    st.session_state.processed_string = sentence.to_tagged_string()
    output_container.write(st.session_state.processed_string)
    output_container.write(loss)

    output_container.write('Detected geospatial entities')
    ner_dict = sentence.to_dict(tag_type='ner')
    ner_df = pd.DataFrame(ner_dict['entities'])
    output_container.write(ner_df)

st.title('Welcome to geo.Query')

st.sidebar.title('geo.Query')
st.sidebar.button('Home', use_container_width=True)
st.sidebar.button('How it works', use_container_width=True)
st.sidebar.button('About', use_container_width=True)
st.sidebar.button('Contact Us', use_container_width=True)
st.sidebar.title('Recent Queries')
recent_queries_container = st.sidebar.empty()


main_container=st.container()
main_container.subheader('Extract geospatial data from your queries using Deep Learning.')

main_container.divider()

main_container.write('Enter your query below')
input_box = main_container.text_input('Query', placeholder='Enter your query here', autocomplete='on', key=None, type='default')

main_container.button('Process Query', type='primary', on_click=handle_click, args=None, kwargs=None)

st.divider()

output_container=st.container()
output_container.title('Processed query')