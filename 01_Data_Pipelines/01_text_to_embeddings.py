##########################################################################################################
'''
The following script translates the list of strings "text_chunks" into vector embeddings and
saves the DataFrame including "text_chunks" and "embeddings" in the csv file "embeddings_df.csv"
'''
##########################################################################################################

import os
import requests
import pandas as pd
import numpy as np

# hugging face token
os.environ['hf_token'] = os.environ.get('HF_TOKEN')
# os.environ['hf_token'] = 'testtoken123'

# example text snippets we want to translate into vector embeddings
text_chunks = [
    "The sky is blue.",
    "The grass is green.",
    "The sun is shining.",
    "I love chocolate.",
    "Pizza is delicious.",
    "Coding is fun.",
    "Roses are red.",
    "Violets are blue.",
    "Water is essential for life.",
    "The moon orbits the Earth.",
]

def _get_embeddings(text_chunk):
    '''
    Use embedding model from hugging face to calculate embeddings for the text snippets provided
   
    Parameters:
        - text_chunk (string): the sentence or text snippet you want to translate into embeddings

    Returns:
        - embedding(list): list with all embedding dimensions
    '''
    # define the embedding model you want to use
    model_id = "sentence-transformers/all-MiniLM-L6-v2"

    # you can find the token to the hugging face api in your settings page https://huggingface.co/settings/tokens
    hf_token = os.environ.get('hf_token')

    # API endpoint for embedding model
    api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
    headers = {"Authorization": f"Bearer {hf_token}"}

    # call API
    response = requests.post(api_url, headers=headers, json={"inputs": text_chunk, "options":{"wait_for_model":True}})
   
    # load response from embedding model into json format
    embedding = response.json()

    return embedding

def from_text_to_embeddings(text_chunks):
    '''
    Translate sentences into vector embeddings

    Attributes:
        - text_chunks (list): list of example strings

    Returns:
        - embeddings_df (DataFrame): data frame with the columns "text_chunk" and "embeddings"
    '''
    # create new data frame using text chunks list
    embeddings_df = pd.DataFrame(text_chunks).rename(columns={0:"text_chunk"})

    # use the _get_embeddings function to retrieve the embeddings for each of the sentences
    embeddings_df["embeddings"] = embeddings_df["text_chunk"].apply(_get_embeddings)

    # split the embeddings column into individuell columns for each vector dimension
    embeddings_df = embeddings_df['embeddings'].apply(pd.Series)
    embeddings_df["text_chunk"] = text_chunks

    return embeddings_df

# get embeddings for each of the text chunks
embeddings_df = from_text_to_embeddings(text_chunks)

# save data frame with text chunks and embeddings to csv
embeddings_df.to_csv('../02_Data/embeddings_df.csv', index=False)