##########################################################################################################
# Calculate cosine similarity between the query vector and all other embedding vectors
##########################################################################################################
import numpy as np
from numpy.linalg import norm
import time
import pandas as pd
import os
import requests

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
    hf_token = os.environ.get('HF_TOKEN')

    # API endpoint for embedding model
    api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
    headers = {"Authorization": f"Bearer {hf_token}"}

    # call API
    response = requests.post(api_url, headers=headers, json={"inputs": text_chunk, "options":{"wait_for_model":True}})
   
    # load response from embedding model into json format
    embedding = response.json()

    return embedding

def calculate_cosine_similarity(text_chunk, embeddings_df):
    '''
    Calculate the cosine similarity between the query sentence and every other sentence
    1. Get the embeddings for the text chunk
    2. Calculate the cosine similarity between the embeddings of our text chunk und every other entry in the data frame

    Parameters:
        - text_chunk (string): the text snippet we want to use to look for similar entries in our database (embeddings_df)
        - embeddings_df (DataFrame): data frame with the columns "text_chunk" and "embeddings"
    Returns:
        -
    '''

    # use the _get_embeddings function the retrieve the embeddings for the text chunk
    sentence_embedding = _get_embeddings(text_chunk)

    # combine all dimensions of the vector embeddings to one array
    embeddings_df['embeddings_array'] = embeddings_df.apply(lambda row: row.values[:-1], axis=1)

    # start the timer
    start_time = time.time()
    print(start_time)

    # create a list to store the calculated cosine similarity
    cos_sim = []

    for index, row in embeddings_df.iterrows():
        A = row.embeddings_array
        B = sentence_embedding

        # calculate the cosine similarity
        cosine = np.dot(A,B)/(norm(A)*norm(B))

        cos_sim.append(cosine)

    embeddings_cosine_df = embeddings_df
    embeddings_cosine_df["cos_sim"] = cos_sim
    embeddings_cosine_df.sort_values(by=["cos_sim"], ascending=False)

    # stop the timer
    end_time = time.time()

    # calculate the time needed to calculate the similarities
    elapsed_time = (end_time - start_time)
    print("Execution Time: ", elapsed_time, "seconds")

    return embeddings_cosine_df

# Load embeddings_df.csv into data frame
embeddings_df = pd.read_csv('../02_Data/embeddings_df.csv')

# test query sentence
text_chunk = "Lilies are white."

# calculate cosine similarity
embeddings_cosine_df = calculate_cosine_similarity(text_chunk, embeddings_df)

# save data frame with text chunks and embeddings to csv
embeddings_cosine_df.to_csv('../02_Data/embeddings_cosine_df.csv', index=False)

# rank based on similarity
similarity_ranked_df = embeddings_cosine_df[["text_chunk", "cos_sim"]].sort_values(by=["cos_sim"], ascending=False)