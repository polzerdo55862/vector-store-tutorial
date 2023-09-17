##########################################################################################################
'''
Includes some functions to create a new vector store collection, fill it and query it
'''
##########################################################################################################
import chromadb
from chromadb.config import Settings
import pandas as pd

# vector store settings
VECTOR_STORE_PATH = r'../02_Data/00_Vector_Store'
COLLECTION_NAME = 'my_collection'

# Load embeddings_df.csv into data frame
embeddings_df = pd.read_csv('../02_Data/embeddings_df.csv')

def get_or_create_client_and_collection(VECTOR_STORE_PATH, COLLECTION_NAME):
    # get/create a chroma client
    chroma_client = chromadb.PersistentClient(path=VECTOR_STORE_PATH)

    # get or create collection
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

    return collection

# get or create collection
collection = get_or_create_client_and_collection(VECTOR_STORE_PATH, COLLECTION_NAME)

def add_to_collection(embeddings_df):
    # add a sample entry to collection
    # collection.add(
    #     documents=["This is a document", "This is another document"],
    #     metadatas=[{"source": "my_source"}, {"source": "my_source"}],
    #     ids=["id1", "id2"]
    # )

    # combine all dimensions of the vector embeddings to one array
    embeddings_df['embeddings_array'] = embeddings_df.apply(lambda row: row.values[:-1], axis=1)
    embeddings_df['embeddings_array'] = embeddings_df['embeddings_array'].apply(lambda x: x.tolist())

    # add data frame to collection
    collection.add(
        embeddings=embeddings_df.embeddings_array.to_list(),
        documents=embeddings_df.text_chunk.to_list(),
        # create a list of string as index
        ids=list(map(str, embeddings_df.index.tolist()))
    )

# add the embeddings_df to our vector store collection
add_to_collection(embeddings_df)

def get_all_entries(collection):
    # query collection
    existing_docs = pd.DataFrame(collection.get()).rename(columns={0: "ids", 1:"embeddings", 2:"documents", 3:"metadatas"})
    existing_docs.to_excel(r"..//02_Data//01_vector_stores_export.xlsx")
    return existing_docs

# extract all entries in vector store collection
existing_docs = get_all_entries(collection)

def query_vector_database(VECTOR_STORE_PATH, COLLECTION_NAME, query, n=2):
    # query collection
    results = collection.query(
        query_texts=query,
        n_results=n
    )

    print(f"Similarity Search: {n} most similar entries:")
    print(results["documents"])
    return results

# similarity search
similar_vector_entries = query_vector_database(VECTOR_STORE_PATH, COLLECTION_NAME, query=["Lilies are white."])