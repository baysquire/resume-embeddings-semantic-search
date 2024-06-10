import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import DistanceMetric

# Load synthetic data
df_resume = pd.read_csv('resumes_train.csv')

# Generate Embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
embedding_arr = model.encode(df_resume['resume'], show_progress_bar=True)

# Function to perform Semantic Search
def find_similar_resumes(query, model, embeddings, df, top_n=10):
    query_embedding = model.encode(query).reshape(1, -1)
    dist = DistanceMetric.get_metric('euclidean')
    dist_arr = dist.pairwise(embeddings, query_embedding).flatten()
    sorted_indices = np.argsort(dist_arr)
    
    top_roles = df['role'].iloc[sorted_indices[:top_n]]
    top_resumes = df['resume'].iloc[sorted_indices[:top_n]]
    
    print("Top roles for the query:")
    print(top_roles)
    print("\nResume closest to the query:")
    print(top_resumes.iloc[0])
    
    return sorted_indices

# Query for semantic search
query = "Data Engineer with Apache Airflow experience"
top_indices = find_similar_resumes(query, model, embedding_arr, df_resume)
