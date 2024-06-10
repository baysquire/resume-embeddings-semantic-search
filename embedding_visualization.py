import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib as mpl

# Load synthetic data
df_resume = pd.read_csv('resumes_train.csv')

# Correct the last role to "Other" if it matches any existing role
if df_resume['role'].iloc[-1] in df_resume['role'][:-1].values:
    df_resume.at[df_resume.index[-1], 'role'] = "Other"

# Generate Embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
embedding_arr = model.encode(df_resume['resume'], show_progress_bar=True)
print(f"Embedding array shape: {embedding_arr.shape}")

# Apply PCA for Visualization
pca = PCA(n_components=2)
embedding_2d = pca.fit_transform(embedding_arr)
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")

# Plot Data along PCA components
plt.figure(figsize=(10, 8))
plt.rcParams.update({'font.size': 14})
plt.grid()

cmap = mpl.colormaps['jet']
color_step = 1 / len(df_resume['role'].unique())
c = 0

for role in df_resume['role'].unique():
    idx = df_resume[df_resume['role'] == role].index
    plt.scatter(embedding_2d[idx, 0], embedding_2d[idx, 1], 
                color=cmap(c), label=role)
    c += color_step

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA of Resume Embeddings")
plt.show()
