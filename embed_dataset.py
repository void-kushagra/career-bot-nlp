from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load dataset
df = pd.read_csv("dataset.csv")
df = df.fillna("")

# Combine fields
texts = (
    df["name"] + ". Fields: " + df["fields"] +
    ". Background: " + df["background"] +
    ". Skills: " + df["skills"] +
    ". Advice: " + df["advice"] +
    ". Future scope: " + df["future_scope"]
).tolist()

# Generate and save embeddings
print("Generating embeddings... (one-time ⏳)")
embeddings = model.encode(texts, show_progress_bar=True)
np.save("embeddings.npy", embeddings)
print("✅ Done! Saved to embeddings.npy")
