import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# Load dataset
df = pd.read_csv("dataset.csv")
df = df.fillna("")

# Combine all relevant fields into one string
combined_fields = [
    "major_domain", "name", "fields", "background", "skills",
    "typical_salary", "demand_level", "course_duration",
    "top_companies", "advice", "future_scope",
    "related_courses", "career_switch_options", "goals_aligned"
]

df["context"] = df[combined_fields].agg(" ".join, axis=1)

# Save updated dataset
df.to_csv("dataset.csv", index=False)

# Generate embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df["context"].tolist(), convert_to_numpy=True)

# Save embeddings
np.save("embeddings.npy", embeddings)

print("✅ Embeddings generated from full context and saved.")
