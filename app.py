from flask import Flask, request, jsonify, render_template
from fuzzywuzzy import fuzz
import pandas as pd
import numpy as np
import faiss
import os

# Load dataset
df = pd.read_csv("dataset.csv")
df = df.fillna("")

# Load pre-generated embeddings
embeddings = np.load("embeddings.npy").astype("float32")

# Build FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Initialize Flask app
app = Flask(__name__, template_folder="templates", static_folder="static")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"answer": "Please enter a valid query."})

    # Convert question to vector using fuzzy matching only
    best_match_score = 0
    best_match_idx = None
    for i, name in enumerate(df["name"]):
        match_score = fuzz.partial_ratio(question.lower(), name.lower())
        if match_score > best_match_score:
            best_match_score = match_score
            best_match_idx = i

    # FAISS search
    q_vec = embeddings[best_match_idx:best_match_idx+1]
    distances, indices = index.search(q_vec, k=1)
    idx = indices[0][0]

    # Optional: Reject if match is too weak
    if distances[0][0] > 30 and best_match_score < 50:
        return jsonify({"answer": "Sorry, couldn't understand. Please rephrase your query."})

    # Generate response
    row = df.iloc[idx]
    answer = (
        f"Career: {row['name']}\n"
        f"Fields: {row['fields']}\n"
        f"Background: {row['background']}\n"
        f"Skills: {row['skills']}\n"
        f"Advice: {row['advice']}\n"
        f"Future scope: {row['future_scope']}"
    )
    return jsonify({"answer": answer})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
