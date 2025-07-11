from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import faiss
import os
import requests

# Load dataset
df = pd.read_csv("dataset.csv")
df = df.fillna("")

# Load pre-generated embeddings
embeddings = np.load("embeddings.npy").astype("float32")

# Build FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Hugging Face API setup
HF_TOKEN = os.environ.get("HF_TOKEN")
API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

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

    # Generate embedding using Hugging Face API
    response = requests.post(API_URL, headers=headers, json={"inputs": question})
    if response.status_code != 200:
        return jsonify({"answer": "Error generating response. Please try again later."})
    
    q_vec = np.array(response.json(), dtype="float32").reshape(1, -1)

    # FAISS search
    distances, indices = index.search(q_vec, k=1)
    idx = indices[0][0]

    # Reject weak matches
    if distances[0][0] > 1.5:
        return jsonify({"answer": "Sorry, I couldn't understand that. Try rephrasing or being more specific."})

    # Format result
    row = df.iloc[idx]
    answer = (
        f"Career: {row['name']}\n"
        f"Domain: {row['major_domain']}\n"
        f"Fields: {row['fields']}\n"
        f"Background: {row['background']}\n"
        f"Skills: {row['skills']}\n"
        f"Typical Salary: {row['typical_salary']}\n"
        f"Demand Level: {row['demand_level']}\n"
        f"Course Duration: {row['course_duration']}\n"
        f"Top Companies: {row['top_companies']}\n"
        f"Advice: {row['advice']}\n"
        f"Future Scope: {row['future_scope']}\n"
        f"Related Courses: {row['related_courses']}\n"
        f"Career Switch Options: {row['career_switch_options']}\n"
        f"Goals Aligned: {row['goals_aligned']}"
    )
    return jsonify({"answer": answer})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
