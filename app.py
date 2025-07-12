from flask import Flask, request, jsonify, render_template
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
    try:
        data = request.get_json()
        print("[INFO] Received data:", data)

        if not data or "vector" not in data:
            return jsonify({"answer": "No embedding vector received."})

        vector = data["vector"]
        q_vec = np.array(vector, dtype="float32").reshape(1, -1)
        print("[INFO] Vector shape:", q_vec.shape)

        # FAISS search
        distances, indices = index.search(q_vec, k=1)
        print("[INFO] FAISS result:", distances, indices)

        idx = indices[0][0]

        # Reject weak matches
        if distances[0][0] > 1.5:
            return jsonify({"answer": "Sorry, I couldn't understand that. Try rephrasing or being more specific."})

        # Fetch matched row
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

    except Exception as e:
        print("[ERROR] Exception occurred:", str(e))
        return jsonify({"answer": "Internal error during processing."})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port, debug=False)
