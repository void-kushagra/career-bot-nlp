from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import faiss
import os
import traceback  # ✅ Added for better error visibility

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
            print("[ERROR] No 'vector' field in request.")
            return jsonify({"answer": "No embedding vector received."}), 400

        vector = data["vector"]

        # Debug info
        print("[INFO] Vector type:", type(vector))
        print("[INFO] Vector length:", len(vector))
        print("[INFO] First 5 values of vector:", vector[:5])

        q_vec = np.array(vector, dtype="float32").reshape(1, -1)
        print("[INFO] Vector shape after reshape:", q_vec.shape)

        if q_vec.shape[1] != 512:
            print("[ERROR] Invalid vector size:", q_vec.shape)
            return jsonify({"answer": "Invalid vector dimensions. Must be 512."}), 400

        # FAISS search
        distances, indices = index.search(q_vec, k=1)
        print("[INFO] FAISS search distances:", distances)
        print("[INFO] FAISS search indices:", indices)

        idx = indices[0][0]

        if distances[0][0] > 1.5:
            return jsonify({"answer": "Sorry, I couldn't understand that. Try rephrasing or being more specific."})

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
        print("[ERROR] Exception occurred:")
        traceback.print_exc()  # ✅ Prints the exact error and line
        return jsonify({"answer": "Internal error during processing."}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port, debug=False)
