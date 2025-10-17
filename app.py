from Lawverse.pipeline.rag_pipeline import rag_chain
from Lawverse.logger import logging
from Lawverse.exception import ExceptionHandle
from flask import Flask, render_template, request, jsonify
import sys

app = Flask(__name__)
qa = rag_chain()
chat_history = []

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/chat", methods=["GET"])
def home():
    return render_template("chat.html")

@app.route("/response", methods=["POST"])
def chat():
    global chat_history
    try:
        data = request.get_json()
        query = data.get("message", "").strip()
        if not query:
            return jsonify({"error": "Empty message"}), 400

        result = qa({"question": query, "chat_history": chat_history})

        chat_history.append((query, result["answer"]))
        if len(chat_history) > 5:
            chat_history = chat_history[-5:]

        return jsonify({"answer": result["answer"]})

    except Exception as e:
        logging.error(f"Chat error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)