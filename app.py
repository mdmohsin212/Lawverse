from Lawverse.pipeline.rag_pipeline import rag_components, create_chat_chian
from Lawverse.logger import logging
from Lawverse.exception import ExceptionHandle
from flask import Flask, render_template, request, jsonify, session
import sys
import os

app = Flask(__name__)
app.secret_key = os.getenv("SECRET")

BASE_COMPONENTS = rag_components()
logging.info("Lawverse RAG components ready.")

active_chains = {}

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/chat", methods=["GET"])
def chat():
    chat_id = session.get("chat_id")
    if not chat_id or chat_id not in active_chains:
        chain, memory_manager = create_chat_chian(BASE_COMPONENTS)
        active_chains[memory_manager.chat_id] = (chain, memory_manager)
        session['chat_id'] = memory_manager.chat_id
    return render_template("chat.html")

@app.route("/new_chat", methods=["POST"])
def new_chat():
    chain, memory_manager = create_chat_chian(BASE_COMPONENTS)
    active_chains[memory_manager.chat_id] = (chain, memory_manager)
    session['chat_id'] = memory_manager.chat_id
    return jsonify({"chat_id" : memory_manager.chat_id, "title" : memory_manager._get_title()})

@app.route("/response", methods=["POST"])
def rag_response():
    try:
        chat_id = session.get("chat_id")
        if not chat_id or chat_id not in active_chains:
            chain, memory_manager = create_chat_chian(BASE_COMPONENTS)
            active_chains[memory_manager.chat_id] = (chain, memory_manager)
            session["chat_id"] = memory_manager.chat_id
            
        qa, memory_manager = active_chains[chat_id]
        
        data = request.get_json()
        query = data.get("message", "").strip()
        if not query:
            return jsonify({"error": "Empty message"}), 400
        
        result = qa({"question": query})
        memory_manager.save_memory()
        
        return jsonify({"answer": result["answer"]})
    
    except Exception as e:
        logging.error(f"Chat error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)