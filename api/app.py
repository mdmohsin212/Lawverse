from Lawverse.pipeline.rag_pipeline import rag_components, create_chat_chian
from flask import Flask, render_template, request, jsonify, session, stream_with_context, Response
from Lawverse.utils.config import MEMORY_DIR
from Lawverse.logger import logging
from Lawverse.monitoring.dashboard import monitor_bp
from api.auth import auth_bp, login_required
from api.models import db
import markdown
import json
import glob
import os
from api.admin import admin

app = Flask(__name__, template_folder="../templates")
# app.secret_key = os.getenv("SECRET")
app.secret_key = "lawverse-secret"
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)
admin.init_app(app)
app.register_blueprint(auth_bp)
app.register_blueprint(monitor_bp)

with app.app_context():
    db.create_all()

BASE_COMPONENTS = rag_components()
logging.info("Lawverse RAG components ready.")

active_chains = {}

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/chat", methods=["GET"])
@login_required
def chat():
    chat_id = session.get("chat_id")
    if not chat_id or chat_id not in active_chains:
        chain, memory_manager = create_chat_chian(BASE_COMPONENTS)
        active_chains[memory_manager.chat_id] = (chain, memory_manager)
        session['chat_id'] = memory_manager.chat_id
        
        memory_manager.save_memory()
        
    return render_template("chat.html")

@app.route("/new_chat", methods=["POST"])
def new_chat():
    chain, memory_manager = create_chat_chian(BASE_COMPONENTS)
    active_chains[memory_manager.chat_id] = (chain, memory_manager)
    session['chat_id'] = memory_manager.chat_id
    return jsonify({"chat_id" : memory_manager.chat_id,"title" : memory_manager._get_title()})

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
        
        def generate():
            try:
                for chunk in qa.stream({"input": query}):
                    yield chunk
                memory_manager.save_memory()

            except Exception as e:
                logging.error(f"Error during stream generation: {e}")
                yield f"**Error:** An error occurred while processing your request."
        
        return Response(stream_with_context(generate()), mimetype='text/plain')
    
    except Exception as e:
        logging.error(f"Chat error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/get_chats', methods=["GET"])
def get_chats():
    chats = []
    user_id = session.get("user_id")
    
    for file_path in glob.glob(f"{MEMORY_DIR}/*.json"):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
            if data.get("user_id") == user_id:
                chats.append({
                    "chat_id": data.get("chat_id"),
                    "last_updated": data.get("last_updated"),
                    "title": data.get("title", f"Chat-{data.get('chat_id')}")
                })
    
    chats.sort(key=lambda x: x.get('last_updated', x['chat_id']), reverse=True)
    return jsonify(chats)


@app.route("/load_chat/<chat_id>", methods=["POST"])
def load_chat(chat_id):
    user_id = session.get("user_id")
    if not os.path.exists(os.path.join(MEMORY_DIR, f"user_{user_id}_{chat_id}.json")):
        return jsonify({"error": "Chat not found"}), 404

    chain, memory_manager = create_chat_chian(BASE_COMPONENTS, chat_id=chat_id)
    active_chains[memory_manager.chat_id] = (chain, memory_manager)
    session["chat_id"] = memory_manager.chat_id

    messages_list = memory_manager.memory.chat_memory.messages
    messages = []

    for i in range(0, len(messages_list), 2):
        user_msg = messages_list[i].content if i < len(messages_list) else None
        ai_msg = messages_list[i + 1].content if i + 1 < len(messages_list) else ""
        if user_msg:
            messages.append({"user": user_msg, "ai": ai_msg})

    return jsonify({
        "chat_id": chat_id,
        "title": memory_manager._get_title(),
        "messages": messages
    })

@app.route("/delete_chat/<chat_id>", methods=['DELETE'])
@login_required
def delete_chat(chat_id):
    try:
        user_id = session.get("user_id")
        memory_path = os.path.join(MEMORY_DIR, f"user_{user_id}_{chat_id}.json")
        
        os.remove(memory_path)
        logging.info(f"Deleted chat for user {user_id}, chat_id: {chat_id}")
        
        was_active = chat_id in active_chains
        if was_active:
            del active_chains[chat_id]
            
        return jsonify({"success": True, "was_active": was_active}), 200
    
    except Exception as e:
        logging.error(f"Error deleting chat {chat_id}")
        return jsonify({"error": "Internal Server Error"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)