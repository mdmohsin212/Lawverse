from flask import Flask, render_template, request, jsonify, session, stream_with_context, Response
from Lawverse.pipeline.rag_pipeline import rag_components
from Lawverse.pipeline.llm_loader import llm
from Lawverse.memory.langchain_memory import ChatMemory
from Lawverse.utils.config import MEMORY_DIR
from Lawverse.logger import logging
from Lawverse.monitoring.dashboard import monitor_bp
from Lawverse.agents.graph import create_agentic_chain
from api.auth import auth_bp, login_required
from api.models import db
from api.admin import admin
from dotenv import load_dotenv
import json
import glob
import os
import secrets

load_dotenv()

app = Flask(__name__, template_folder="../templates")
app.secret_key = os.getenv("SECRET_KEY") or secrets.token_hex(32)
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL", "sqlite:///users.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)
admin.init_app(app)
app.register_blueprint(auth_bp)
app.register_blueprint(monitor_bp)

with app.app_context():
    db.create_all()

BASE_COMPONENTS = None
active_chains = {}


def get_base_components():
    global BASE_COMPONENTS

    if BASE_COMPONENTS is None:
        logging.info("Loading Lawverse RAG base components...")
        BASE_COMPONENTS = rag_components()
        logging.info("Lawverse RAG base components loaded successfully.")

    return BASE_COMPONENTS


def create_agent_session(chat_id=None):
    components = get_base_components()
    chain = create_agentic_chain(components, llm)
    memory_manager = ChatMemory(chat_id=chat_id)

    active_chains[memory_manager.chat_id] = (chain, memory_manager)
    session["chat_id"] = memory_manager.chat_id
    memory_manager.save_memory()

    return chain, memory_manager

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/chat", methods=["GET"])
@login_required
def chat():
    chat_id = session.get("chat_id")
    if not chat_id or chat_id not in active_chains:
        create_agent_session()
        
    return render_template("chat.html")

@app.route("/new_chat", methods=["POST"])
@login_required
def new_chat():
    _, memory_manager = create_agent_session()

    return jsonify({
        "chat_id": memory_manager.chat_id,
        "title": memory_manager._get_title()
    })


@app.route("/response", methods=["POST"])
@login_required
def rag_response():
    try:
        chat_id = session.get("chat_id")
        if not chat_id or chat_id not in active_chains:
            _, memory_manager = create_agent_session()
            chat_id = memory_manager.chat_id

        qa, memory_manager = active_chains[chat_id]
        data = request.get_json(silent=True) or {}
        query = data.get("message", "").strip()

        if not query:
            return jsonify({"error": "Empty message"}), 400

        def generate():
            answer_parts = []
            try:
                for chunk in qa.stream({
                    "input": query,
                    "chat_history": memory_manager.memory.chat_memory.messages,
                }):
                    text = chunk if isinstance(chunk, str) else str(chunk)
                    answer_parts.append(text)
                    yield text

                full_answer = "".join(answer_parts).strip()
                memory_manager.append_exchange(query, full_answer)
                memory_manager.save_memory()

            except Exception as e:
                logging.error(f"Error during stream generation: {e}")
                yield "**Error:** An error occurred while processing your request."

        return Response(stream_with_context(generate()), mimetype="text/plain")

    except Exception as e:
        logging.error(f"Chat error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/get_chats", methods=["GET"])
@login_required
def get_chats():
    chats = []
    user_id = session.get("user_id")
    os.makedirs(MEMORY_DIR, exist_ok=True)
    for file_path in glob.glob(f"{MEMORY_DIR}/*.json"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if data.get("user_id") == user_id:
                chats.append({
                    "chat_id": data.get("chat_id"),
                    "last_updated": data.get("last_updated"),
                    "title": data.get("title", f"Chat-{data.get('chat_id')}")
                })

        except Exception as e:
            logging.warning(f"Skipping unreadable memory file {file_path}: {e}")

    chats.sort(key=lambda x: x.get("last_updated") or x.get("chat_id"), reverse=True)
    return jsonify(chats)


@app.route("/load_chat/<chat_id>", methods=["POST"])
@login_required
def load_chat(chat_id):
    user_id = session.get("user_id")
    memory_path = os.path.join(MEMORY_DIR, f"user_{user_id}_{chat_id}.json")
    if not os.path.exists(memory_path):
        return jsonify({"error": "Chat not found"}), 404

    _, memory_manager = create_agent_session(chat_id=chat_id)

    messages_list = memory_manager.memory.chat_memory.messages
    messages = []

    for i in range(0, len(messages_list), 2):
        user_msg = messages_list[i].content if i < len(messages_list) else None
        ai_msg = messages_list[i + 1].content if i + 1 < len(messages_list) else ""
        if user_msg:
            messages.append({
                "user": user_msg,
                "ai": ai_msg
            })

    return jsonify({
        "chat_id": chat_id,
        "title": memory_manager._get_title(),
        "messages": messages,
    })


@app.route("/delete_chat/<chat_id>", methods=["DELETE"])
@login_required
def delete_chat(chat_id):
    try:
        user_id = session.get("user_id")
        memory_path = os.path.join(MEMORY_DIR, f"user_{user_id}_{chat_id}.json")

        if os.path.exists(memory_path):
            os.remove(memory_path)
            logging.info(f"Deleted chat for user {user_id}, chat_id: {chat_id}")

        was_active = chat_id in active_chains
        if was_active:
            del active_chains[chat_id]

        if session.get("chat_id") == chat_id:
            session.pop("chat_id", None)

        return jsonify({
            "success": True,
            "was_active": was_active
        }), 200

    except Exception as e:
        logging.error(f"Error deleting chat {chat_id}: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 7860)))