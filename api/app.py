from flask import Flask, render_template, request, jsonify, session, stream_with_context, Response
from dotenv import load_dotenv
import os
from threading import Lock
import logging as py_logging
from Lawverse.pipeline.rag_pipeline import rag_components
from Lawverse.pipeline.llm_loader import llm
from Lawverse.memory.langchain_memory import ChatMemory
from Lawverse.logger import logging
from Lawverse.monitoring.dashboard import monitor_bp
from Lawverse.agents.graph import create_agentic_chain
from Lawverse.storage.factory import get_chat_store
from api.auth import auth_bp, login_required
for logger_name in [
    "httpcore",
    "httpx",
    "hpack",
    "filelock",
    "sentence_transformers",
    "urllib3",
]:
    py_logging.getLogger(logger_name).setLevel(py_logging.WARNING)
    
load_dotenv()
app = Flask(__name__, template_folder="../templates")
app.secret_key = os.getenv("SECRET_KEY")

app.register_blueprint(auth_bp)
app.register_blueprint(monitor_bp)

BASE_COMPONENTS = None
BASE_COMPONENTS_LOCK = Lock()
active_chains = {}

def get_base_components():
    global BASE_COMPONENTS
    
    if BASE_COMPONENTS is not None:
        return BASE_COMPONENTS

    with BASE_COMPONENTS_LOCK:
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
    try:
        user_id = str(session.get("user_id"))
        chats = get_chat_store().list_chats(user_id)
        return jsonify(chats), 200

    except Exception as e:
        logging.error(f"Failed to list chats from cloud store: {e}")
        return jsonify([]), 200


@app.route("/load_chat/<chat_id>", methods=["POST"])
@login_required
def load_chat(chat_id):
    try:
        user_id = str(session.get("user_id"))
        data = get_chat_store().load_chat(user_id, chat_id)
        if not data:
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
        }), 200

    except Exception as e:
        logging.error(f"Failed to load chat from cloud store: {e}")
        return jsonify({"error": "Internal Server Error"}), 500


@app.route("/delete_chat/<chat_id>", methods=["DELETE"])
@login_required
def delete_chat(chat_id):
    try:
        user_id = str(session.get("user_id"))

        deleted = get_chat_store().delete_chat(user_id, chat_id)

        was_active = chat_id in active_chains
        if was_active:
            del active_chains[chat_id]

        if session.get("chat_id") == chat_id:
            session.pop("chat_id", None)

        return jsonify({
            "success": deleted,
            "was_active": was_active
        }), 200

    except Exception as e:
        logging.error(f"Error deleting cloud chat {chat_id}: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 7860)))