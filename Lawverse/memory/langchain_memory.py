import sys
from flask import session, has_request_context
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage
from Lawverse.logger import logging
from Lawverse.exception import ExceptionHandle
from Lawverse.storage.factory import get_chat_store


class _SimpleChatHistory:
    def __init__(self):
        self.messages = []

    def add_user_message(self, content: str):
        self.messages.append(HumanMessage(content=content or ""))

    def add_ai_message(self, content: str):
        self.messages.append(AIMessage(content=content or ""))

    def clear(self):
        self.messages.clear()


class _SimpleMemory:
    def __init__(self):
        self.chat_memory = _SimpleChatHistory()

    def clear(self):
        self.chat_memory.clear()

class ChatMemory:
    def __init__(self, chat_id=None, user_id=None):
        try:
            self.chat_id = chat_id or self._create_new_chat_id()
            self.user_id = str(user_id or self._get_current_user_id())
            self.store = get_chat_store()
            self.memory = _SimpleMemory()
            self._load_memory()
            logging.info(f"ChatMemory initialized for user_id={self.user_id}, chat_id={self.chat_id}")
        except Exception as e:
            raise ExceptionHandle(e, sys) from e

    def _get_current_user_id(self):
        if has_request_context():
            return session.get("user_id", "anonymous")
        return "anonymous"

    def _create_new_chat_id(self):
        return datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")

    def _load_memory(self):
        try:
            data = self.store.load_chat(self.user_id, self.chat_id)
            if not data:
                logging.info(f"No existing memory found for chat_id: {self.chat_id}")
                return

            for msg in data.get("history", []) or []:
                user_msg = msg.get("user", "")
                ai_msg = msg.get("ai", "")
                if user_msg:
                    self.memory.chat_memory.add_user_message(user_msg)
                if ai_msg:
                    self.memory.chat_memory.add_ai_message(ai_msg)

            logging.info(f"Memory loaded successfully for chat_id: {self.chat_id}")
        except Exception as e:
            raise ExceptionHandle(e, sys) from e

    def append_exchange(self, user_message: str, ai_message: str):
        if user_message:
            self.memory.chat_memory.add_user_message(user_message)
        if ai_message:
            self.memory.chat_memory.add_ai_message(ai_message)

    def _history_as_pairs(self):
        messages = self.memory.chat_memory.messages
        history = []
        i = 0
        while i < len(messages):
            user_msg = messages[i].content if i < len(messages) else ""
            ai_msg = messages[i + 1].content if i + 1 < len(messages) else ""
            if user_msg or ai_msg:
                history.append({"user": user_msg, "ai": ai_msg})
            i += 2
        return history

    def save_memory(self) -> bool:
        try:
            self.store.save_chat(
                user_id=self.user_id,
                chat_id=self.chat_id,
                title=self._get_title(),
                history=self._history_as_pairs(),
            )
            logging.info(f"Memory saved successfully for chat_id: {self.chat_id}")
            return True
        except Exception as e:
            logging.error(f"Memory save failed for chat_id={self.chat_id}: {e}")
            return False

    def _get_title(self):
        for msg in self.memory.chat_memory.messages:
            content = getattr(msg, "content", "") or ""
            if content.strip():
                return content.strip()[:40]
        return "New legal query"

    def clear_memory(self):
        try:
            self.memory.clear()
            self.store.delete_chat(self.user_id, self.chat_id)
            logging.info(f"Memory deleted for chat_id: {self.chat_id}")
        except Exception as e:
            raise ExceptionHandle(e, sys) from e