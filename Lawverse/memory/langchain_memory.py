import os
import sys
import json
from datetime import datetime
from langchain.memory import ConversationBufferMemory
from Lawverse.logger import logging
from Lawverse.exception import ExceptionHandle
from Lawverse.utils.config import MEMORY_DIR


class ChatMemory:
    def __init__(self, chat_id=None):
        try:
            self.chat_id = chat_id or self._create_new_chat_id()
            self.memory_file = os.path.join(MEMORY_DIR, f"{self.chat_id}.json")
            self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
            self._load_memory()
            logging.info(f"ChatMemory initialized for chat_id: {self.chat_id}")
        except Exception as e:
            raise ExceptionHandle(e, sys) from e

    def _create_new_chat_id(self):
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _load_memory(self):
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                for msg in data.get("history", []):
                    self.memory.chat_memory.add_user_message(msg["user"])
                    self.memory.chat_memory.add_ai_message(msg["ai"])
                logging.info(f"Memory loaded successfully for chat_id: {self.chat_id}")
            else:
                logging.info(f"No existing memory found for chat_id: {self.chat_id}")
        except Exception as e:
            raise ExceptionHandle(e, sys) from e

    def save_memory(self):
        try:
            messages = self.memory.chat_memory.messages
            data = {
                "chat_id": self.chat_id,
                "title": self._get_title(),
                "last_updated": datetime.now().isoformat(),
                "history": [
                    {"user": messages[i].content, "ai": messages[i + 1].content}
                    for i in range(0, len(messages) - 1, 2)
                ],
            }
            with open(self.memory_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

            logging.info(f"Memory saved successfully for chat_id: {self.chat_id}")
        except Exception as e:
            raise ExceptionHandle(e, sys) from e

    def _get_title(self):
        if self.memory.chat_memory.messages:
            return self.memory.chat_memory.messages[0].content[:30]
        return f"Chat - {self.chat_id}"

    def clear_memory(self):
        try:
            self.memory.clear()
            if os.path.exists(self.memory_file):
                os.remove(self.memory_file)
                logging.info(f"Memory file deleted for chat_id: {self.chat_id}")
            else:
                logging.info(f"No memory file found to delete for chat_id: {self.chat_id}")
        except Exception as e:
            raise ExceptionHandle(e, sys) from e