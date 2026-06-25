from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

class ChatStore(ABC):
    @abstractmethod
    def save_chat(self, user_id: str, chat_id: str, title: str, history: List[Dict[str, str]]) -> None:
        pass

    @abstractmethod
    def load_chat(self, user_id: str, chat_id: str) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    def list_chats(self, user_id: str) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def delete_chat(self, user_id: str, chat_id: str) -> bool:
        pass