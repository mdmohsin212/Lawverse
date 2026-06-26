from __future__ import annotations
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from supabase import create_client
from Lawverse.storage.base import ChatStore


class SupabaseChatStore(ChatStore):
    def __init__(self, supabase_url: str, supabase_key: str):
        self.client = create_client(supabase_url, supabase_key)
        self.table = "lawverse_chats"

    def save_chat(self, user_id: str, chat_id: str, title: str, history: List[Dict[str, str]]) -> None:
        payload = {
            "user_id": str(user_id),
            "chat_id": str(chat_id),
            "title": title or "New legal query",
            "history": history or [],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        existing = (
            self.client
            .table(self.table)
            .select("id")
            .eq("user_id", str(user_id))
            .eq("chat_id", str(chat_id))
            .limit(1)
            .execute()
        )

        if existing.data:
            row_id = existing.data[0]["id"]
            self.client.table(self.table).update(payload).eq("id", row_id).execute()
        else:
            self.client.table(self.table).insert(payload).execute()

    def load_chat(self, user_id: str, chat_id: str) -> Optional[Dict[str, Any]]:
        result = (
            self.client
            .table(self.table)
            .select("*")
            .eq("user_id", str(user_id))
            .eq("chat_id", str(chat_id))
            .limit(1)
            .execute()
        )

        if not result.data:
            return None

        row = result.data[0]
        return {
            "user_id": row.get("user_id"),
            "chat_id": row.get("chat_id"),
            "title": row.get("title") or "New legal query",
            "history": row.get("history") or [],
            "last_updated": row.get("updated_at") or row.get("created_at"),
        }

    def list_chats(self, user_id: str) -> List[Dict[str, Any]]:
        result = (
            self.client
            .table(self.table)
            .select("chat_id,title,updated_at,created_at")
            .eq("user_id", str(user_id))
            .order("updated_at", desc=True)
            .execute()
        )

        chats = []
        for row in result.data or []:
            chats.append({
                "chat_id": row.get("chat_id"),
                "title": row.get("title") or "New legal query",
                "last_updated": row.get("updated_at") or row.get("created_at"),
            })

        return chats

    def delete_chat(self, user_id: str, chat_id: str) -> bool:
        self.client.table(self.table).delete().eq("user_id", str(user_id)).eq("chat_id", str(chat_id)).execute()
        return True