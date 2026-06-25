from __future__ import annotations
from Lawverse.storage.supabase_store import SupabaseChatStore
from Lawverse.storage.user_store import SupabaseUserStore
import os

_chat_store = None
_user_store = None

def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required because Lawverse is configured as cloud-only.")
    return value

def get_chat_store():
    global _chat_store
    if _chat_store is not None:
        return _chat_store

    _chat_store = SupabaseChatStore(
        supabase_url=_require_env("SUPABASE_URL"),
        supabase_key=_require_env("SUPABASE_KEY"),
    )

    return _chat_store

def get_user_store():
    global _user_store
    if _user_store is not None:
        return _user_store

    _user_store = SupabaseUserStore(
        supabase_url=_require_env("SUPABASE_URL"),
        supabase_key=_require_env("SUPABASE_KEY"),
    )

    return _user_store