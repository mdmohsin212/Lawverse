from __future__ import annotations
from typing import Any, Dict, Optional
from supabase import create_client
from werkzeug.security import generate_password_hash, check_password_hash


class SupabaseUserStore:
    def __init__(self, supabase_url: str, supabase_key: str):
        self.client = create_client(supabase_url, supabase_key)
        self.table = "lawverse_users"

    def get_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        result = (
            self.client
            .table(self.table)
            .select("*")
            .eq("email", email.lower().strip())
            .limit(1)
            .execute()
        )
        if not result.data:
            return None

        return result.data[0]

    def create_user(self, first_name: str, last_name: str, email: str, password: str) -> Dict[str, Any]:
        email = email.lower().strip()
        if self.get_by_email(email):
            raise ValueError("Email already registered.")

        payload = {
            "first_name": first_name.strip() if first_name else "",
            "last_name": last_name.strip() if last_name else "",
            "email": email,
            "password_hash": generate_password_hash(password),
        }

        result = self.client.table(self.table).insert(payload).execute()
        if not result.data:
            raise RuntimeError("Failed to create user in Supabase.")

        return result.data[0]

    def verify_user(self, email: str, password: str) -> Optional[Dict[str, Any]]:
        user = self.get_by_email(email)
        if not user:
            return None

        if check_password_hash(user["password_hash"], password):
            return user

        return None