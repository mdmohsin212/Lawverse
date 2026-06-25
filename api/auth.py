from functools import wraps
from flask import Blueprint, render_template, request, redirect, url_for, flash, session
from Lawverse.storage.factory import get_user_store
auth_bp = Blueprint("auth", __name__)


def _set_session(user_id, first_name, last_name):
    session["user_id"] = str(user_id)
    session["user_name"] = first_name or ""
    session["last_name"] = last_name or ""


@auth_bp.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        first_name = request.form.get("first_name", "").strip()
        last_name = request.form.get("last_name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not email or not password:
            flash("Email and password are required.")
            return redirect(url_for("auth.signup"))

        if password != confirm_password:
            flash("Passwords do not match.")
            return redirect(url_for("auth.signup"))

        if len(password) < 8:
            flash("Password must be at least 8 characters.")
            return redirect(url_for("auth.signup"))

        try:
            store = get_user_store()
            if store.get_by_email(email):
                flash("Email already registered.")
                return redirect(url_for("auth.signup"))

            user = store.create_user(first_name, last_name, email, password)

            _set_session(
                user_id=user["id"],
                first_name=user.get("first_name"),
                last_name=user.get("last_name"),
            )

            flash("Signup successful.")
            return redirect(url_for("chat"))

        except Exception as e:
            flash(f"Signup failed: {str(e)}")
            return redirect(url_for("auth.signup"))

    return render_template("signup.html")


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        try:
            store = get_user_store()
            user = store.verify_user(email, password)

            if not user:
                flash("Invalid email or password.")
                return redirect(url_for("auth.login"))

            _set_session(
                user_id=user["id"],
                first_name=user.get("first_name"),
                last_name=user.get("last_name"),
            )

            flash("Login successful.")
            return redirect(url_for("chat"))

        except Exception as e:
            flash(f"Login failed: {str(e)}")
            return redirect(url_for("auth.login"))

    return render_template("login.html")


@auth_bp.route("/logout")
def logout():
    session.pop("user_id", None)
    session.pop("user_name", None)
    session.pop("last_name", None)
    session.pop("chat_id", None)
    return redirect(url_for("home"))


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("auth.login"))
        return f(*args, **kwargs)
    return decorated_function