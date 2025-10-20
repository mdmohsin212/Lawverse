from flask import Blueprint, render_template, request, redirect, url_for, flash, session
from api.models import db, User
from functools import wraps

auth_bp = Blueprint("auth", __name__)

@auth_bp.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        first_name = request.form.get("first_name")
        last_name = request.form.get("last_name")
        email = request.form.get("email")
        password = request.form.get("password")
        confirm_password = request.form.get("confirm_password")
        
        if password != confirm_password:
            flash("Passwords do not match!")
            return redirect(url_for("auth.signup"))

        if User.query.filter_by(email=email).first():
            flash("Email already registered!")
            return redirect(url_for("auth.signup"))
        
        user = User(first_name=first_name, last_name=last_name, email=email)
        user.set_password(password=password)
        db.session.add(user)
        db.session.commit()
        
        session['user_id'] = user.id
        session['user_name'] = user.first_name

        flash("Login Successful!")
        return render_template("signup.html")
        
    return render_template("signup.html")

@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        
        user = User.query.filter_by(email=email).first()
        if user and user.check_password(password):
            session['user_id'] = user.id
            session['user_name'] = user.first_name
            flash("Login Successful!")
            return render_template("login.html")
        else:
            flash("Invalid email or password.")
            return redirect(url_for("auth.login"))  
        
    return render_template("login.html")

@auth_bp.route("/logout")
def logout():
    session.pop('user_id', None)
    session.pop('user_name', None)
    return redirect(url_for("home"))


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("auth.login"))
        return f(*args, **kwargs)
    return decorated_function