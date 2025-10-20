from flask_admin import Admin
from flask_admin.contrib.sqla import ModelView
from flask import redirect, url_for, session
from api.models import db, User

class MyModelView(ModelView):
    def is_accessible(self):
        return session.get('user_id') == 1

    def inaccessible_callback(self, name, **kwargs):
        return redirect(url_for('auth.login'))

admin = Admin(name="Admin", template_mode="bootstrap4", url="/admin")
admin.add_view(MyModelView(User, db.session))