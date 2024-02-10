from flask import Flask


def createApp():
    app = Flask(__name__, template_folder='templates')
    app.config['SECRET_KEY'] = "060804"

    from .views import views

    app.register_blueprint(views, url_prefix="/")

    return app
