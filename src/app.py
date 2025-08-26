from flask import Flask
from routes import all_blueprints

app = Flask(__name__)

for bp in all_blueprints:
    app.register_blueprint(bp)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=False)
