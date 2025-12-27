from flask import Flask
from controllers.prediction_controller import prediction_bp
import os

app = Flask(__name__)
app.register_blueprint(prediction_bp)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
