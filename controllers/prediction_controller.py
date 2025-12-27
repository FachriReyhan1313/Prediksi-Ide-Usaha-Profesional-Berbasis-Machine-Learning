from flask import Blueprint, request, render_template
from services.prediction_service import run_prediction

prediction_bp = Blueprint("prediction", __name__)

@prediction_bp.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        result = run_prediction(request.form)

    return render_template("index.html", result=result)
