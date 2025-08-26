from flask import Blueprint, request, jsonify
from transformers import pipeline

classify_bp = Blueprint("classify", __name__)

classifier = pipeline("text-classification", model="madhurjindal/Jailbreak-Detector")

@classify_bp.route("/classify", methods=["POST"])
def classify():
    data = request.get_json()
    if not data or "prompt" not in data:
        return jsonify({"error": "Missing 'prompt' field"}), 400

    result = classifier(data["prompt"])[0]
    return jsonify({
        "label": result["label"],
        "score": float(result["score"])
    })
