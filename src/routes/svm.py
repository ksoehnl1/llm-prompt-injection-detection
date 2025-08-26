from flask import Blueprint, request, jsonify
from detector import MaliciousPromptDetector

predict_bp = Blueprint("predict", __name__)

try:
    detector = MaliciousPromptDetector(model_dir="models")
except Exception as e:
    print(f"CRITICAL ERROR: Failed to initialize MaliciousPromptDetector: {e}")
    detector = None

@predict_bp.route('/predict', methods=['POST'])
def predict_endpoint():
    if detector is None:
        return jsonify({"error": "Server is not operational; models failed to load."}), 503

    if not request.json or 'text' not in request.json:
        return jsonify({"error": "Invalid request. JSON body must contain a 'text' key."}), 400

    text_to_classify = request.json['text']
    try:
        result = detector.predict(text_to_classify)[0]
        return jsonify(result)
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return jsonify({"error": "An internal error occurred during prediction."}), 500
