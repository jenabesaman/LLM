# Flask API for Handling Embeddings and Classification (confidential_flask_api.py)
from flask import Flask, request, jsonify
import requests
import json

NGROK_URL = "http://27fd-151-243-15-201.ngrok-free.app"

app = Flask(__name__)

def send_bulk_embedding_request(texts):
    api_url = f"{NGROK_URL}/bulk-embedding"
    headers = {'Content-Type': 'application/json'}

    if not texts:
        return {"error": "No texts provided for embedding"}

    # Proper encoding for Persian and Unicode texts
    payload = json.dumps({"texts": texts}, ensure_ascii=False).encode('utf-8')

    response = requests.post(api_url, headers=headers, data=payload)

    if response.status_code == 200:
        return response.json()
    else:
        return {
            "error": f"Failed to get a response. Status code: {response.status_code}",
            "details": response.text
        }

def send_classification_request(text):
    api_url = f"{NGROK_URL}/classify-text"
    headers = {'Content-Type': 'application/json'}

    if not text:
        return {"error": "No text provided for classification"}

    # Proper encoding for Persian and Unicode texts
    payload = json.dumps({"text": text}, ensure_ascii=False).encode('utf-8')

    response = requests.post(api_url, headers=headers, data=payload)

    if response.status_code == 200:
        return response.json()
    else:
        return {
            "error": f"Failed to get a response. Status code: {response.status_code}",
            "details": response.text
        }

@app.route("/process-text", methods=["POST"])
def process_text():
    data = request.get_json()
    if not data or "texts" not in data or "text" not in data:
        return jsonify({"error": "Invalid input. 'texts' and 'text' keys are required."}), 400

    texts_input = data.get("texts")
    text = data.get("text")

    # Split 'texts' by ',' if it's a single string
    if isinstance(texts_input, str):
        texts = [t.strip() for t in texts_input.split('","') if t.strip()]
    else:
        return jsonify({"error": "'texts' must be a string with texts separated by ','."}), 400

    if not isinstance(text, str):
        return jsonify({"error": "'text' must be a string."}), 400

    # Perform both embedding and classification
    embedding_response = send_bulk_embedding_request(texts)
    classification_response = send_classification_request(text)

    return jsonify({
        "embedding_response": embedding_response,
        "classification_response": classification_response
    }), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
