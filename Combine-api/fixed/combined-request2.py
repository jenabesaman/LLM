from flask import Flask, request, jsonify
import subprocess
import torch
import re
import gc
import requests
import json
import os

app = Flask(__name__)

# Ngrok URL (Hardcoded) - Make sure this is the correct and current URL
NGROK_URL = "http://34d2-34-124-224-115.ngrok-free.app"  # Replace with your actual Ngrok URL

# Default folder path for confidential texts
DEFAULT_CONFIDENTIAL_FOLDER = "/content/drive/MyDrive/confidential_project/conf_files"

# Function to send bulk embedding request to the API
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

# Function to send a classification request to the API

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

# Function to send a question to the API
def send_question_to_api(base_url, query, context, prompt):
    api_url = f"{base_url}/process-question"
    headers = {'Content-Type': 'application/json'}
    payload = json.dumps({"query": query, "context": context, "prompt": prompt})

    try:
        response = requests.post(api_url, headers=headers, data=payload , timeout=700)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to get a response. Status code: {response.status_code}, Response: {response.text}")
            return {"error": f"Failed to get a response. Status code: {response.status_code}, Response: {response.text}"}
    except requests.exceptions.RequestException as e:
        print(f"Request failed when sending question to API: {e}")
        return {"error": "Failed to connect to the API"}

# Function to send a summarization request to the API
def send_summarization_request(text: str, prompt: str, min_length: int, max_length: int):
    """Send a summarization request to the FastAPI server via Ngrok."""
    url = f"{NGROK_URL}/summarize"
    payload = {
        "text": text,
        "prompt": prompt,
        "min_length": min_length,
        "max_length": max_length
    }
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, data=json.dumps(payload), headers=headers, timeout=700, verify=False)
        if response.status_code == 200:
            return response.json().get("summary", "No summary available.")
        else:
            print(f"Error: {response.status_code}, {response.content}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None

# Flask route for handling confidential embedding requests
# @app.route('/bulk-embedding', methods=['POST'])
# def bulk_embedding():
#     """Flask route to handle bulk embedding requests for confidential texts."""
#     data = request.json
#
#     if data and 'texts' in data:
#         texts = data.get('texts')
#         response_data = send_bulk_embedding_request(NGROK_URL, texts)
#         if 'error' in response_data:
#             return jsonify(response_data), 500
#         return jsonify(response_data), 200
#     else:
#         return jsonify({"error": "Request body must contain 'texts' field."}), 400

# Flask route for handling text classification requests
# @app.route('/classify-text', methods=['POST'])
# def classify_text():
#     """Flask route to handle text classification requests."""
#     data = request.json
#
#     if data and 'text' in data:
#         text = data.get('text')
#         response_data = send_classification_request(NGROK_URL, text)
#         if 'error' in response_data:
#             return jsonify(response_data), 500
#         return jsonify(response_data), 200
#     else:
#         return jsonify({"error": "Request body must contain 'text' field."}), 400

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


# Flask route for handling question requests
@app.route('/send-question', methods=['POST'])
def send_question():
    """Flask route to handle question requests."""
    data = request.json

    if not data:
        return jsonify({"error": "Request body must be in JSON format"}), 400

    query = data.get('query')
    context = data.get('context')
    prompt = data.get('prompt', "با توجه به متن زیر فقط به سوال پاسخ دهید و هیچ اطلاعات اضافی ندهید: ")

    if not query or not context:
        return jsonify({"error": "Both 'query' and 'context' fields are required."}), 400

    response_data = send_question_to_api(NGROK_URL, query, context, prompt)
    if 'error' in response_data:
        return jsonify(response_data), 500

    return jsonify(response_data), 200

# Flask route for handling summarization requests
@app.route('/summarize', methods=['POST'])
def summarize():
    """Handle summarization requests."""
    if not request.is_json:
        return jsonify({"error": "Request must be JSON with 'Content-Type: application/json' header"}), 415

    data = request.get_json()

    text = data.get('text')
    prompt = data.get('prompt', "لطفا این متن را به صورت خلاصه و به زبان فارسی بنویسید:")
    num_paragraphs = data.get('num_paragraphs', 1)

    if not text or not text.strip():
        return jsonify({"error": "The 'text' field cannot be empty."}), 400

    min_tokens = 100 * num_paragraphs
    max_tokens = 300 * num_paragraphs

    summary = send_summarization_request(text, prompt, min_tokens, max_tokens)
    if summary:
        return jsonify({"summary": summary}), 200
    else:
        return jsonify({"error": "Failed to generate summary"}), 500

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5001)


from waitress import serve

if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=5001, threads=6, connection_limit=700)
