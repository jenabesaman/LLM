from flask import Flask, request, jsonify
import subprocess
import torch
import re
import gc
import requests
import json
import os
import re
import logging

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# Ngrok URL (Hardcoded) - Make sure this is the correct and current URL
NGROK_URL = "http://6f2a-34-142-249-57.ngrok-free.app"  # Replace with your actual Ngrok URL
DEFAULT_CONTEXT_API_URL = "https://develop.ddss.local:44305/api/Document/GetContents"

# Default folder path for confidential texts
DEFAULT_CONFIDENTIAL_FOLDER = "/content/drive/MyDrive/confidential_project/conf_files"

# Function to send bulk embedding request to the API
def send_bulk_embedding_request(texts, ngrok_url=None):
    api_url = f"{ngrok_url or NGROK_URL}/bulk-embedding"
    headers = {'Content-Type': 'application/json'}

    if not texts:
        return {"error": "No texts provided for embedding"}

    # Proper encoding for Persian and Unicode texts
    payload = json.dumps({"texts": texts}, ensure_ascii=False).encode('utf-8')

    logging.info(f"Sending bulk embedding request to: {api_url}")

    response = requests.post(api_url, headers=headers, data=payload)

    if response.status_code == 200:
        return response.json()
    else:
        logging.error(f"Failed to get a response. Status code: {response.status_code}, Response: {response.text}")
        return {
            "error": f"Failed to get a response. Status code: {response.status_code}",
            "details": response.text
        }

# Function to send a classification request to the API
def send_classification_request(text, ngrok_url=None):
    api_url = f"{ngrok_url or NGROK_URL}/classify-text"
    headers = {'Content-Type': 'application/json'}

    if not text:
        return {"error": "No text provided for classification"}

    # Proper encoding for Persian and Unicode texts
    payload = json.dumps({"text": text}, ensure_ascii=False).encode('utf-8')

    logging.info(f"Sending classification request to: {api_url}")

    response = requests.post(api_url, headers=headers, data=payload)

    if response.status_code == 200:
        return response.json()
    else:
        logging.error(f"Failed to get a response. Status code: {response.status_code}, Response: {response.text}")
        return {
            "error": f"Failed to get a response. Status code: {response.status_code}",
            "details": response.text
        }

# Function to send the question request to the main API
def send_question_to_api(base_url, query, context, prompt):
    api_url = f"{base_url}/process-question"
    headers = {'Content-Type': 'application/json'}
    payload = json.dumps({"query": query, "context": context, "prompt": prompt})

    logging.info(f"Sending question request to: {api_url}")

    try:
        response = requests.post(api_url, headers=headers, data=payload, timeout=700)
        if response.status_code == 200:
            return response.json()
        else:
            logging.error(f"Failed to get response. Status: {response.status_code}, Response: {response.text}")
            return {"error": f"Failed to get response. Status: {response.status_code}", "details": response.text}
    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed when sending question to API: {e}")
        return {"error": "Failed to connect to the API"}
# Function to fetch context from an external API
# Function to fetch context from the external API
# Function to fetch context from the external API without opening new pages
def fetch_context_from_api(api_url):
    headers = {'Content-Type': 'application/json'}

    try:
        logging.info(f"Fetching context from API: {api_url}")
        # Perform the GET request to the API
        response = requests.get(api_url, headers=headers, timeout=10, verify=False)

        if response.status_code == 200:
            # Parse the JSON response and extract 'contents'
            data = response.json()
            contents = data.get("contents", "")  # Extract 'contents' field
            if contents:
                logging.info("Successfully fetched contents from the API.")
                return contents
            else:
                logging.error("No 'contents' found in the API response.")
                return None
        else:
            logging.error(f"Failed to fetch context. Status: {response.status_code}, Response: {response.text}")
            return None

    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching context from API: {e}")
        return None



# Function to send a summarization request to the API
def send_summarization_request(text: str, prompt: str, min_length: int, max_length: int, ngrok_url=None):
    """Send a summarization request to the FastAPI server via Ngrok."""
    url = f"{ngrok_url or NGROK_URL}/summarize"
    payload = {
        "text": text,
        "prompt": prompt,
        "min_length": min_length,
        "max_length": max_length
    }
    headers = {"Content-Type": "application/json"}

    logging.info(f"Sending summarization request to: {url}")

    try:
        response = requests.post(url, data=json.dumps(payload), headers=headers, timeout=700, verify=False)
        if response.status_code == 200:
            return response.json().get("summary", "No summary available.")
        else:
            logging.error(f"Error: {response.status_code}, {response.content}")
            return None
    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed: {e}")
        return None

# Flask route for handling confidential embedding requests
@app.route("/process-text", methods=["POST"])
def process_text():
    data = request.get_json()
    ngrok_url = data.get("ngrok_url", None)

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
    logging.info("Processing text for embedding and classification...")
    embedding_response = send_bulk_embedding_request(texts, ngrok_url)
    classification_response = send_classification_request(text, ngrok_url)

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
    context = data.get('context')  # Check if user provided context
    prompt = data.get('prompt', "با توجه به متن زیر فقط به سوال پاسخ دهید و هیچ اطلاعات اضافی ندهید: ")

    if not query:
        return jsonify({"error": "'query' field is required."}), 400

    # Always fetch context from the external API if not provided by the user
    if not context or not context.strip():
        logging.info("Context is missing, fetching from the external API...")
        context = fetch_context_from_api(DEFAULT_CONTEXT_API_URL)
        if not context:
            return jsonify({"error": "Unable to fetch context from the external API."}), 500

    logging.info("Processing question request...")
    response_data = send_question_to_api(NGROK_URL, query, context, prompt)

    if 'error' in response_data:
        logging.error("Error in question response.")
        return jsonify(response_data), 500

    return jsonify(response_data), 200


# Flask route for handling summarization requests
@app.route('/summarize', methods=['POST'])
def summarize():
    """Handle summarization requests."""
    if not request.is_json:
        return jsonify({"error": "Request must be JSON with 'Content-Type: application/json' header"}), 415

    data = request.get_json()
    ngrok_url = data.get("ngrok_url", None)

    text = data.get('text')
    prompt = data.get('prompt', "لطفا این متن را به صورت خلاصه و به زبان فارسی بنویسید:")
    num_paragraphs = data.get('num_paragraphs', 1)

    if not text or not text.strip():
        return jsonify({"error": "The 'text' field cannot be empty."}), 400

    min_tokens = 100 * num_paragraphs
    max_tokens = 300 * num_paragraphs

    logging.info("Processing summarization request...")
    summary = send_summarization_request(text, prompt, min_tokens, max_tokens, ngrok_url)
    if summary:
        return jsonify({"summary": summary}), 200
    else:
        logging.error("Failed to generate summary.")
        return jsonify({"error": "Failed to generate summary"}), 500

from waitress import serve

if __name__ == '__main__':
    logging.info("Starting server on port 5001...")
    serve(app, host='0.0.0.0', port=5001, threads=6, connection_limit=1100)
