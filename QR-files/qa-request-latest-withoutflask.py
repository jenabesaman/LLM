# Request Application (request_app.py)
import requests
import json

NGROK_URL = "http://cd38-34-91-74-91.ngrok-free.app"


def load_model_via_api(base_url):
    api_url = f"{base_url}/load-model"
    headers = {'Content-Type': 'application/json'}
    response = requests.post(api_url, headers=headers)

    if response.status_code == 200:
        print("Model loaded successfully via API.")
    else:
        print(f"Failed to load model. Status code: {response.status_code}, Response: {response.text}")


def send_question_to_api(base_url, query, context, prompt):
    api_url = f"{base_url}/process-question"
    headers = {'Content-Type': 'application/json'}
    payload = json.dumps({"query": query, "context": context, "prompt": prompt})

    response = requests.post(api_url, headers=headers, data=payload)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to get a response. Status code: {response.status_code}, Response: {response.text}")
        return {"error": f"Failed to get a response. Status code: {response.status_code}"}


if __name__ == "__main__":
    base_url = NGROK_URL
    load_model_via_api(base_url)
    query = input("Enter your question: ")
    if not query:
        query = "به نظرت برم فرانسه یا نه؟"
    context = input("Enter the context: ")
    if not context:
        context = "پایتخت فرانسه، پاریس است. پاریس یکی از مشهورترین شهرهای جهان به شمار می‌رود و به عنوان مرکز فرهنگی، اقتصادی و سیاسی فرانسه شناخته می‌شود. این شهر به خاطر جاذبه‌های گردشگری خود، مانند برج ایفل، موزه لوور و کلیسای نوتردام، شهرت جهانی دارد. پاریس همچنین به عنوان یک مرکز مهم در زمینه هنر، مد و تاریخ شناخته می‌شود."
    prompt = input("Enter the prompt: ")
    if not prompt:
        prompt = "با توجه به متن به سوال پاسخ دهید:"
    response_data = send_question_to_api(base_url, query, context, prompt)
    print(response_data)