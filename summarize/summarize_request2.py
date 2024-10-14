import requests
import json


# Function to send summarization request to the API through ngrok
def summarize_text_api(
        text: str,
        min_length: int = 100,
        max_length: int = 300,
        temperature: float = 0.5,
        num_beams: int = 5,
        repetition_penalty: float = 1.3,
        do_sample: bool = False
) -> str:
    """Sends a summarization request to the ngrok-exposed API."""

    # The ngrok URL from the API you provided
    ngrok_url = "http://1c58-2a0d-5600-44-6003-00-21f0.ngrok-free.app"

    headers = {'Content-Type': 'application/json'}
    data = {
        "text": text,
        "min_length": min_length,
        "max_length": max_length,
        "temperature": temperature,
        "num_beams": num_beams,
        "repetition_penalty": repetition_penalty,
        "do_sample": do_sample
    }

    try:
        # Send the request to the ngrok URL
        response = requests.post(ngrok_url, headers=headers, data=json.dumps(data))
        response.raise_for_status()  # Raise an error for bad responses
        result = response.json()
        return result.get("summary", "No summary returned")
    except requests.exceptions.RequestException as e:
        return f"Error during API call: {e}"


# Main execution flow
try:
    # Step 1: Get input from user or use default text
    user_input = input("Please enter the text to summarize or press Enter to use default text:\n")
    default_text = (
        "پایتخت فرانسه، پاریس است. پاریس یکی از مشهورترین شهرهای جهان به شمار می‌رود و به عنوان مرکز فرهنگی، "
        "اقتصادی و سیاسی فرانسه شناخته می‌شود. این شهر به خاطر جاذبه‌های گردشگری خود، مانند برج ایفل، موزه لوور و "
        "کلیسای نوتردام، شهرت جهانی دارد. پاریس همچنین به عنوان یک مرکز مهم در زمینه هنر، مد و تاریخ شناخته می‌شود. "
        "فرانسه کشوری با تاریخ غنی است و پاریس در قلب این تاریخ جای دارد. این شهر همچنین مرکز تحولات بزرگ فرهنگی و "
        "تاریخی اروپا بوده است. پاریس به عنوان یکی از مهمترین شهرهای اروپایی همیشه نقش کلیدی در سیاست و فرهنگ جهانی "
        "داشته است. پاریس همچنین با ارائه ده‌ها رستوران معروف، کافه‌های تاریخی و بازارهای محلی، مرکز بزرگی برای فرهنگ "
        "غذا و هنر آشپزی به شمار می‌رود. پاریس یک شهر پر جنب و جوش با زندگی شبانه، جشنواره‌ها و هنرهای نمایشی است.")

    text_to_summarize = user_input if user_input.strip() else default_text

    # Step 2: Get the number of paragraphs for summarization
    num_paragraphs = int(input("Enter the number of paragraphs for the summary: "))

    # Step 3: Set min and max tokens per paragraph
    min_tokens = 200 * num_paragraphs
    max_tokens = 500 * num_paragraphs

    # Step 4: Call the summarize_text_api function to get the summary
    summary = summarize_text_api(text_to_summarize, min_length=min_tokens, max_length=max_tokens)

    print("\nSummary:")
    print(summary)

except RuntimeError as e:
    print(f"RuntimeError: {e}")
