import subprocess
import torch
import re
import gc  # For garbage collection
import requests  # To send requests to the API
import json

# Ngrok URL (Hardcoded)
NGROK_URL = "http://63db-35-197-20-109.ngrok-free.app"  # Replace with your actual Ngrok URL


# Function to check GPU processes
def check_gpu_processes():
    """Check which processes are using the GPU."""
    try:
        gpu_processes = subprocess.check_output("nvidia-smi", shell=True).decode("utf-8")
        print(gpu_processes)
    except subprocess.CalledProcessError as e:
        print(f"Error checking GPU processes: {e}")


# Function to kill all processes using the GPU
def kill_gpu_processes():
    """Kill all processes using the GPU."""
    try:
        gpu_processes = subprocess.check_output("nvidia-smi --query-compute-apps=pid --format=csv,noheader",
                                                shell=True).decode("utf-8")
        pids = [pid.strip() for pid in gpu_processes.split("\n") if pid.strip()]
        for pid in pids:
            try:
                print(f"Killing process {pid}")
                subprocess.run(f"kill -9 {pid}", shell=True, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error killing process {pid}: {e}")
    except subprocess.CalledProcessError as e:
        print(f"Error fetching GPU processes: {e}")


# Function to release GPU memory and clean up resources
def cleanup():
    """Release GPU memory and clean up resources."""
    global model
    if 'model' in globals():
        del model
        torch.cuda.empty_cache()
        gc.collect()
    else:
        print("Model not defined, no need to clean up.")


# Improved function to fix spacing in Persian
def fix_spacing(text: str) -> str:
    """Adds space before Persian punctuation marks."""
    return re.sub(r'(\S)([،؛?!])', r'\1 \2', text)


# Main function to send a summarization request to the API
# def send_summarization_request(text: str, min_length: int, max_length: int):
#     """Send a summarization request to the FastAPI server via Ngrok."""
#     url = f"{NGROK_URL}/summarize"
#     payload = {
#         "text": text,
#         "min_length": min_length,
#         "max_length": max_length
#     }
#     headers = {"Content-Type": "application/json"}
#
#     try:
#         response = requests.post(url, data=json.dumps(payload), headers=headers)
#         if response.status_code == 200:
#             return response.json().get("summary", "No summary available.")
#         else:
#             print(f"Error: {response.status_code}, {response.content}")
#             return None
#     except requests.exceptions.RequestException as e:
#         print(f"Request failed: {e}")
#         return None
def send_summarization_request(text: str, min_length: int, max_length: int):
    """Send a summarization request to the FastAPI server via Ngrok."""
    url = f"{NGROK_URL}/summarize"
    payload = {
        "text": text,
        "min_length": min_length,
        "max_length": max_length
    }
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, data=json.dumps(payload), headers=headers, verify=False)  # Disable SSL verification
        if response.status_code == 200:
            return response.json().get("summary", "No summary available.")
        else:
            print(f"Error: {response.status_code}, {response.content}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None


# Main execution flow
try:
    # Step 1: Check current GPU processes
    print("Checking GPU processes before starting...")
    check_gpu_processes()

    # Step 2: Ask if the user wants to kill GPU processes
    kill_choice = input("Do you want to kill all GPU processes? (y/n): ")
    if kill_choice.lower() == "y":
        print("Killing all GPU processes to free up memory...")
        kill_gpu_processes()
    else:
        print("Proceeding without killing GPU processes...")

    # Step 3: Clean up any previous GPU memory usage
    cleanup()

    # Step 4: Get input from user or use default text
    user_input = input("Please enter the text to summarize or press Enter to use default text:\n")
    default_text = (
        "پایتخت فرانسه، پاریس است. پاریس یکی از مشهورترین شهرهای جهان به شمار می‌رود و به عنوان مرکز فرهنگی، اقتصادی و "
        "سیاسی فرانسه شناخته می‌شود. این شهر به خاطر جاذبه‌های گردشگری خود، مانند برج ایفل، موزه لوور و کلیسای نوتردام، "
        "شهرت جهانی دارد. پاریس همچنین به عنوان یک مرکز مهم در زمینه هنر، مد و تاریخ شناخته می‌شود."
    )
    text_to_summarize = user_input if user_input.strip() else default_text

    # Step 5: Get the number of paragraphs for summarization
    num_paragraphs = int(input("Enter the number of paragraphs for the summary: "))

    # Step 6: Set min and max tokens per paragraph
    min_tokens = 200 * num_paragraphs
    max_tokens = 500 * num_paragraphs

    # Step 7: Send request to the API to get the summary
    print("\nSending request to the API for summarization...")
    summary = send_summarization_request(text_to_summarize, min_length=min_tokens, max_length=max_tokens)

    if summary:
        print("\nSummary:")
        print(summary)

    # Step 8: Clean up after summarization
    cleanup()

except RuntimeError as e:
    print(f"RuntimeError: {e}")
