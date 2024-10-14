import subprocess
import torch
import re
import gc  # For garbage collection
import requests  # Added to send requests to the Ngrok URL
from transformers import AutoModelForCausalLM, AutoTokenizer

# os.system('ngrok authtoken <your-ngrok-authtoken>')
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


# Function to load the model and tokenizer
def load_model():
    """Load the model and tokenizer with optimized settings."""
    global model
    if 'model' in globals() and model is not None:
        print("Model is already loaded, skipping re-load.")
        return tokenizer, device
    model_name = "C:/Workarea/DSTV3.Danadrive.QA.Ai/AVA-Llama-3-V2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    return tokenizer, device


# Improved function to fix spacing in Persian
def fix_spacing(text: str) -> str:
    """Adds space before Persian punctuation marks."""
    return re.sub(r'(\S)([،؛?!])', r'\1 \2', text)


def remove_non_persian(text: str) -> str:
    """Keeps Persian characters and allows named entities and punctuation."""
    return re.sub(r'[^\u0600-\u06FF\s.,،؛?!\w]', '', text)


# Main function to summarize the text
def summarize_text(
        text: str,
        min_length: int = 100,  # Minimum token length for the summary
        max_length: int = 300,  # Maximum token length for the summary
        temperature: float = 0.3,  # Reduced randomness for better-quality results
        num_beams: int = 10,  # Increased number of beams for higher quality
        repetition_penalty: float = 1.5,  # Higher repetition penalty to avoid redundant phrases
        do_sample: bool = False  # Use greedy decoding for deterministic results
) -> str:
    """Summarizes the given text using the LLM model with controlled length."""

    # Prepare the input prompt for summarization
    input_text = f"لطفا این متن را به صورت خلاصه و به زبان فارسی بنویسید: {text}"

    # Tokenize the input text and move it to the appropriate device
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
    attention_mask = (inputs != tokenizer.pad_token_id).to(device)

    # Generate the summary
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_length + 50,  # Set the maximum length of the summary
            min_length=min_length,  # Set the minimum length of the summary
            repetition_penalty=repetition_penalty,  # Penalize repetitive phrases
            no_repeat_ngram_size=2,  # Avoid repeating any 2-grams
            temperature=temperature,  # Control randomness
            do_sample=do_sample,  # Sampling or greedy decoding
            num_beams=num_beams,  # Use beam search for better quality
            attention_mask=attention_mask,  # Apply attention mask
            early_stopping=True  # Stop generation early if conditions are met
        )

    # Decode the generated summary
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # If the generated summary repeats the input, clean it manually
    if input_text in summary:
        summary = summary.replace(input_text, "").strip()

    # Clean up non-Persian parts from the summary
    summary = remove_non_persian(summary).strip()

    # Further clean up potential model artifacts and fix spacing
    summary = fix_spacing(summary)

    return summary


# Ngrok URL
ngrok_url = "http://1c58-2a0d-5600-44-6003-00-21f0.ngrok-free.app"


# Function to send a request to the FastAPI summarization endpoint via Ngrok
def send_request_to_ngrok(text: str):
    """Send a request to the Ngrok URL."""
    payload = {
        "text": text,
        "min_length": 100,
        "max_length": 300,
        "temperature": 0.5,
        "num_beams": 5,
        "repetition_penalty": 1.2,
        "do_sample": False
    }
    response = requests.post(ngrok_url, json=payload)

    if response.status_code == 200:
        print("Summary:", response.json()["summary"])
    else:
        print(f"Error: {response.status_code}, {response.text}")


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

    # Step 4: Load the model and tokenizer
    tokenizer, device = load_model()

    # Step 5: Get input from user or use default text
    user_input = input("Please enter the text to summarize or press Enter to use default text:\n")
    default_text = "پایتخت فرانسه، پاریس است..."
    text_to_summarize = user_input if user_input.strip() else default_text

    # Step 6: Send the request to the API via Ngrok
    send_request_to_ngrok(text_to_summarize)

    # Step 7: Clean up after summarization
    cleanup()

except RuntimeError as e:
    print(f"RuntimeError: {e}")
