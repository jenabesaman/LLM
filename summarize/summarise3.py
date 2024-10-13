import subprocess
import torch
import re
import gc  # For garbage collection
from transformers import AutoModelForCausalLM, AutoTokenizer


# Function to check GPU processes
def check_gpu_processes():
    """Check which processes are using the GPU."""
    gpu_processes = subprocess.check_output("nvidia-smi", shell=True).decode("utf-8")
    print(gpu_processes)


# Function to kill all processes using the GPU
def kill_gpu_processes():
    """Kill all processes using the GPU."""
    gpu_processes = subprocess.check_output("nvidia-smi --query-compute-apps=pid --format=csv,noheader",
                                            shell=True).decode("utf-8")
    pids = [pid.strip() for pid in gpu_processes.split("\n") if pid.strip()]
    for pid in pids:
        try:
            print(f"Killing process {pid}")
            subprocess.run(f"kill -9 {pid}", shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error killing process {pid}: {e}")


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


# Function to fix spacing in Persian
def fix_spacing(text):
    """Adds space before Persian punctuation marks."""
    return re.sub(r'(\S)([،؛?!])', r'\1 \2', text)


# Function to filter out non-Persian characters but keep necessary punctuation


def remove_non_persian(text):
    """Keeps Persian characters and common Persian punctuation."""
    return re.sub(r'[^\u0600-\u06FF\s.,،؛?!]', '', text)


# Function to summarize the text
def summarize_text(text, min_length=100, max_length=300):
    """
    Summarize the given text using the LLM model with controlled length.

    Args:
    text (str): The input text to summarize.
    min_length (int): The minimum length of the summary (in tokens).
    max_length (int): The maximum length of the summary (in tokens).

    Returns:
    str: The generated summary.
    """
    # Prepare the input prompt for summarization
    input_text = f"لطفا این متن را به صورت خلاصه و به زبان فارسی بنویسید: {text}"

    # Tokenize the input text and move it to the appropriate device
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
    attention_mask = (inputs != tokenizer.pad_token_id).to(device)

    # Generate the summary
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_length,
            min_length=min_length,
            repetition_penalty=1.3,
            no_repeat_ngram_size=2,
            temperature=0.5,  # Lower temperature for less randomness
            do_sample=False,  # Use greedy decoding
            num_beams=5,  # Increase the number of beams for better results
            attention_mask=attention_mask,
            early_stopping=True
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

    default_text = (
        "پایتخت فرانسه، پاریس است. پاریس یکی از مشهورترین شهرهای جهان به شمار می‌رود "
        "و به عنوان مرکز فرهنگی، اقتصادی و سیاسی فرانسه شناخته می‌شود. این شهر به خاطر "
        "جاذبه‌های گردشگری خود، مانند برج ایفل، موزه لوور و کلیسای نوتردام، شهرت جهانی دارد. "
        "پاریس همچنین به عنوان یک مرکز مهم در زمینه هنر، مد و تاریخ شناخته می‌شود."
    )

    text_to_summarize = user_input if user_input.strip() else default_text

    # Step 6: Get the number of paragraphs for summarization
    num_paragraphs = int(input("Enter the number of paragraphs for the summary: "))

    # Step 7: Set min and max tokens per paragraph
    min_tokens = 200 * num_paragraphs
    max_tokens = 500 * num_paragraphs

    # Step 8: Generate the summary
    summary = summarize_text(text_to_summarize, min_length=min_tokens, max_length=max_tokens)
    print("\nSummary:")
    print(summary)

    # Step 9: Clean up after summarization
    cleanup()

except RuntimeError as e:
    print(f"RuntimeError: {e}")
