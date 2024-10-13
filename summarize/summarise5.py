import subprocess
import torch
import re
import gc  # For garbage collection
from transformers import AutoModelForCausalLM, AutoTokenizer
import os


# Function to check GPU processes
def check_gpu_processes():
    try:
        gpu_processes = subprocess.check_output("nvidia-smi", shell=True).decode("utf-8")
        print(gpu_processes)
    except subprocess.CalledProcessError as e:
        print(f"Error checking GPU processes: {e}")


# Function to kill all processes using the GPU
def kill_gpu_processes():
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
    global model
    if 'model' in globals():
        del model
        torch.cuda.empty_cache()
        gc.collect()
    else:
        print("Model not defined, no need to clean up.")


# # Function to load the model and tokenizer
# def load_model():
#     global model
#     if 'model' in globals() and model is not None:
#         print("Model is already loaded, skipping re-load.")
#         return tokenizer, device
#     model_name = "C:/Workarea/DSTV3.Danadrive.QA.Ai/AVA-Llama-3-V2"
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
#         device_map="auto",
#         low_cpu_mem_usage=True
#     )
#     return tokenizer, device

# Function to check if the model path exists and load the model accordingly
def load_model():
    """Load the model and tokenizer with optimized settings."""
    global model

    # Define the model path
    model_path = "C:/Workarea/DSTV3.Danadrive.QA.Ai/AVA-Llama-3-V2"

    # Check if model is already loaded to avoid reloading
    if 'model' in globals() and model is not None:
        print("Model is already loaded, skipping re-load.")
        return tokenizer, device

    # Check if the model path exists
    if not os.path.exists(model_path):
        print(f"Model path {model_path} does not exist. Loading model from Hugging Face.")
        model_name = "MehdiHosseiniMoghadam/AVA-Llama-3-V2"  # Use Hugging Face model
    else:
        print(f"Model path exists. Loading model from {model_path}.")
        model_name = model_path

    # Load the tokenizer and model
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
    return re.sub(r'(\S)([،؛?!])', r'\1 \2', text)


def remove_non_persian(text: str) -> str:
    return re.sub(r'[^\u0600-\u06FF\s.,،؛?!\w]', '', text)


# Main function to summarize the text
def summarize_text(
        text: str,
        min_length: int = 100,  # Minimum token length for the summary
        max_length: int = 300,  # Maximum token length for the summary
        temperature: float = 0.5,  # Controls the randomness; lower values make output more deterministic
        num_beams: int = 5,  # Number of beams for beam search; higher value improves quality at the cost of speed
        repetition_penalty: float = 1.3,  # Penalize repetition in the generated text
        do_sample: bool = False,  # Use greedy decoding (False) for deterministic results
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

    # Remove "متن اصلی" if it appears in the summary
    summary = summary.replace("متن اصلی", "").strip()

    summary = summary.replace("لطفا این متن را به صورت خلاصه و به زبان فارسی بنویسید: ", "").strip()

    # Clean up non-Persian parts from the summary
    summary = remove_non_persian(summary).strip()

    # Further clean up potential model artifacts and fix spacing
    summary = fix_spacing(summary)

    return summary



# Main execution flow
try:
    print("Checking GPU processes before starting...")
    check_gpu_processes()

    kill_choice = input("Do you want to kill all GPU processes? (y/n): ")
    if kill_choice.lower() == "y":
        print("Killing all GPU processes to free up memory...")
        kill_gpu_processes()
    else:
        print("Proceeding without killing GPU processes...")

    cleanup()
    tokenizer, device = load_model()

    user_input = input("Please enter the text to summarize or press Enter to use default text:\n")
    default_text=("پایتخت فرانسه، پاریس است. پاریس یکی از مشهورترین شهرهای جهان به شمار می‌رود و به عنوان مرکز فرهنگی، اقتصادی و سیاسی فرانسه شناخته می‌شود. این شهر به خاطر جاذبه‌های گردشگری خود، مانند برج ایفل، موزه لوور و کلیسای نوتردام، شهرت جهانی دارد. پاریس همچنین به عنوان یک مرکز مهم در زمینه هنر، مد و تاریخ شناخته می‌شود. فرانسه کشوری با تاریخ غنی است و پاریس در قلب این تاریخ جای دارد. این شهر همچنین مرکز تحولات بزرگ فرهنگی و تاریخی اروپا بوده است. پاریس به عنوان یکی از مهمترین شهرهای اروپایی همیشه نقش کلیدی در سیاست و فرهنگ جهانی داشته است. پاریس همچنین با ارائه ده‌ها رستوران معروف، کافه‌های تاریخی و بازارهای محلی، مرکز بزرگی برای فرهنگ غذا و هنر آشپزی به شمار می‌رود. پاریس یک شهر پر جنب و جوش با زندگی شبانه، جشنواره‌ها و هنرهای نمایشی است.")
    text_to_summarize = user_input if user_input.strip() else default_text

    num_paragraphs = int(input("Enter the number of paragraphs for the summary: "))
    min_tokens = 200 * num_paragraphs
    max_tokens = 500 * num_paragraphs

    summary = summarize_text(text_to_summarize, min_length=min_tokens, max_length=max_tokens)
    print("\nSummary:")
    print(summary)

    cleanup()
except RuntimeError as e:
    print(f"RuntimeError: {e}")
