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
def summarize_text(tokenizer, text, device, num_paragraphs=1):
    """
    Summarize the given text using the LLM model with controlled paragraph length.

    Args:
    tokenizer: The tokenizer to encode text.
    text (str): The input text to summarize.
    device (str): The device (cuda or cpu).
    num_paragraphs (int): The number of paragraphs for the summary.

    Returns:
    str: The generated summary with the specified number of paragraphs.
    """
    tokens_per_paragraph = 50
    max_new_tokens = num_paragraphs * tokens_per_paragraph
    input_text = f"لطفا این متن را به صورت خلاصه و به زبان فارسی بنویسید: {text}"

    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
    attention_mask = (inputs != tokenizer.pad_token_id).to(device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            min_length=int(max_new_tokens * 0.75),
            repetition_penalty=1.3,
            no_repeat_ngram_size=2,
            temperature=0.7,
            do_sample=True,
            num_beams=3,
            attention_mask=attention_mask,
            early_stopping=True
        )

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    summary = remove_non_persian(summary).strip()
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
        "پاریس همچنین به عنوان یک مرکز مهم در زمینه هنر، مد و تاریخ شناخته می‌شود. "
        "فرانسه کشوری با تاریخ غنی است و پاریس در قلب این تاریخ جای دارد. این شهر همچنین "
        "مرکز تحولات بزرگ فرهنگی و تاریخی اروپا بوده است. پاریس به عنوان یکی از مهمترین "
        "شهرهای اروپایی همیشه نقش کلیدی در سیاست و فرهنگ جهانی داشته است. "
        "پاریس همچنین با ارائه ده‌ها رستوران معروف، کافه‌های تاریخی و بازارهای محلی، مرکز "
        "بزرگی برای فرهنگ غذا و هنر آشپزی به شمار می‌رود. پاریس یک شهر پر جنب و جوش با "
        "زندگی شبانه، جشنواره‌ها و هنرهای نمایشی است."
    )

    text_to_summarize = user_input if user_input.strip() else default_text

    # Step 6: Get the number of paragraphs for summarization
    num_paragraphs = int(input("Enter the number of paragraphs for the summary: "))

    # Step 7: Generate the summary
    summary = summarize_text(tokenizer, text_to_summarize, device, num_paragraphs=num_paragraphs)
    print("\nSummary:")
    print(summary)

    # Step 8: Clean up after summarization
    cleanup()

except RuntimeError as e:
    print(f"RuntimeError: {e}")

# import subprocess
# import torch
# import re
# import gc  # For garbage collection
# from transformers import AutoModelForCausalLM, AutoTokenizer
#
# # Function to check GPU processes
# def check_gpu_processes():
#     """Check which processes are using the GPU."""
#     gpu_processes = subprocess.check_output("nvidia-smi", shell=True).decode("utf-8")
#     print(gpu_processes)
#
# # Function to kill all processes using the GPU (for Windows)
# def kill_gpu_processes():
#     """Kill all processes using the GPU."""
#     gpu_processes = subprocess.check_output("nvidia-smi --query-compute-apps=pid --format=csv,noheader", shell=True).decode("utf-8")
#     pids = [pid.strip() for pid in gpu_processes.split("\n") if pid.strip()]
#     for pid in pids:
#         try:
#             print(f"Killing process {pid}")
#             subprocess.run(f"taskkill /PID {pid} /F", shell=True, check=True)
#         except subprocess.CalledProcessError as e:
#             print(f"Error killing process {pid}: {e}")
#
# # Function to release GPU memory and clean up resources
# def cleanup():
#     """Release GPU memory and clean up resources."""
#     global model
#     if 'model' in globals():
#         del model
#         torch.cuda.empty_cache()
#         gc.collect()
#     else:
#         print("Model not defined, no need to clean up.")
#
# # Function to load the model and tokenizer
# def load_model():
#     """Load the model and tokenizer with optimized settings."""
#     global model
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
#
# # Function to fix spacing in Persian
# def fix_spacing(text):
#     """Adds space before Persian punctuation marks."""
#     return re.sub(r'(\S)([،؛?!])', r'\1 \2', text)
#
# # Function to filter out non-Persian characters but keep necessary punctuation
# def remove_non_persian(text):
#     """Keeps Persian characters and common Persian punctuation."""
#     return re.sub(r'[^\u0600-\u06FF\s.,،؛?!]', '', text)
#
# # Function to summarize the text
# def summarize_text(tokenizer, text, device, num_paragraphs=1):
#     tokens_per_paragraph = 10
#     max_new_tokens = num_paragraphs * tokens_per_paragraph
#
#     # Use a more direct summarization prompt
#     input_text = f"لطفا این متن را به یک پاراگراف خلاصه کنید: {text}"
#
#     inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
#     attention_mask = (inputs != tokenizer.pad_token_id).to(device)
#
#     # Adjust model generation parameters
#     with torch.no_grad():
#         outputs = model.generate(
#             inputs,
#             max_new_tokens=max_new_tokens,
#             min_length=int(max_new_tokens * 0.5),  # Ensure output isn't too long
#             repetition_penalty=2.0,  # Penalize repeated phrases more
#             no_repeat_ngram_size=3,  # Avoid repeating 3-grams
#             temperature=0.5,  # Control randomness, lower values make output more focused
#             num_beams=5,  # Use beam search for better quality
#             do_sample=False,  # Disable sampling for deterministic output
#             attention_mask=attention_mask,
#             early_stopping=True
#         )
#
#     # Decode and clean up the output
#     summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     summary = remove_non_persian(summary).strip()
#     summary = fix_spacing(summary)
#
#     return summary
#

# def summarize_text(tokenizer, text, device, num_paragraphs=1):
#     """
#     Summarize the given text using the LLM model with controlled paragraph length.
#     """
#     tokens_per_paragraph = 10
#     max_new_tokens = num_paragraphs * tokens_per_paragraph
#     input_text = f"لطفا این متن را به صورت خلاصه و به زبان فارسی بنویسید: {text}"
#     inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
#     attention_mask = (inputs != tokenizer.pad_token_id).to(device)
#
#     with torch.no_grad():
#         outputs = model.generate(
#             inputs,
#             max_new_tokens=max_new_tokens,
#             min_length=int(max_new_tokens * 0.75),
#             repetition_penalty=1.3,
#             no_repeat_ngram_size=2,
#             temperature=0.7,
#             do_sample=True,
#             num_beams=3,
#             attention_mask=attention_mask,
#             early_stopping=True
#         )
#
#     summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     summary = remove_non_persian(summary).strip()
#     summary = fix_spacing(summary)
#     return summary

# Main execution flow
try:
    # Ask if user wants to kill GPU processes
    kill_process_input = int(input("Enter 1 to kill GPU processes or 0 to skip: "))

    # Step 1: Check current GPU processes
    print("Checking GPU processes before starting...")
    check_gpu_processes()

    # Step 2: Kill all GPU processes if requested
    if kill_process_input == 1:
        print("Killing all GPU processes to free up memory...")
        kill_gpu_processes()
    else:
        print("Skipping killing GPU processes...")

    # Step 3: Clean up any previous GPU memory usage
    cleanup()

    # Step 4: Load the model and tokenizer
    tokenizer, device = load_model()

    # Step 5: Get the text to summarize from the user
    text_to_summarize = input("لطفاً متن مورد نظر برای خلاصه‌سازی را وارد کنید:\n")
    print(text_to_summarize)

    # Step 6: Get the number of paragraphs for summarization
    while True:
        num_paragraphs_input = input("تعداد پاراگراف‌های خلاصه را وارد کنید: ")
        if num_paragraphs_input.strip():
            try:
                num_paragraphs = int(num_paragraphs_input)
                break
            except ValueError:
                print("Please enter a valid number.")
        else:
            print("Input cannot be empty.")

    # Step 7: Generate the summary
    summary = summarize_text(tokenizer, text_to_summarize, device, num_paragraphs=num_paragraphs)
    print("\nخلاصه متن:")
    print(summary)

    # Step 8: Clean up after summarization
    cleanup()

except RuntimeError as e:
    print(f"RuntimeError: {e}")
except ValueError as ve:
    print(f"Invalid input: {ve}")


# import subprocess
# import torch
# import re
# import gc  # For garbage collection
# from transformers import AutoModelForCausalLM, AutoTokenizer
#
# # Function to check GPU processes
# def check_gpu_processes():
#     """Check which processes are using the GPU."""
#     # Run the nvidia-smi command to list GPU processes
#     gpu_processes = subprocess.check_output("nvidia-smi", shell=True).decode("utf-8")
#     print(gpu_processes)
#
# # Function to kill all processes using the GPU
# def kill_gpu_processes():
#     """Kill all processes using the GPU."""
#     # Get the list of processes using the GPU with nvidia-smi and extract the PIDs
#     gpu_processes = subprocess.check_output("nvidia-smi --query-compute-apps=pid --format=csv,noheader", shell=True).decode("utf-8")
#
#     # Split the output into lines and extract PIDs
#     pids = [pid.strip() for pid in gpu_processes.split("\n") if pid.strip()]
#
#     # Kill each process
#     for pid in pids:
#         try:
#             print(f"Killing process {pid}")
#             subprocess.run(f"kill -9 {pid}", shell=True, check=True)
#         except subprocess.CalledProcessError as e:
#             print(f"Error killing process {pid}: {e}")
#
# # Function to release GPU memory and clean up resources
# def cleanup():
#     """Release GPU memory and clean up resources."""
#     global model
#     if 'model' in globals():  # Check if model is defined before deleting it
#         del model  # Delete the model to release memory
#         torch.cuda.empty_cache()  # Clear GPU cache
#         gc.collect()  # Trigger garbage collection
#     else:
#         print("Model not defined, no need to clean up.")
#
# # Function to load the model and tokenizer
# def load_model():
#     """Load the model and tokenizer with optimized settings."""
#     global model
#     model_name = "C:/Workarea/DSTV3.Danadrive.QA.Ai/AVA-Llama-3-V2"
#     # model_name = "MehdiHosseiniMoghadam/AVA-Llama-3-V2"
#     # C:\Workarea\DSTV3.Danadrive.QA.Ai
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#
#     # Determine the device (GPU or CPU)
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#
#     # Load the model
#     model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,  # Use FP16 if on GPU
#         device_map="auto",               # Automatically map model to GPU if available
#         low_cpu_mem_usage=True           # Optimize memory usage for large models
#     )
#
#     return tokenizer, device
#
# # Function to fix spacing in Persian
# def fix_spacing(text):
#     """Adds space before Persian punctuation marks."""
#     return re.sub(r'(\S)([،؛?!])', r'\1 \2', text)
#
# # Function to filter out non-Persian characters but keep necessary punctuation
# def remove_non_persian(text):
#     """Keeps Persian characters and common Persian punctuation."""
#     return re.sub(r'[^\u0600-\u06FF\s.,،؛?!]', '', text)
#
# # Function to summarize the text
# def summarize_text(tokenizer, text, device, num_paragraphs=1):
#     """
#     Summarize the given text using the LLM model with controlled paragraph length.
#
#     Args:
#     tokenizer: The tokenizer to encode text.
#     text (str): The input text to summarize.
#     device (str): The device (cuda or cpu).
#     num_paragraphs (int): The number of paragraphs for the summary.
#
#     Returns:
#     str: The generated summary with the specified number of paragraphs.
#     """
#     # Calculate approximate token length per paragraph (100 tokens per paragraph)
#     tokens_per_paragraph = 10
#     max_new_tokens = num_paragraphs * tokens_per_paragraph
#
#     # Prepare the input prompt for summarization
#     input_text = f"لطفا این متن را به صورت خلاصه و به زبان فارسی بنویسید: {text}"
#
#     # Tokenize the input text and move it to the same device as the model
#     inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)  # Ensure tensors are on the same device
#     attention_mask = (inputs != tokenizer.pad_token_id).to(device)  # Create attention mask and move to same device
#
#     # Generate the summary
#     with torch.no_grad():
#         outputs = model.generate(
#             inputs,
#             max_new_tokens=max_new_tokens,   # Set the maximum length of the summary based on paragraph count
#             min_length=int(max_new_tokens * 0.75),  # Ensure at least 75% of the max length is used
#             repetition_penalty=1.3,          # Penalize repetitive phrases
#             no_repeat_ngram_size=2,          # Avoid repeating any 2-grams
#             temperature=0.7,                 # Control randomness
#             do_sample=True,                  # Enable sampling for more diverse output
#             num_beams=3,                     # Use beam search for better quality
#             attention_mask=attention_mask,   # Apply attention mask
#             early_stopping=True              # Stop generation early if no more tokens can be predicted
#         )
#
#     # Decode the generated summary
#     summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
#
#     # Clean up non-Persian parts from the summary
#     summary = remove_non_persian(summary).strip()
#
#     # Further clean up potential model artifacts and fix spacing
#     summary = fix_spacing(summary)
#
#     # Return the summary
#     return summary
#
# # Main execution flow
# try:
#     # Step 1: Check current GPU processes
#     print("Checking GPU processes before starting...")
#     check_gpu_processes()
#
#     # Step 2: Kill all GPU processes to free up memory if needed
#     print("Killing all GPU processes to free up memory...")
#     kill_gpu_processes()
#
#     # Step 3: Clean up any previous GPU memory usage
#     cleanup()
#
#     # Step 4: Load the model and tokenizer
#     tokenizer, device = load_model()
#
#     # Step 5: Get the text to summarize from the user
#     text_to_summarize = input("لطفاً متن مورد نظر برای خلاصه‌سازی را وارد کنید:\n")
#     print(text_to_summarize)
#     # Step 6: Get the number of paragraphs for summarization
#     num_paragraphs = int(input("تعداد پاراگراف‌های خلاصه را وارد کنید: "))
#
#     # Step 7: Generate the summary
#     summary = summarize_text(tokenizer, text_to_summarize, device, num_paragraphs=num_paragraphs)
#     print("\nخلاصه متن:")
#     print(summary)
#
#     # Step 8: Clean up after summarization
#     cleanup()
#
# except RuntimeError as e:
#     print(f"RuntimeError: {e}")
