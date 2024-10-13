from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re

# Load the model and tokenizer with optimized settings
model_name = "C:/Workarea/DSTV3.Danadrive.QA.Ai/AVA-Llama-3-V2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the model with optimized memory usage
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,       # Use FP16 to reduce memory usage
    device_map="auto",               # Automatically map model to GPU if available
    low_cpu_mem_usage=True           # Optimize memory usage for large models
)

# Improved function to fix spacing in Persian
def fix_spacing(text):
    """Adds space before Persian punctuation marks."""
    return re.sub(r'(\S)([،؛?!])', r'\1 \2', text)

# Function to filter out non-Persian characters but keep necessary punctuation
def remove_non_persian(text):
    """Keeps Persian characters and common Persian punctuation."""
    return re.sub(r'[^\u0600-\u06FF\s.,،؛?!]', '', text)

def summarize_text(text, min_length=50, max_length=150):
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
    attention_mask = (inputs != tokenizer.pad_token_id).to(device)  # Create attention mask

    # Generate the summary
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_length,       # Set the maximum length of the summary
            min_length=min_length,           # Set the minimum length of the summary
            repetition_penalty=1.3,          # Penalize repetitive phrases
            no_repeat_ngram_size=2,          # Avoid repeating any 2-grams
            temperature=0.7,                 # Control randomness
            do_sample=True,                  # Enable sampling for more diverse output
            num_beams=3,                     # Use beam search for better quality
            attention_mask=attention_mask,   # Apply attention mask
            early_stopping=True              # Stop generation early if no more tokens can be predicted
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

# Example usage for summarization
text_to_summarize = (
    "پایتخت فرانسه، پاریس است. پاریس یکی از مشهورترین شهرهای جهان به شمار می‌رود "
    "و به عنوان مرکز فرهنگی، اقتصادی و سیاسی فرانسه شناخته می‌شود. این شهر به خاطر "
    "جاذبه‌های گردشگری خود، مانند برج ایفل، موزه لوور و کلیسای نوتردام، شهرت جهانی دارد. "
    "پاریس همچنین به عنوان یک مرکز مهم در زمینه هنر، مد و تاریخ شناخته می‌شود."
)

# Set desired summary length: min_length=50, max_length=100
summary = summarize_text(text_to_summarize, min_length=10, max_length=20)
print(summary)
