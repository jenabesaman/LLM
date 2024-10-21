import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time  # Import time module for adding delays


def process_question(query):
    """
    Processes a question using a pre-trained model, saves context to a file, loads it, and returns the response.

    Parameters:
    - query (str): The question to ask the model.

    Returns:
    - dict: The response containing the answer.
    """

    # Hardcoded context data
    context = """
    پایتخت فرانسه، پاریس است. پاریس یکی از مشهورترین شهرهای جهان به شمار می‌رود و به عنوان مرکز فرهنگی، اقتصادی و سیاسی فرانسه شناخته می‌شود.
    این شهر به خاطر جاذبه‌های گردشگری خود، مانند برج ایفل، موزه لوور و کلیسای نوتردام، شهرت جهانی دارد. پاریس همچنین به عنوان یک مرکز مهم در زمینه هنر، مد و تاریخ شناخته می‌شود.
    """

    # Save the context to a file
    with open('context.txt', 'w', encoding='utf-8') as f:
        f.write(context)

    # Load the context from the file
    with open('context.txt', 'r', encoding='utf-8') as f:
        context_from_file = f.read()

    # Check CUDA availability and load the model/tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model_name = "../AVA-Llama-3-V2"
    model_name = "meta-llama/Meta-Llama-3-8B"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    ).to(device)

    # Add a short delay before generating the response to allow the GPU to stabilize
    time.sleep(1)  # 1-second delay; adjust as needed based on your GPU's performance

    # Prepare the input prompt for the model using the loaded context
    input_text = f"پاسخ به این سوال: {query} با توجه به متن زیر: {context_from_file}"
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)

    # Generate the answer
    with torch.no_grad():
        outputs = model.generate(inputs, max_new_tokens=100)

    # Decode the generated answer
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = answer.replace(input_text, "").strip()

    return {"answer": answer}


# Example usage
if __name__ == "__main__":
    query = "به نظرت برم فرانسه یا نه؟"
    response_data = process_question(query)
    print(response_data)
