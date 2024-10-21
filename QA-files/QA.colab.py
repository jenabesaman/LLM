1# prompt: modify code of model to ask question at end of code then print it without run on port 8000:

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import uvicorn
import nest_asyncio # import nest_asyncio
import requests
import json



def check_cuda():
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print("CUDA is available.")
    else:
        print("CUDA is not available.")


check_cuda()


# Define the question model
class Question(BaseModel):
    query: str

# Load the model and tokenizer with optimized settings
model_name = "./AVA-Llama-3-V2"  # Ensure this is the correct model path
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the model once when the app starts
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
)


# Read the context from test.txt
# context_file_path = os.path.join(os.path.dirname(__file__), 'data', 'test.txt')
# context_file_path = os.path.join('.', 'test.txt') # removed __file__ as it is not defined in this context.
# with open(context_file_path, 'r', encoding='utf-8') as f:
#     context = f.read()
#context="""پایتخت فرانسه، پاریس است. پاریس یکی از مشهورترین شهرهای جهان به شمار می‌رود و به عنوان مرکز فرهنگی، اقتصادی و سیاسی فرانسه شناخته می‌شود. این شهر به خاطر جاذبه‌های گردشگری خود، مانند برج ایفل، موزه لوور و کلیسای نوتردام، شهرت جهانی دارد. پاریس همچنین به عنوان یک مرکز مهم در زمینه هنر، مد و تاریخ شناخته می‌شود.
#"""



context = """پایتخت فرانسه، پاریس است. پاریس یکی از مشهورترین شهرهای جهان به شمار می‌رود و به عنوان مرکز فرهنگی، اقتصادی و سیاسی فرانسه شناخته می‌شود. این شهر به خاطر جاذبه‌های گردشگری خود، مانند برج ایفل، موزه لوور و کلیسای نوتردام، شهرت جهانی دارد. پاریس همچنین به عنوان یک مرکز مهم در زمینه هنر، مد و تاریخ شناخته می‌شود.
"""

with open('test2.txt', 'w', encoding='utf-8') as f:
  f.write(context)

with open('test2.txt', 'r', encoding='utf-8') as f:
  context_from_file = f.read()

#print(context_from_file)





# Create a FastAPI instance
app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Persian Q&A API using AVA-Llama-3. Use the /ask endpoint to ask questions."}

@app.post("/ask")
async def ask_question(question: Question):
    """Answer questions sent to the API"""
    # Prepare the input prompt for the model
    input_text = f"پاسخ به این سوال: {question.query} با توجه به متن زیر: {context}"
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)  # Ensure inputs are on the correct device

    # Generate the answer
    with torch.no_grad():
        outputs = model.generate(inputs, max_new_tokens=100)  # Adjust max_new_tokens as necessary

    # Decode the generated answer
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Post-process the answer to make it more readable
    answer = answer.replace(input_text, "").strip()

    if answer:
        return {"answer": answer}
    else:
        raise HTTPException(status_code=404, detail="No answer found.")


def send_request(query):
  # Simulate sending a request to the /ask endpoint without actually running the server
  # input_text = f"پاسخ به این سوال: {query} با توجه به متن زیر: {context}"
  input_text = f"پاسخ به این سوال: {query} با توجه به متن زیر: {context_from_file}"
  inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
  with torch.no_grad():
    outputs = model.generate(inputs, max_new_tokens=100)
  answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
  answer = answer.replace(input_text, "").strip()
  return {"answer": answer}

# Example usage
query = "به نظرت برم فرانسه یا نه؟"
response_data = send_request(query)
print(response_data)