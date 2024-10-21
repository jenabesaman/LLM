
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

# Define the question model
class Question(BaseModel):
    query: str

# Load the model and tokenizer with optimized settings
# model_name = "MehdiHosseiniMoghadam/AVA-Llama-3-V2"  # Ensure this is the correct model path
model_name = "AVA-Llama-3-V2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the model once when the app starts
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    offload_folder="offload",  # Specify the folder for offloading
    low_cpu_mem_usage=True,
).to(device)

# Read the context from test.txt
context_file_path = os.path.join(os.path.dirname(__file__), 'data', 'test.txt')
with open(context_file_path, 'r', encoding='utf-8') as f:
    context = f.read()

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

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)