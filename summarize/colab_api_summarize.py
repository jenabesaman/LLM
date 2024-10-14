import os
import subprocess
import torch
import gc  # For garbage collection
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI, HTTPException, Request
from pyngrok import ngrok  # Ngrok for public URL exposure
import uvicorn

app = FastAPI()

# Global variables for model and tokenizer status
model_loaded = False
model = None
tokenizer = None
device = None


# Function to load the model and tokenizer
def load_model():
    """Load the model and tokenizer if they are not loaded."""
    global model, tokenizer, device, model_loaded
    model_path = "C:/Workarea/DSTV3.Danadrive.QA.Ai/AVA-Llama-3-V2"

    # Check if model directory exists before loading, else use Hugging Face base model
    if not os.path.exists(model_path):
        print(f"Model directory '{model_path}' does not exist, loading from Hugging Face...")
        model_name = "MehdiHosseiniMoghadam/AVA-Llama-3-V2"  # Hugging Face base model
    else:
        model_name = model_path

    if model_loaded:
        print("Model is already loaded, skipping re-load.")
        return tokenizer, device

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    model_loaded = True
    print("Model loaded successfully.")
    return tokenizer, device


# Endpoint to load the model
@app.post("/model-status")
async def model_status(request: Request):
    try:
        data = await request.json()
        action = data.get("action")

        if action == "load":
            load_model()
            return {"status": "Model loaded"}
        elif action == "cleanup":
            cleanup()
            return {"status": "Model cleaned up"}
        else:
            raise HTTPException(status_code=400, detail="Invalid action")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


# Function to check if the model is loaded before processing requests
def check_model_loaded():
    if not model_loaded:
        load_model()


# Summarization endpoint with model load check
@app.post("/summarize")
async def summarize(request: Request):
    check_model_loaded()  # Ensure the model is loaded before proceeding
    global model_loaded
    try:
        if not model_loaded:
            raise HTTPException(status_code=500, detail="Model not loaded. Please load the model first.")

        data = await request.json()
        text = data.get("text")
        min_length = data.get("min_length", 100)
        max_length = data.get("max_length", 300)

        if not text:
            raise HTTPException(status_code=400, detail="The 'text' field is required.")

        # Your summarization logic here...
        summary = summarize_text(text, min_length, max_length)
        return {"summary": summary}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


# Function to release GPU memory and clean up resources
def cleanup():
    """Release GPU memory and clean up resources."""
    global model, model_loaded
    if not model_loaded:
        print("Model not loaded, skipping cleanup.")
        return

    print("Cleaning up model and freeing GPU memory...")
    del model
    torch.cuda.empty_cache()
    gc.collect()
    model_loaded = False


# Function to summarize text (add your custom summarization logic here)
def summarize_text(text, min_length, max_length):
    """Summarizes the given text using the loaded model."""
    input_text = f"لطفا این متن را به صورت خلاصه و به زبان فارسی بنویسید: {text}"
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
    attention_mask = (inputs != tokenizer.pad_token_id).to(device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_length + 50,
            min_length=min_length,
            repetition_penalty=1.3,
            no_repeat_ngram_size=2,
            temperature=0.5,
            do_sample=False,
            num_beams=5,
            attention_mask=attention_mask,
            early_stopping=True
        )

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary.replace(input_text, "").strip()


import nest_asyncio
import uvicorn

# Apply nest_asyncio to handle the running event loop
nest_asyncio.apply()
# Start Ngrok tunnel and server
if __name__ == "__main__":
    # Start Ngrok tunnel with HTTP only (no TLS)
    public_url = ngrok.connect(8000, bind_tls=False)  # This disables TLS (HTTPS)
    print(f"Ngrok Tunnel URL: {public_url}")

    # Run Uvicorn server
    uvicorn.run(app, host="0.0.0.0", port=8000)
