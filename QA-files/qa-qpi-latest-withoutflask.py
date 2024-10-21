# API Application (qa_api.py)
import os
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI, HTTPException, Request
import uvicorn
from pyngrok import ngrok
import logging

logging.basicConfig(level=logging.DEBUG)

app = FastAPI()

# Global variables for model and tokenizer
model_loaded = False
model = None
tokenizer = None
device = None


# Function to load the model and tokenizer
def load_model():
    global model, tokenizer, device, model_loaded
    model_path = "MehdiHosseiniMoghadam/AVA-Llama-3-V2"

    try:
        if model_loaded:
            print("Model is already loaded, skipping re-load.")
            return tokenizer, device

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        model_loaded = True
        print("Model loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        raise e


# Endpoint to load the model
@app.post("/load-model")
async def load_model_endpoint():
    try:
        load_model()
        return {"status": "Model loaded"}
    except Exception as e:
        logging.error(f"Error in load_model_endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred while loading the model: {str(e)}")


# Endpoint to process a question
@app.post("/process-question")
async def process_question(request: Request):
    try:
        data = await request.json()
        query = data.get("query")
        context = data.get("context")
        prompt = data.get("prompt")

        if not query:
            raise HTTPException(status_code=400, detail="The 'query' field is required.")
        if not context:
            raise HTTPException(status_code=400, detail="The 'context' field is required.")
        if not prompt:
            prompt = "با توجه به متن به سوال پاسخ دهید:"

        # Ensure model is loaded
        if not model_loaded:
            load_model()

        # Prepare the input prompt for the model using the loaded context
        input_text = f"{prompt} {query} با توجه به متن زیر: {context}"
        inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)

        # Generate the answer
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=150,  # Increased from 100 to 150 for a more elaborate answer
                min_length=20,  # Set a minimum length to ensure some output is generated
                temperature=0.7,  # Adjusted temperature for more diverse responses
                num_beams=3,  # Using beam search to explore multiple possibilities
                repetition_penalty=1.2  # Penalty to avoid repetitive outputs
            )

        # Decode the generated answer
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = answer.replace(input_text, "").strip()
        answer = answer.replace(prompt, "").strip()
        answer = answer.replace("\u200c", " ").strip()

        return {"answer": answer}

    except Exception as e:
        logging.error(f"Error processing question: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the question: {str(e)}")


# Apply nest_asyncio to handle the running event loop
import nest_asyncio

nest_asyncio.apply()

# Start Ngrok tunnel and server
if __name__ == "__main__":
    # Start Ngrok tunnel with HTTP only (no TLS)
    public_url = ngrok.connect(8000, bind_tls=False)  # This disables TLS (HTTPS)
    print(f"Ngrok Tunnel URL: {public_url}")

    # Run Uvicorn server
    uvicorn.run(app, host="0.0.0.0", port=8000)