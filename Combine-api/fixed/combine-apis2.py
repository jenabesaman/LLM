# Combined API Application (combined_api.py)
import os
import pickle
import torch
import gc
import logging
import uvicorn
import requests
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI, HTTPException, Request
from pyngrok import ngrok

logging.basicConfig(level=logging.DEBUG)

# Initialize FastAPI app
app = FastAPI()

# Global variables for the SentenceTransformer (embedding) model
embedding_model = None
confidential_embeddings = []

# Global variables for the LLaMA (summarization/QA) model
qa_model = None
qa_tokenizer = None
qa_device = None
qa_model_loaded = False

VRAM_THRESHOLD = 0.9  # VRAM usage threshold

# ---- Shared Functions ----

# Function to monitor VRAM and switch to CPU if needed
def check_vram_and_switch_to_cpu():
    if torch.cuda.is_available():
        total_vram = torch.cuda.get_device_properties(0).total_memory
        reserved_vram = torch.cuda.memory_reserved(0)
        vram_usage = reserved_vram / total_vram

        if vram_usage > VRAM_THRESHOLD:
            print("VRAM usage is high. Switching to CPU.")
            return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"

# Function to load the embedding model (SentenceTransformer)
def load_embedding_model():
    global embedding_model
    if embedding_model is None:
        print("Loading SentenceTransformer embedding model...")
        model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
        embedding_model = SentenceTransformer(
            model_name, device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        print("Embedding model loaded successfully.")
    else:
        print("Embedding model already loaded.")

# Function to load the summarization/QA model (LLaMA)
def load_qa_model():
    global qa_model, qa_tokenizer, qa_device, qa_model_loaded
    if not qa_model_loaded:
        print("Loading LLaMA model for summarization/QA...")
        model_path = "MehdiHosseiniMoghadam/AVA-Llama-3-V2"

        qa_tokenizer = AutoTokenizer.from_pretrained(model_path)
        qa_device = check_vram_and_switch_to_cpu()

        qa_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        qa_model_loaded = True
        print("LLaMA model loaded successfully.")
    else:
        print("LLaMA model already loaded.")

# Function to clean up the summarization model from memory
def cleanup_qa_model():
    global qa_model, qa_model_loaded
    if not qa_model_loaded:
        print("QA model not loaded, skipping cleanup.")
        return

    print("Cleaning up QA model and freeing GPU memory...")
    del qa_model
    torch.cuda.empty_cache()  # Free GPU memory
    gc.collect()  # Invoke Python's garbage collector
    qa_model_loaded = False

# Function for summarization
def summarize_text(text, prompt, min_length, max_length):
    input_text = f"{prompt}\nمتن: {text}"
    inputs = qa_tokenizer.encode(input_text, return_tensors="pt").to(qa_device)

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            outputs = qa_model.generate(
                inputs,
                max_new_tokens=max_length + 50,
                min_length=min_length,
                repetition_penalty=1.3,
                no_repeat_ngram_size=2,
                temperature=0.5,
                do_sample=False,
                num_beams=5,
                attention_mask=(inputs != qa_tokenizer.pad_token_id),
                early_stopping=True
            )
    summary = qa_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary.replace(input_text, "").strip()

# ---- API Endpoints ----

# Endpoint to generate and store embeddings for confidential texts
@app.post("/bulk-embedding")
async def bulk_embedding(request: Request):
    global confidential_embeddings
    load_embedding_model()  # Ensure the embedding model is loaded

    try:
        data = await request.json()
        texts = data.get("texts")

        if not texts or not isinstance(texts, list):
            raise HTTPException(status_code=400, detail="The 'texts' field is required and must be a list.")

        confidential_embeddings = embedding_model.encode(
            texts, convert_to_tensor=True, device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        logging.info(f"Generated embeddings for {len(texts)} texts.")

        return {"status": "Confidential embeddings generated and stored."}
    except Exception as e:
        logging.error(f"Error generating embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred while generating the embeddings: {str(e)}")

# Endpoint to classify a text as confidential or not
@app.post("/classify-text")
async def classify_text(request: Request):
    load_embedding_model()  # Ensure the embedding model is loaded

    try:
        data = await request.json()
        text = data.get("text")

        if not text:
            raise HTTPException(status_code=400, detail="The 'text' field is required.")

        if len(confidential_embeddings) == 0:
            raise HTTPException(status_code=400, detail="Confidential embeddings are not available. Please load them first.")

        # Generate embedding for the input text
        input_embedding = embedding_model.encode(
            text, convert_to_tensor=True, device='cuda' if torch.cuda.is_available() else 'cpu'
        ).reshape(1, -1)

        # Calculate cosine similarity with all confidential embeddings
        similarities = cosine_similarity(input_embedding.cpu(), confidential_embeddings.cpu())
        max_similarity = similarities.max()

        # Classify as confidential or not based on a threshold
        threshold = 0.8
        is_confidential = bool(max_similarity >= threshold)

        return {"is_confidential": is_confidential, "max_similarity": float(max_similarity)}
    except Exception as e:
        logging.error(f"Error classifying text: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred while classifying the text: {str(e)}")

# Model management endpoint
@app.post("/model-status")
async def model_status(request: Request):
    try:
        data = await request.json()
        action = data.get("action")

        if action == "load":
            load_qa_model()
            return {"status": "QA model loaded"}
        elif action == "cleanup":
            cleanup_qa_model()
            return {"status": "QA model cleaned up"}
        else:
            raise HTTPException(status_code=400, detail="Invalid action")
    except Exception as e:
        logging.error(f"Error in model-status endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# Summarization endpoint
@app.post("/summarize")
async def summarize(request: Request):
    load_qa_model()  # Ensure the QA model is loaded

    try:
        data = await request.json()
        text = data.get("text")
        prompt = data.get("prompt", "لطفا این متن را به صورت خلاصه و به زبان فارسی بنویسید:")
        min_length = data.get("min_length", 100)
        max_length = data.get("max_length", 300)

        if not text:
            raise HTTPException(status_code=400, detail="The 'text' field is required.")

        summary = summarize_text(text, prompt, min_length, max_length)
        return {"summary": summary}
    except Exception as e:
        logging.error(f"Error in summarize endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# Question answering endpoint
@app.post("/process-question")
async def process_question(request: Request):
    load_qa_model()  # Ensure the QA model is loaded

    try:
        data = await request.json()
        query = data.get("query")
        context = data.get("context")
        prompt = data.get("prompt", "با توجه به متن زیر فقط به سوال پاسخ دهید و هیچ اطلاعات اضافی ندهید: ")

        if not query or not context:
            raise HTTPException(status_code=400, detail="Both 'query' and 'context' fields are required.")

        input_text = f"{prompt}\nمتن: {context}\nسوال: {query}"
        inputs = qa_tokenizer.encode(input_text, return_tensors="pt").to(qa_device)

        with torch.no_grad():
            outputs = qa_model.generate(
                inputs,
                max_new_tokens=100,
                min_length=20,
                temperature=0.7,
                num_beams=3,
                repetition_penalty=1.2
            )

        answer = qa_tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = answer.replace(input_text, "").strip()
        return {"answer": answer}
    except Exception as e:
        logging.error(f"Error in process-question endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the question: {str(e)}")

# ---- Helper Functions ----

def check_model_loaded():
    if not qa_model_loaded:
        load_qa_model()

# Set environment variable to handle memory fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Apply nest_asyncio to handle the running event loop
import nest_asyncio
nest_asyncio.apply()

# Start Ngrok tunnel and server
if __name__ == "__main__":
    public_url = ngrok.connect(8000, bind_tls=False)  # Ngrok tunnel
    print(f"Ngrok Tunnel URL: {public_url}")

    uvicorn.run(app, host="0.0.0.0", port=8000)
