# Updated API Application (confidential_classification_api.py)
import os
import pickle
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException, Request
import uvicorn
from pyngrok import ngrok
import logging
from sklearn.metrics.pairwise import cosine_similarity
import torch

logging.basicConfig(level=logging.DEBUG)

app = FastAPI()

# Load the SentenceTransformer model
model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
model = SentenceTransformer(model_name, device='cuda' if torch.cuda.is_available() else 'cpu')

confidential_embeddings = []

# Endpoint to generate and store embeddings for confidential texts
@app.post("/bulk-embedding")
async def bulk_embedding(request: Request):
    global confidential_embeddings
    try:
        data = await request.json()
        texts = data.get("texts")

        if not texts or not isinstance(texts, list):
            raise HTTPException(status_code=400, detail="The 'texts' field is required and must be a list.")

        confidential_embeddings = model.encode(texts, convert_to_tensor=True, device='cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Generated embeddings for {len(texts)} texts.")

        return {"status": "Confidential embeddings generated and stored."}
    except Exception as e:
        logging.error(f"Error generating embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred while generating the embeddings: {str(e)}")

# Endpoint to classify a text as confidential or not
@app.post("/classify-text")
async def classify_text(request: Request):
    try:
        data = await request.json()
        text = data.get("text")

        if not text:
            raise HTTPException(status_code=400, detail="The 'text' field is required.")

        if len(confidential_embeddings) == 0:
            raise HTTPException(status_code=400, detail="Confidential embeddings are not available. Please load them first.")

        # Generate embedding for the input text
        input_embedding = model.encode(text, convert_to_tensor=True, device='cuda' if torch.cuda.is_available() else 'cpu').reshape(1, -1)

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
