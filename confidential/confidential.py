from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import pickle
import torch

# Load the SentenceTransformer model for multilingual embeddings
device = 'cuda' if torch.cuda.is_available() else 'cpu'
embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', device=device)

def preprocess_text(text):
    """Preprocess the text (e.g., lowercasing, stripping whitespace)."""
    return text.lower().strip()

def get_embedding(text):
    """Generate embedding for a given text using SentenceTransformer after preprocessing."""
    text = preprocess_text(text)
    return embedding_model.encode(text, convert_to_tensor=True, device=device)

def save_embeddings(embeddings, file_path):
    """Save embeddings to a file for future use."""
    with open(file_path, 'wb') as f:
        pickle.dump(embeddings, f)

def load_embeddings(file_path):
    """Load embeddings from a file."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def load_confidential_texts(folder_path, save_path=None):
    """Load all text files from a given folder and return their embeddings. Optionally save the embeddings."""
    embeddings = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                embedding = get_embedding(text)
                embeddings.append(embedding)

    # Save embeddings if a save path is provided
    if save_path:
        save_embeddings(embeddings, save_path)

    return embeddings

def classify_text(input_text, confidential_embeddings, threshold=0.8, method='max'):
    """Classify a new text as confidential or not based on cosine similarity."""
    input_embedding = get_embedding(input_text)

    # Move the input embedding to CPU for cosine similarity calculation
    input_embedding = input_embedding.cpu()

    # Calculate similarities between the input text and all confidential embeddings
    similarities = [cosine_similarity(input_embedding.numpy().reshape(1, -1), emb.cpu().numpy().reshape(1, -1))[0][0] for emb in confidential_embeddings]

    # Choose the method of aggregation
    if method == 'max':
        similarity_score = max(similarities)
    elif method == 'avg':
        similarity_score = sum(similarities) / len(similarities)
    else:
        raise ValueError("Invalid method. Use 'max' or 'avg'.")

    return similarity_score >= threshold

# Folder paths
confidential_folder_path = "./conf_files"
embedding_save_path = "./conf_embeddings.pkl"

# Load confidential document embeddings (load from file if it exists, otherwise generate and save)
if os.path.exists(embedding_save_path):
    confidential_embeddings = load_embeddings(embedding_save_path)
else:
    confidential_embeddings = load_confidential_texts(confidential_folder_path, save_path=embedding_save_path)

# Example usage: classify a new document
new_file_path = "./not_conf_files/not_conf1.txt"
# new_file_path = "/content/drive/MyDrive/confidential_project/conf_files/conf3.txt"
with open(new_file_path, 'r', encoding='utf-8') as f:
    new_text = f.read()

# Classify the new text
is_confidential = classify_text(new_text, confidential_embeddings, method='avg')

if is_confidential:
    print("The document is classified as confidential.")
else:
    print("The document is not classified as confidential.")
