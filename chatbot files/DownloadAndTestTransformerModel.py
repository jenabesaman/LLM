from sentence_transformers import SentenceTransformer
import os
import shutil
import json

# Define target directory - must match what's in your appsettings.json
target_dir = '/home/haystack2/models/all-MiniLM-L12-v2/'
os.makedirs(target_dir, exist_ok=True)
print(f"Target directory: {target_dir}")

# Download the model using the correct model name from Hugging Face
print("Downloading model...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')

# Save the model to the target directory
print(f"Saving model to {target_dir}")
model.save(target_dir)

print("Model download complete!")

# Create a simple test to verify the model works
test_sentence = "Testing the model installation"
embedding = model.encode(test_sentence)
print(f"Test successful! Generated embedding with {len(embedding)} dimensions")

# Check if model can be loaded from this path
try:
    test_model = SentenceTransformer(target_dir)
    print(f"? Successfully loaded model from {target_dir}")
    test_embedding = test_model.encode("Verification test")
    print(f"? Model works correctly from the saved path!")
except Exception as e:
    print(f"Error loading model from path: {e}")