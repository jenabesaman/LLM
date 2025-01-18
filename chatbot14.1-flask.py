import os
import queue
import time
from flask import Flask, request, jsonify
import threading
import requests
import logging
from haystack.components.generators.chat import HuggingFaceLocalChatGenerator
import torch
from haystack import Document
from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore
from haystack_integrations.components.retrievers.elasticsearch import ElasticsearchEmbeddingRetriever
from haystack.document_stores.types import DuplicatePolicy
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
from haystack.dataclasses import ChatMessage
from itertools import chain
from typing import Any, List
from haystack import Pipeline, component
from haystack.core.component.types import Variadic
from haystack.components.builders import ChatPromptBuilder

logging.basicConfig(level=logging.INFO)
ElasticAuth =("elastic", "PVdbNWLf=iJXIh3fnywZ")
ElasticUrl="https://localhost:9200/"
# Define embedding model
#EmbedderModel = "sentence-transformers/all-MiniLM-L12-v2"
EmbedderModel="/home/Haystack2/sentence-transformers-all-MiniLM-L12-v2/"
# Initialize model
local_model_path = "/home/Haystack2/AVA-Llama-3-V2/"
#local_model_path = "E:/Workarea/ai-chatbots/Haystack.llama.cpp/models/AVA-Llama-3-V2"

# Add this near the top of your file with other global variables
MODEL_INITIALIZED = False
def initialize_model():
    global MODEL_INITIALIZED
    if MODEL_INITIALIZED:
        return True
    """
    Initialize and warm up the model before any requests are made.
    """
    try:
        logging.info("Initializing model...")

        # Warm up the text embedder
        text_embedder = SentenceTransformersTextEmbedder(model=EmbedderModel, progress_bar=True, prefix="query:")
        text_embedder.warm_up()
        logging.info("Text embedder initialized and warmed up")

        # Warm up the document embedder
        document_embedder.warm_up()
        logging.info("Document embedder initialized and warmed up")

        # Warm up the LLM
        system_message = ChatMessage.from_system("Test initialization")
        test_message = ChatMessage.from_user("Hello")
        messages = [system_message, test_message]

        # Run a test inference
        res = pipeline.run(
            data={
                "text_embedder": {"text": "Hello"},
                "llm": {"messages": messages}
            },
            include_outputs_from=["llm"]
        )
        logging.info("LLM initialized and warmed up")

        # Clear GPU memory after initialization
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logging.info("Cleared GPU memory after initialization")

        logging.info("Model initialization completed successfully")
        MODEL_INITIALIZED = True
        return True

    except Exception as e:
        logging.error(f"Error during model initialization: {e}")
        return False

#Initialize Elasticsearch Document Store
document_store = ElasticsearchDocumentStore(
    hosts="https://localhost:9200/",
    basic_auth=ElasticAuth,
    ca_certs="/home/LLMBot.Haystack/elastic/elasticsearch-8.16.1/config/certs/http_ca.crt",
    index="document",
    verify_certs=False
)

document_embedder = SentenceTransformersDocumentEmbedder(model=EmbedderModel, progress_bar=True, batch_size=32,prefix="passage:")
document_embedder.warm_up()

HuggingFaceLocalChatGenerator = HuggingFaceLocalChatGenerator(
    model=local_model_path,
    task="text2text-generation",
    generation_kwargs={
        "max_new_tokens": 500,
        "do_sample": True,
        "temperature": 0.7,
        "repetition_penalty": 1.2,
        "top_k": 70,
        "top_p": 0.9,
        "batch_size": 8
    },
    huggingface_pipeline_kwargs={
        "device_map": "auto",
        "torch_dtype": torch.float16,
    }
)

@component
class ListJoiner:
    def __init__(self, _type: Any):
        component.set_output_types(self, values=_type)

    def run(self, values: Variadic[Any]):
        result = list(chain(*values))
        return {"values": result}


pipeline = Pipeline()
pipeline.add_component("text_embedder",
                       SentenceTransformersTextEmbedder(model=EmbedderModel, progress_bar=True, prefix="query:"))
pipeline.add_component("retriever", ElasticsearchEmbeddingRetriever(document_store=document_store))
pipeline.add_component("llm", HuggingFaceLocalChatGenerator)
# Connect pipeline components
pipeline.connect("text_embedder.embedding", "retriever.query_embedding")

def delete_chat_history_index(user_id=None):
    """
    Deletes chat history index for a specific user or all users.
    """
    global ElasticAuth
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Accept": "application/json; charset=utf-8"
    }
    try:
        if user_id:
            # Delete specific user's chat history
            index_name = f"chat_history_{user_id}"
            url = f"https://localhost:9200/{index_name}"
            response = requests.delete(url, auth=ElasticAuth, headers=headers, verify=False)
        else:
            # First get all indices matching the pattern
            url = f"https://localhost:9200/_cat/indices/chat_history_*?format=json"
            response = requests.get(url, auth=ElasticAuth, headers=headers, verify=False)
            indices = response.json()

            # Delete each index individually
            for index in indices:
                index_name = index['index']
                delete_url = f"https://localhost:9200/{index_name}"
                delete_response = requests.delete(delete_url, auth=ElasticAuth, headers=headers, verify=False)
                if delete_response.status_code == 200:
                    print(f"Deleted index: {index_name}")

    except requests.exceptions.RequestException as e:
        print("Error deleting chat history:", e)


def fetch_chat_history(user_id):
    """
    Fetch the latest question and answer from Elasticsearch using user_id as index.
    """
    index_name = f"chat_history_{user_id}"
    url = f"https://localhost:9200/{index_name}/_doc/history"
    global ElasticAuth
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Accept": "application/json; charset=utf-8"
    }

    try:
        response = requests.get(url, auth=ElasticAuth, verify=False)

        # Check if the index exists
        if response.status_code == 404:
            logging.info(f"Chat history not found for user_id: {user_id}")
            return {"response": "No previous chat history found"}

        if response.status_code == 200:
            data = response.json()
            if "_source" in data and "last_interaction" in data["_source"]:
                return data["_source"]["last_interaction"]

        # If we get here, something went wrong with the data structure
        return {"response": "Invalid chat history format"}

    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching chat history: {e}")
        return {"response": "Error accessing chat history"}

def fetch_data_from_api():
    try:
        params = {
            "token": "nNjiIyuHJDAKHi76jGZwQmqCqMsk9ashjhSd8s5xKLcxF"
        }
        response = requests.get("https://172.16.40.170:44305/api/Document/GetContents",
                                params=params, verify=False)
        response.raise_for_status()

        data = response.json()
        if data.get("status") and data.get("content"):
            documents = []
            for item in data["content"]:
                content = item.get("content", "")
                name = item.get("name", "")
                if content:
                    try:
                        content = content.encode('utf-8').decode('utf-8', errors='ignore')
                        doc = Document(content=content, meta={"name": name})
                        documents.append(doc)
                    except UnicodeDecodeError:
                        logging.warning(f"Failed to decode content for file {name}")

            # Embed documents before storing
            documents_with_embeddings = document_embedder.run(documents)
            return documents_with_embeddings.get("documents")
        return []
    except Exception as e:
        logging.error(f"Error fetching data: {e}")
        return []

def fetch_documents_from_elasticsearch(query):
    """
    Fetch relevant documents from Elasticsearch document store.
    """
    index_name = "document"
    url = f"https://localhost:9200/{index_name}/_search"
    global ElasticAuth
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Accept": "application/json; charset=utf-8"
    }

    # Search query matching the exact document structure
    search_body = {
        "query": {
            "match": {
                "content": {
                    "query": query,
                    "fuzziness": "AUTO"
                }
            }
        },
        "_source": ["id", "content"],  # Only fetch needed fields
        "size": 10000  # Limit the number of documents
    }

    try:
        response = requests.post(url, auth=ElasticAuth, headers=headers, json=search_body, verify=False)
        response.raise_for_status()
        data = response.json()

        documents = []
        if "hits" in data and "hits" in data["hits"]:
            for hit in data["hits"]["hits"]:
                if "_source" in hit and "content" in hit["_source"]:
                    doc_content = hit["_source"]["content"]
                    doc_id = hit["_source"]["id"]
                    documents.append({
                        "id": doc_id,
                        "content": doc_content
                    })
        return documents
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching documents: {e}")
        return []


def save_chat_history(user_id, user_message, assistant_response):
    """
    Save chat history using user_id as index name.
    """
    index_name = f"chat_history_{user_id}"
    url = f"https://localhost:9200/{index_name}/_doc/history"
    global ElasticAuth
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Accept": "application/json; charset=utf-8"
    }

    # Fetch the existing interaction
    existing_interaction = fetch_chat_history(user_id)

    if existing_interaction and 'history' in existing_interaction:
        updated_history = (
            f"{existing_interaction['history']}\n"
            f"{user_message}\n"
            f"{assistant_response}"
        )
    else:
        updated_history = f"{user_message}\n{assistant_response}"

    updated_document = {
        "last_interaction": {
            "history": updated_history
        }
    }

    try:
        response = requests.put(url, auth=ElasticAuth, headers=headers, json=updated_document, verify=False)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print("Error saving chat history:", e)


def is_mostly_persian(text):
    # Define the range of Persian Unicode characters
    persian_chars = set(range(0x0600, 0x06FF))  # Persian/Arabic Unicode block
    persian_chars.update(range(0xFB50, 0xFDFF))  # Additional Persian characters
    persian_chars.update(range(0xFE70, 0xFEFF))  # More Persian characters

    # Split the text into words and take the first 10 words
    words = text.split()[:5]
    first_10_words_text = ' '.join(words)

    # Count the number of Persian characters in the first 10 words
    persian_count = sum(1 for char in first_10_words_text if ord(char) in persian_chars)

    # Calculate the percentage of Persian characters in the first 10 words
    total_chars = len(first_10_words_text)
    if total_chars == 0:
        return False  # If the text is empty, return False

    persian_percentage = (persian_count / total_chars) * 100

    # Return True if more than 30% of the first 10 words are Persian
    return persian_percentage > 20

# Modified process_question function to use the fetched documents
def process_question(question, user_id):
    try:
        # Step 1: Fetch documents from the API and save them to the document store
        new_docs = fetch_data_from_api()
        if not new_docs:
            new_docs = [Document(content="DDSS means Dana Data Security Suit")]
        document_store.write_documents(documents=new_docs, policy=DuplicatePolicy.OVERWRITE)

        # Step 2: Fetch the latest interaction from Elasticsearch chat history
        last_interaction = fetch_chat_history(user_id)

        # Step 2.5: Fetch relevant documents from Elasticsearch
        relevant_docs = fetch_documents_from_elasticsearch(question)
        documents_context = "\n".join([f"Document {i + 1}: {doc['content']}" for i, doc in enumerate(relevant_docs)])

        if is_mostly_persian(question):
            language_instruction = "Please respond in Persian (فارسی) but keep english technical words in english"
        else:
            language_instruction = "Please respond in English."

        # Step 3: Construct the prompt for the LLM
        if last_interaction and 'history' in last_interaction:
            prompt = (
               # "Just respond in the same language as the user's input language like persian language."
                f"Relevant documents:\n{documents_context}\n\n"
                f"Conversation history:\n{last_interaction['history']}\n\n"
                f"Question: {question}"
                f"{language_instruction}"
            )
        else:
            prompt = (
                f"Relevant documents:\n{documents_context}\n\n"
                f"Question: {question}"
                f"{language_instruction}"
            )

        if is_mostly_persian(question):
            basic_prompt = """You are a helpful AI assistant using provided supporting documents and conversation history and your information to assist humans.
            Always respond in Persian (فارسی) but keep english technical words in english like mongodb"""
        else:
            basic_prompt="You are a helpful AI assistant using provided supporting documents and conversation history and your information to assist humans."
        system_message = ChatMessage.from_system(basic_prompt
            # "You are a helpful AI assistant using provided supporting documents and conversation history and your information to assist humans."
        )

        # Rest of the function remains the same...
        messages = [system_message, ChatMessage.from_user(prompt)]

        res = pipeline.run(
            data={
                "text_embedder": {"text": question},
                "llm": {"messages": messages}  # Direct connection to LLM
            },
            include_outputs_from=["llm"]
        )

        response_text = res['llm']['replies'][0].content.strip()
        answer=f"You as ai last answer:\n{extract_answer(response_text)}"
        user_message=f"User last message:\n{question}"
        save_chat_history(user_id=user_id, user_message=user_message, assistant_response=answer)
        return response_text

    except KeyError as e:
        logging.error(f"Error processing question: Missing key {e}")
        return f"Error: Missing key {e}"
    except Exception as e:
        logging.error(f"Error processing question: {e}")
        return f"Error: {str(e)}"


def extract_answer(output):
    # Find the last "Question: " in the output
    question_index = output.rfind("Question: ")

    # If no "Question: " is found, return an empty string
    if question_index == -1:
        return ""

    # Find the first "assistant" after the last "Question: "
    assistant_index = output.find("assistant", question_index)

    # If no "assistant" is found, return an empty string
    if assistant_index == -1:
        return ""

    # Find the next "assistant" after the first one
    next_assistant_index = output.find("assistant", assistant_index + 1)

    # If no next "assistant" is found, set the end index to the end of the string
    if next_assistant_index == -1:
        end_index = len(output)
    else:
        end_index = next_assistant_index

    # Extract the answer text
    answer = output[assistant_index + len("assistant"):end_index].strip()

    return answer
def clear_chat_history_index(user_id):
    """
    Clears chat history for specific user.
    """
    try:
        index_name = f"chat_history_{user_id}"
        url = f"https://localhost:9200/{index_name}/_delete_by_query"
        global ElasticAuth
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "application/json; charset=utf-8"
        }

        delete_query = {
            "query": {
                "match_all": {}
            }
        }

        response = requests.post(url, auth=ElasticAuth, headers=headers, json=delete_query, verify=False)
        response.raise_for_status()

        if response.status_code == 200:
            logging.info(f"Chat history cleared for user {user_id}")
        else:
            logging.warning(f"Failed to clear chat history for user {user_id}")

    except requests.exceptions.RequestException as e:
        logging.error(f"Error clearing chat history: {e}")

app = Flask(__name__)

# Create a request queue
request_queue = queue.Queue()

def process_queue():
    while True:
        try:
            # Get request from queue
            request_data = request_queue.get()
            if request_data is None:  # Poison pill to stop the thread
                break

            question = request_data['prompt']
            user_id = request_data.get('user_id', 'default_user')

            # Process the question using existing logic
            response = process_question(question, user_id)
            answer = extract_answer(response)

            request_data['result'] = {
                "response": answer,
                "user_id": user_id
            }

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logging.info("Cleared GPU memory.")

        except Exception as e:
            logging.error(f"Error processing request: {e}")
        finally:
            request_queue.task_done()


# Start worker thread
worker_thread = threading.Thread(target=process_queue, daemon=True)
worker_thread.start()


@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()

        if not data or 'prompt' not in data:
            return jsonify({"error": "Missing 'prompt' in request body"}), 400

        question = data['prompt']
        user_id = data.get('user_id', 'default_user')

        # Handle exit commands
        if question.lower() in ['quit', 'exit', 'q']:
            # Put poison pill in queue and wait for worker to finish
            request_queue.put(None)
            worker_thread.join()

            # Send response and wait before shutting down
            response = jsonify({"response": "Server shutting down..."})
            response.headers.add('Connection', 'close')  # Add this header
            time.sleep(1)  # Wait for 1 second to ensure response is sent

            def shutdown():
                os._exit(0)

            threading.Timer(0.1, shutdown).start()  # Schedule shutdown after response
            return response

        # Handle special commands without queueing
        if question.lower() in ["delete all"]:
            delete_chat_history_index(user_id=None)
            return jsonify({"response": "All chat indexes deleted."}), 200

        if question.lower() in ["clear"]:
            clear_chat_history_index(user_id=user_id)
            return jsonify({"response": f"Chat history of id: {user_id} cleared."}), 200

        if question.lower() in ["delete"]:
            delete_chat_history_index(user_id=user_id)
            return jsonify({"response": f"Chat index of id: {user_id} cleared."}), 200

        if question.lower() in ["view"]:
            chat_history = fetch_chat_history(user_id)
            return jsonify(chat_history), 200

        # Add request to queue and process
        request_data = {
            'prompt': question,
            'user_id': user_id,
            'result': None
        }
        request_queue.put(request_data)
        request_queue.join()  # Wait for processing to complete

        return jsonify(request_data['result']), 200

    except Exception as e:
        logging.error(f"Error in /chat endpoint: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    if initialize_model():
        logging.info("Starting chat session...")
        delete_chat_history_index()
        app.run(debug=True, host='0.0.0.0', port=8503, use_reloader=False)
    else:
        logging.error("Failed to initialize model. Exiting...")
        exit(1)