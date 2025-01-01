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

#Initialize Elasticsearch Document Store
document_store = ElasticsearchDocumentStore(
    hosts="https://localhost:9200/",
    basic_auth=("elastic", "B1+Zz5*6CtgThyeTZive"),
    ca_certs="/home/LLMBot.Haystack/elastic/elasticsearch-8.16.1/config/certs/http_ca.crt",
    index="document",
    verify_certs=False
)

chat_history_document_store = ElasticsearchDocumentStore(
    hosts="https://localhost:9200/",
    basic_auth=("elastic", "B1+Zz5*6CtgThyeTZive"),
    ca_certs="/home/LLMBot.Haystack/elastic/elasticsearch-8.16.1/config/certs/http_ca.crt",
    index="chat_history",
    verify_certs=False
)

# Define embedding model
model = "sentence-transformers/all-MiniLM-L12-v2"
document_embedder = SentenceTransformersDocumentEmbedder(model=model, progress_bar=True, batch_size=32,
                                                         prefix="passage:")
document_embedder.warm_up()

# Define embedding model for chat history
chat_history_embedder = SentenceTransformersDocumentEmbedder(model=model, progress_bar=True, batch_size=32,
                                                             prefix="chat_history:")
chat_history_embedder.warm_up()

# Initialize model
local_model_path = "/home/DSTV3.LLMEmbedding.Ai.Api/AVA-Llama-3-V2/"
#local_model_path = "E:/Workarea/ai-chatbots/Haystack.llama.cpp/models/AVA-Llama-3-V2"

HuggingFaceLocalChatGenerator = HuggingFaceLocalChatGenerator(
    model=local_model_path,
    task="text2text-generation",
    generation_kwargs={
        "max_new_tokens": 150,
        "do_sample": True,
        "temperature": 0.7,
        "repetition_penalty": 1.2,
        "top_k": 50,
        "top_p": 0.9,
        "batch_size": 8
    },
    huggingface_pipeline_kwargs={
        "device_map": "auto",
        "torch_dtype": torch.float16,
    }
)

# System message and templates
system_message = ChatMessage.from_system(
    "You are a helpful AI assistant using provided supporting documents and conversation history to assist humans."
    "Note that supporting documents are not part of the conversation. If question can't be answered from supporting documents, say so.")

@component
class ListJoiner:
    def __init__(self, _type: Any):
        component.set_output_types(self, values=_type)

    def run(self, values: Variadic[Any]):
        result = list(chain(*values))
        return {"values": result}


pipeline = Pipeline()
pipeline.add_component("text_embedder",
                       SentenceTransformersTextEmbedder(model=model, progress_bar=True, prefix="query:"))
pipeline.add_component("retriever", ElasticsearchEmbeddingRetriever(document_store=document_store))
pipeline.add_component("chat_history_retriever", ElasticsearchEmbeddingRetriever(document_store=chat_history_document_store))
pipeline.add_component("prompt_builder", ChatPromptBuilder(variables=["query", "documents", "chat_histories"],
                                                           required_variables=["query", "documents", "chat_histories"]))
pipeline.add_component("llm", HuggingFaceLocalChatGenerator)

# Connect pipeline components
pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
pipeline.connect("text_embedder.embedding", "chat_history_retriever.query_embedding")
pipeline.connect("retriever.documents", "prompt_builder.documents")
pipeline.connect("chat_history_retriever.documents", "prompt_builder.chat_histories")
pipeline.connect("prompt_builder.prompt", "llm.messages")

def delete_chat_history_index():
    """
    Deletes the chat_history index from Elasticsearch to ensure it's empty.
    """
    # Elasticsearch index for chat history
    index_name = "chat_history"
    url = f"https://localhost:9200/{index_name}"
    auth = ("elastic", "B1+Zz5*6CtgThyeTZive")
    headers = {"Content-Type": "application/json"}

    try:
        # Send a DELETE request to remove the index
        response = requests.delete(url, auth=auth, headers=headers, verify=False)
        if response.status_code == 200:
            print(f"Index '{index_name}' deleted successfully.")
        elif response.status_code == 404:
            print(f"Index '{index_name}' does not exist (nothing to delete).")
        else:
            print(f"Failed to delete index '{index_name}'. Status code: {response.status_code}")
            print("Response:", response.text)
    except requests.exceptions.RequestException as e:
        print("Error deleting chat history index:", e)

def fetch_chat_history(conversation_id):
    """
    Fetch the latest question and answer from Elasticsearch.
    """
    # Elasticsearch index for chat history
    index_name = "chat_history"
    url = f"https://localhost:9200/{index_name}/_doc/{conversation_id}"
    auth = ("elastic", "B1+Zz5*6CtgThyeTZive")
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.get(url, auth=auth, verify=False)
        response.raise_for_status()
        data = response.json()
        if "_source" in data and "last_interaction" in data["_source"]:
            return data["_source"]["last_interaction"]
        else:
            return None
    except requests.exceptions.RequestException as e:
        print("Error fetching chat history:", e)
        return None

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
    auth = ("elastic", "B1+Zz5*6CtgThyeTZive")
    headers = {"Content-Type": "application/json"}

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
        "size": 3  # Limit the number of documents
    }

    try:
        response = requests.post(url, auth=auth, headers=headers, json=search_body, verify=False)
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


def save_chat_history(conversation_id, user_message, assistant_response):
    """
    Save chat history in chronological order: user1-assistance1-user2-assistance2
    """
    index_name = "chat_history"
    url = f"https://localhost:9200/{index_name}/_doc/{conversation_id}"
    auth = ("elastic", "B1+Zz5*6CtgThyeTZive")
    headers = {"Content-Type": "application/json"}

    # Fetch the existing interaction
    existing_interaction = fetch_chat_history(conversation_id)

    if existing_interaction and 'history' in existing_interaction:
        updated_history = (
            f"{existing_interaction['history']}\n"
            f"Human: {user_message}\n"
            f"Assistant: {assistant_response}"
        )
    else:
        updated_history = f"Human: {user_message}\nAssistant: {assistant_response}"

    # Update document with the complete history
    updated_document = {
        "last_interaction": {
            "history": updated_history
        }
    }

    try:
        response = requests.put(url, auth=auth, headers=headers, json=updated_document, verify=False)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print("Error saving chat history:", e)

# Modified process_question function to use the fetched documents
def process_question(question, conversation_id):
    try:
        # Step 1: Fetch documents from the API and save them to the document store
        new_docs = fetch_data_from_api()
        if not new_docs:
            new_docs = [Document(content="DDSS means Dana Data Security Suit")]
        document_store.write_documents(documents=new_docs, policy=DuplicatePolicy.OVERWRITE)

        # Step 2: Fetch the latest interaction from Elasticsearch chat history
        last_interaction = fetch_chat_history(conversation_id)

        # Step 2.5: Fetch relevant documents from Elasticsearch
        relevant_docs = fetch_documents_from_elasticsearch(question)
        documents_context = "\n".join([f"Document {i + 1}: {doc['content']}" for i, doc in enumerate(relevant_docs)])

        # Step 3: Construct the prompt for the LLM
        if last_interaction and 'history' in last_interaction:
            prompt = (
                f"Relevant documents:\n{documents_context}\n\n"
                f"Conversation history:\n{last_interaction['history']}\n\n"
                f"Question: {question}"
            )
        else:
            prompt = (
                f"Relevant documents:\n{documents_context}\n\n"
                f"Question: {question}"
            )

        # Rest of the function remains the same...
        messages = [system_message, ChatMessage.from_user(prompt)]

        res = pipeline.run(
            data={
                "text_embedder": {"text": question},
                "prompt_builder": {"template": messages, "query": question}
            },
            include_outputs_from=["llm"]
        )

        response_text = res['llm']['replies'][0].content.strip()

        answer = extract_answer(response_text)
        save_chat_history(conversation_id, question, answer)
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
def clear_chat_history_index():
    """
    Clears all data from the chat_history index without deleting the index.
    """
    try:
        # Elasticsearch index for chat history
        index_name = "chat_history"
        url = f"https://localhost:9200/{index_name}/_delete_by_query"
        auth = ("elastic", "B1+Zz5*6CtgThyeTZive")
        headers = {"Content-Type": "application/json"}

        # Delete all documents in the index using a match_all query
        delete_query = {
            "query": {
                "match_all": {}
            }
        }

        # Send the DELETE request
        response = requests.post(url, auth=auth, headers=headers, json=delete_query, verify=False)
        response.raise_for_status()

        if response.status_code == 200:
            logging.info(f"All documents in index '{index_name}' cleared successfully.")
        else:
            logging.warning(f"Failed to clear documents in index '{index_name}'. Status code: {response.status_code}")
            logging.warning("Response:", response.text)

    except requests.exceptions.RequestException as e:
        logging.error(f"Error clearing chat history index: {e}")

def main():
    conversation_id = "unique_conversation_id"  # Replace with a unique ID for each conversation
    while True:
        try:
            # Get user input
            question = input("\nYou: ").strip()

            # Check for exit command
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            # Skip empty inputs
            if not question:
                continue
            if question.lower() in ["clear history"]:
                # Clear chat history index
                clear_chat_history_index()
                continue

            # Process the question and get response
            response = process_question(question, conversation_id)
            answer = extract_answer(response)

            print("\n" + answer)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logging.info("Cleared GPU memory.")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    delete_chat_history_index()
    main()

