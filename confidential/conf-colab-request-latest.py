# Updated Request Application (confidential_request_app.py)
import requests
import json
import os

NGROK_URL = "http://d2ff-34-124-233-193.ngrok-free.app"


def send_bulk_embedding_request(base_url, folder_path):
    api_url = f"{base_url}/bulk-embedding"
    headers = {'Content-Type': 'application/json'}

    texts = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith(".txt") and os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                texts.append(f.read())

    payload = json.dumps({"texts": texts})
    response = requests.post(api_url, headers=headers, data=payload)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to get a response. Status code: {response.status_code}, Response: {response.text}")
        return {"error": f"Failed to get a response. Status code: {response.status_code}"}


def send_classification_request(base_url, file_path):
    api_url = f"{base_url}/classify-text"
    headers = {'Content-Type': 'application/json'}
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    payload = json.dumps({"text": text})

    response = requests.post(api_url, headers=headers, data=payload)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to get a response. Status code: {response.status_code}, Response: {response.text}")
        return {"error": f"Failed to get a response. Status code: {response.status_code}"}


if __name__ == "__main__":
    base_url = NGROK_URL
    folder_path = "/content/drive/MyDrive/confidential_project/conf_files"
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        print("The folder path does not exist.")
    else:
        # Additional check to list the contents of the folder to ensure it is accessible
        print(f"Checking contents of folder: {folder_path}")
        print(os.listdir(folder_path))
        response_data = send_bulk_embedding_request(base_url, folder_path)
        print(response_data)

    new_file_path = "/content/drive/MyDrive/confidential_project/not_conf_files/not_conf1.txt"
    if not os.path.exists(new_file_path):
        print("The file path does not exist.")
    else:
        # Additional check to ensure the file is accessible
        print(f"Checking file: {new_file_path}")
        with open(new_file_path, 'r', encoding='utf-8') as f:
            print(f"File contents: {f.read()[:100]}...")  # Print first 100 characters as a check
        response_data = send_classification_request(base_url, new_file_path)
        print(response_data)