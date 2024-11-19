import requests
import json
hosturl = "192.168.8.143"
def generate_answer_with_ollama(user_query, context):
    ollama_url = f"http://{hosturl}:11435/api/generate"
    prompt = f"Context: {context}\n\nUser question: {user_query}\n\nAnswer based on the context and try to provide the reference list"
    payload = {
        "model": "llama3.1:latest",
        "prompt": prompt
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(ollama_url, json=payload, headers=headers, stream=True)
    if response.status_code == 200:
        full_response = ""
        for line in response.iter_lines():
            if line:
                try:
                    json_line = json.loads(line.decode('utf-8'))
                    full_response += json_line.get('response', '')
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
        return full_response
    print(f"Error: {response.status_code}, {response.text}")
    return None
