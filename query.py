import requests
from pymilvus import connections, Collection

# Connect to Milvus
connections.connect("default", host="127.0.0.1", port="19530")

# Define the collection name
collection_name = "song_of_ice_and_fire"

# Load the collection
collection = Collection(name=collection_name)

# Ensure the collection is loaded into memory before querying
collection.load()

# Function to embed text using the manual API request
def embed_text(text):
    url = "http://127.0.0.1:11434/api/embeddings"  # Correct endpoint
    payload = {
        "model": "mxbai-embed-large:latest",  # Ensure correct model name
        "prompt": text  # Use 'prompt' for the input text
    }
    headers = {"Content-Type": "application/json"}

    # Send the POST request to the /api/embeddings endpoint
    response = requests.post(url, json=payload, headers=headers)

    # Handle the response
    if response.status_code == 200:
        embedding = response.json().get('embedding')

        # Check if the embedding is a list and has the correct dimensions
        if isinstance(embedding, list) and len(embedding) == 1024:  # Ensure embedding is of correct dimension
            return embedding
        else:
            print(f"Error: Received embedding is not of size 1024. Got {len(embedding)} dimensions.")
            return None
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None

# Function to query Milvus using an embedding and return similar texts
def query_by_embedding(user_query, limit=5):
    # Generate embedding for the user's query
    query_embedding = embed_text(user_query)

    if not query_embedding:
        print("Failed to generate an embedding for the query.")
        return None

    # Perform the search in Milvus using the embedding
    search_params = {
        "metric_type": "L2",  # Use L2 (Euclidean) distance metric, you can also use "IP" for inner product
        "params": {"nprobe": 10}  # Adjust nprobe based on your index configuration
    }

    # Execute the search
    search_results = collection.search(
        data=[query_embedding],  # The query embedding
        anns_field="embedding",  # The field where embeddings are stored
        param=search_params,  # The search parameters
        limit=limit,  # Number of results to return
        output_fields=["text"]  # Retrieve the 'text' field along with the results
    )

    if search_results:
        # Concatenate all the found text for use as context
        context = "\n".join([result.entity.get('text') for result in search_results[0]])
        return context
    else:
        print("No results found.")
        return None

# Function to generate answer using Ollama
# Function to generate answer using Ollama
import json
import requests

# Function to generate answer using Ollama with the "llama3.1:latest" model
def generate_answer_with_ollama(user_query, context):
    # Define the Ollama API endpoint
    #print(context)
    ollama_url = "http://127.0.0.1:11434/api/generate"  # Correct Ollama endpoint

    # Create the payload including the context
    prompt = f"Context: {context}\n\nUser question: {user_query}\n\nAnswer based on the context: and try to combine and explain more try to provide the reference list if from chapter or a website link"

    payload = {
        "model": "llama3.1:latest",  # Use llama3.1:latest model here
        "prompt": prompt
    }

    headers = {"Content-Type": "application/json"}

    # Send the POST request to Ollama API and stream the response
    response = requests.post(ollama_url, json=payload, headers=headers, stream=True)

    # Handle the response
    if response.status_code == 200:
        full_response = ""

        # Process the response line by line
        for line in response.iter_lines():
            if line:
                # Parse each line as a separate JSON object
                try:
                    json_line = json.loads(line.decode('utf-8'))
                    full_response += json_line.get('response', '')  # Extract the 'response' part
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
                    continue

        # Return the full combined response
        return full_response
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None

# Main function to combine Milvus and Ollama
def query_and_generate_answer(user_query, limit=5):
    # Step 1: Query Milvus using the embedding to get relevant context
    context = query_by_embedding(user_query, limit=limit)

    if context:
        # Step 2: Feed the context into Ollama to generate an answer
        answer = generate_answer_with_ollama(user_query, context)
        print(f"Answer: {answer}")
    else:
        print("No relevant context found to generate an answer.")

# Example user query
user_query = ("explain the battle of the Ninepenny Kings")
query_and_generate_answer(user_query, limit=3)
