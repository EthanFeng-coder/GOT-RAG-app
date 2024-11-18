import requests
import json
from pymilvus import Collection, connections

# Ollama API details
ollama_url = "http://127.0.0.1:11434/api/generate"
query_text = "who is the knight of the seven kingdoms"

# Payload for Ollama API
payload = {
    "model": "llama3.1:latest",  # Specify the model you're using
    "prompt": f"Extract the most relevant name or person from '{query_text}'. Focus only on proper names and people such as 'Dunk the Tall'.",
    "temperature": 0.5
}

# Send POST request to Ollama API to extract the name of the knight
response = requests.post(ollama_url, json=payload)

# Initialize a variable to store the extracted keyword (name)
keyword = None

# Try parsing the response to extract the most relevant entity (name)
try:
    # Split the response text by newline (or adjust based on actual delimiters)
    json_objects = response.text.splitlines()

    # Iterate through the individual JSON objects
    for json_str in json_objects:
        json_data = json.loads(json_str)

        # Extract the 'response' field where the most relevant name is contained
        if "response" in json_data:
            keyword_candidate = json_data["response"].strip()

            # Ensure it's a valid name (ignore single letters like "A" and common words)
            if keyword_candidate and len(keyword_candidate) > 1 and not keyword_candidate.isspace():
                # Filter out non-person names (like "the", "a", etc.)
                if keyword_candidate.lower() not in ['the', 'knight', 'a', 'of', 'kingdoms', 'is', 'who']:
                    keyword = keyword_candidate
                    break  # Exit the loop once we have the relevant name

    # If a keyword was found, proceed to query Milvus
    if keyword:
        print(f"Extracted Keyword: {keyword}")
    else:
        print("No relevant name was extracted.")

except json.JSONDecodeError as e:
    print(f"Failed to parse JSON: {e}")

# Proceed with Milvus query if a valid keyword was extracted
if keyword:
    # Connect to Milvus
    connections.connect("default", host="127.0.0.1", port="19530")

    # Define collection name
    collection_name = "song_of_ice_and_fire"

    # Load the collection
    collection = Collection(name=collection_name)
    collection.load()

    # Function to query records containing the extracted name (case-insensitive)
    def query_by_text(search_text, limit=5):
        try:
            search_text_lower = search_text.lower()  # Convert the search text to lowercase
            print(f"Querying for records containing (case-insensitive) '{search_text_lower}'...")

            # Perform the query assuming the text in the collection is stored in lowercase
            results = collection.query(
                expr=f"text like '%{search_text_lower}%'",  # Search for the lowercased search term
                output_fields=["id", "text"],  # Retrieve 'id' and 'text'
                limit=limit  # Limit the results to specified number of records
            )

            # Print the results
            if results:
                print(f"Records containing (case-insensitive) '{search_text}':")
                for result in results:
                    print(f"ID: {result['id']}, Text: {result['text'][:100]}...")  # Show first 100 characters of the text
            else:
                print(f"No records found containing '{search_text}'.")
        except Exception as e:
            print(f"Error querying by text: {e}")

    # Run the query using the extracted keyword
    query_by_text(keyword, limit=3)
