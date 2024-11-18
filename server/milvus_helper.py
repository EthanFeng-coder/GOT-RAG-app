from pymilvus import connections, Collection
import requests

# Connect to Milvus
connections.connect("default", host="milvus-standalone", port="19530")
collection_name = "song_of_ice_and_fire"
collection = Collection(name=collection_name)
collection.load()
hosturl = "192.168.8.143"
def embed_text(text):
    url = f"http://{hosturl}:11434/api/embeddings"
    payload = {
        "model": "mxbai-embed-large:latest",
        "prompt": text
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        embedding = response.json().get('embedding')
        if isinstance(embedding, list) and len(embedding) == 1024:
            return embedding
    print(f"Error: {response.status_code}, {response.text}")
    return None

def query_by_embedding(user_query, limit=5):
    query_embedding = embed_text(user_query)
    if not query_embedding:
        return None
    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10}
    }
    search_results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=limit,
        output_fields=["text"]
    )
    if search_results:
        context = "\n".join([result.entity.get('text') for result in search_results[0]])
        return context
    return None
