from pymilvus import connections, Collection, utility

# Connect to Milvus
connections.connect("default", host="127.0.0.1", port="19530")

# Define collection name
collection_name = "song_of_ice_and_fire"

# Check if the collection exists
if not utility.has_collection(collection_name):
    print(f"Collection '{collection_name}' does not exist.")
    exit()

# Load the existing collection
collection = Collection(name=collection_name)

# Create an index on the embedding field
index_params = {
    "index_type": "IVF_FLAT",  # Example index type, could also be 'HNSW' or others
    "metric_type": "L2",       # Metric type for distance, 'L2' for Euclidean distance
    "params": {"nlist": 128}   # Index-specific parameter
}

# Create the index
collection.create_index(field_name="embedding", index_params=index_params)

# Now load the collection after creating the index
collection.load()
print(f"Collection '{collection_name}' has been loaded with an index.")
