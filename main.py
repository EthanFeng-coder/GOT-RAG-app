import os
import requests
from PyPDF2 import PdfReader
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from tqdm import tqdm  # Progress bar

# Connect to Milvus
connections.connect("default", host="127.0.0.1", port="19530")

# Define the schema for the collection with dynamic fields enabled
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),  # Auto-generated IDs
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),  # Embedding size of 1024 dimensions
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=10000)  # Field to store the text
]
schema = CollectionSchema(fields, description="Embeddings and Text of Song of Ice and Fire PDFs", enable_dynamic_field=True)

# Check if the collection exists and drop it if it does
collection_name = "song_of_ice_and_fire"
if utility.has_collection(collection_name):
    utility.drop_collection(collection_name)
    print(f"Dropped existing collection '{collection_name}'.")

# Recreate the collection with the correct schema and dynamic fields enabled
collection = Collection(name=collection_name, schema=schema)
print(f"Created collection '{collection_name}' with dynamic fields enabled.")

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

# Function to process one page at a time and store embeddings and text in Milvus
def process_pdf_and_store_embeddings(pdf_path):
    # Extract text page-by-page from PDF
    reader = PdfReader(pdf_path)
    num_pages = len(reader.pages)

    # Progress bar to track page processing
    with tqdm(total=num_pages, desc=f"Processing {pdf_path}") as pbar:
        for page_num in range(num_pages):
            page = reader.pages[page_num]
            text = page.extract_text()

            if text and text.strip():  # Check if text is non-empty
                # Get embeddings for the extracted page text
                embeddings = embed_text(text)

                if embeddings:
                    # Insert both the embedding and the text into Milvus
                    try:
                        print(f"Inserting text and embeddings for page {page_num + 1}...")
                        # Insert both embedding and text into Milvus
                        collection.insert([
                            {
                                "embedding": embeddings,  # Embedding data
                                "text": text.strip().lower()  # Correct text field name
                            }
                        ])
                        print(f"Embeddings and text for page {page_num + 1} stored in Milvus.")
                    except Exception as e:
                        print(f"Error inserting embedding for page {page_num + 1}: {e}")
                else:
                    print(f"Failed to get embeddings for page {page_num + 1}")
            else:
                print(f"Empty or invalid text on page {page_num + 1}")

            pbar.update(1)  # Update progress bar for each page processed

# Directory containing the PDFs
pdf_directory = "pdfFolder"  # Update this with the actual directory

# Get the list of PDF files in the directory
pdf_files = [filename for filename in os.listdir(pdf_directory) if filename.endswith(".pdf")]

# Process all PDFs in the directory
for filename in pdf_files:
    pdf_path = os.path.join(pdf_directory, filename)
    process_pdf_and_store_embeddings(pdf_path)
    print(f"Processed: {filename}")

# Create index after inserting the data
print("Creating index on 'embedding' field for collection...")
index_params = {
    "metric_type": "L2",  # You can also use "IP" for inner product
    "index_type": "IVF_FLAT",  # Index type, such as IVF_FLAT, IVF_SQ8, etc.
    "params": {"nlist": 128}  # Number of clusters
}

# Create index on the 'embedding' field
collection.create_index(field_name="embedding", index_params=index_params)

# Load the collection to apply the index
collection.load()

print("Index created and applied successfully.")
