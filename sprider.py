import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import time
import random

# Connect to Milvus
connections.connect("default", host="127.0.0.1", port="19530")

# Define the schema for the collection with dynamic fields enabled
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),  # Auto-generated IDs
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),  # Embedding size of 1024 dimensions
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=10000)  # Field to store the text
]
schema = CollectionSchema(fields, description="Embeddings and Text from Crawled Web Pages", enable_dynamic_field=True)

# Collection name
collection_name = "song_of_ice_and_fire"

# Check if the collection exists, and if not, create it
if not utility.has_collection(collection_name):
    collection = Collection(name=collection_name, schema=schema)
    print(f"Created collection '{collection_name}' with dynamic fields enabled.")
else:
    collection = Collection(name=collection_name)
    print(f"Loaded existing collection '{collection_name}'.")

# Function to embed text using the API
def embed_text(text):
    url = "http://127.0.0.1:11434/api/embeddings"
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
        else:
            print(f"Error: Embedding size mismatch. Expected 1024, got {len(embedding)}.")
            return None
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None

# Function to fetch page content with retry and delay to avoid 403 errors
def fetch_page_content(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    retries = 3
    for i in range(retries):
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.content
        except requests.exceptions.RequestException as e:
            print(f"Error fetching {url}: {e}")
            if response.status_code == 403:
                print(f"Received 403 Forbidden for {url}. Retrying in {i+1} seconds...")
                time.sleep(i + 1)
            else:
                return None
    return None

# Function to parse and extract URLs from a page
def find_urls(content, current_url):
    soup = BeautifulSoup(content, 'html.parser')
    found_urls = []
    for link in soup.find_all('a', href=True):
        href = link['href']
        if href.startswith('/index.php'):
            full_url = urljoin(current_url, href)
            if full_url not in visited:
                found_urls.append(full_url)
    return found_urls

# Function to store crawled text and embeddings in Milvus
def store_crawled_text_in_milvus(text, url):
    clean_text = text.strip().lower()
    embeddings = embed_text(clean_text)

    if embeddings:
        try:
            collection.insert([
                {
                    "embedding": embeddings,
                    "text": clean_text
                }
            ])
            print(f"Stored text and embeddings from {url}")
        except Exception as e:
            print(f"Error inserting data from {url}: {e}")
    else:
        print(f"Failed to get embeddings for {url}")

# Function to crawl a single page with delay between requests
def crawl_page(url):
    print(f"Visiting {url}")
    content = fetch_page_content(url)
    if content:
        visited.add(url)
        soup = BeautifulSoup(content, 'html.parser')
        page_content = soup.get_text(separator=' ', strip=True)

        if len(page_content) > 500:
            store_crawled_text_in_milvus(page_content[:10000], url)  # Store first 10,000 characters
        else:
            print(f"Skipping URL due to insufficient content: {url}")

        time.sleep(random.uniform(1, 3))  # Random delay to avoid rate limiting
        return find_urls(content, url)
    return []

# Main crawling function
def crawl_wiki(start_urls):
    global visited, to_visit
    visited = set()
    to_visit = set(start_urls)

    with ThreadPoolExecutor(max_workers=5) as executor:
        while to_visit:
            futures = {executor.submit(crawl_page, url): url for url in list(to_visit)}
            to_visit.clear()

            for future in as_completed(futures):
                new_urls = future.result()
                if new_urls:
                    to_visit.update(new_urls)

        print("Crawling completed.")

# List of starting URLs
start_urls = [
    "https://awoiaf.westeros.org/index.php/Chapters",
    "https://awoiaf.westeros.org/index.php/Portal:Characters",
    "https://awoiaf.westeros.org/index.php/Houses_of_Westeros",
    "https://awoiaf.westeros.org/index.php/Timeline_of_major_events",
    "https://awoiaf.westeros.org/index.php/Portal:Geography",
    "https://awoiaf.westeros.org/index.php/Portal:Culture",
    "https://awoiaf.westeros.org/index.php/Portal:TV_Show"
]

# Start crawling
crawl_wiki(start_urls)

# Create index after inserting data
print("Creating index on 'embedding' field...")
index_params = {
    "metric_type": "L2",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 128}
}
collection.create_index(field_name="embedding", index_params=index_params)
collection.load()

print("Index creation and loading completed.")
