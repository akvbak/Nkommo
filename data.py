import os
import api
from uuid import uuid4
from hashlib import sha256
import json

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma

# Configuration
os.environ["OPENAI_API_KEY"] = api.APIKEY
DATAPATH = r"Nkommo/files"
CHROMAPATH = r"Nkommo/chroma_db"
HASH_FILE = "Nkommo/pdf_hashes.json"

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = Chroma(
    collection_name="Nkommov1",
    embedding_function=embeddings_model,
    persist_directory=CHROMAPATH,
)

# Load PDF hash cache
if os.path.exists(HASH_FILE):
    with open(HASH_FILE, "r") as f:
        hash_cache = json.load(f)
else:
    hash_cache = {}

loader = PyPDFDirectoryLoader(DATAPATH)
raw_documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False,
)

new_chunks = []
new_ids = []

for doc in raw_documents:
    content = doc.page_content.encode("utf-8")
    file_hash = sha256(content).hexdigest()
    
    if file_hash not in hash_cache:
        chunks = text_splitter.split_documents([doc])
        uuids = [str(uuid4()) for _ in range(len(chunks))]

        new_chunks.extend(chunks)
        new_ids.extend(uuids)

        hash_cache[file_hash] = uuids

if new_chunks:
    vector_store.add_documents(documents=new_chunks, ids=new_ids)
    vector_store.persist()

    with open(HASH_FILE, "w") as f:
        json.dump(hash_cache, f)

print(f"Added {len(new_chunks)} new chunks.")
