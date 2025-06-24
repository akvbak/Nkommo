import os
import api
os.environ["OPENAI_API_KEY"] = api.APIKEY

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from uuid import uuid4

# configuration
datapath = r"Nkommo/files"
chromapath = r"Nkommo/chroma_db"

# initiate the embeddings model
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

# initiate the vector store
vector_store = Chroma(
    collection_name="Nkommov1",
    embedding_function=embeddings_model,
    persist_directory=chromapath,
)

# loading the PDF document
loader = PyPDFDirectoryLoader(datapath)

raw_documents = loader.load()

# splitting the document
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)

# creating the chunks
chunks = text_splitter.split_documents(raw_documents)

# creating unique ID's
uuids = [str(uuid4()) for _ in range(len(chunks))]

print(embeddings_model.embed_query("Bra y3n b) Nkommo"))


# adding chunks to vector store
vector_store.add_documents(documents=chunks, ids=uuids)