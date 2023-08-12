from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders.chatgpt import ChatGPTLoader
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.document_transformers import DoctranQATransformer

import json
import pickle

from dotenv import load_dotenv

load_dotenv()

# Load ChatGPT data
print("Loading data...")
loader = ChatGPTLoader(
    log_file="conversations.json", num_logs=1
)  # TODO: Change to real data
raw_documents = loader.load()

# Doctran QA
# qa_transformer = DoctranQATransformer()
# transformed_document = qa_transformer.transform_documents(raw_documents)
# print(json.dumps(transformed_document[0].metadata, indent=2))

# Split text
print("Splitting text...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
documents = text_splitter.split_documents(raw_documents)

# Create vectorstore
print("Creating vectorstore...")
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)
with open("vectorstore.pkl", "wb") as f:
    pickle.dump(vectorstore, f)
