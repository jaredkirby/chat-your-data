from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PDFPlumberLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_transformers import DoctranQATransformer
from langchain.schema import Document
from langchain.vectorstores import LanceDB

import lancedb
import json
import pickle

from dotenv import load_dotenv

load_dotenv()

# Load PDF data
print("Loading data...")
loader = PDFPlumberLoader("KPM Offsite 2021 PDF .pdf")
raw_documents = loader.load()
print(raw_documents)

# Doctran QA
# print("Running Doctran QA...")
# qa_transformer = DoctranQATransformer(openai_api_model="gpt-3.5-turbo")
# transformed_document = qa_transformer.transform_documents(documents)
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
db = lancedb.connect("/tmp/lancedb")
table = db.create_table(
    "my_table",
    data=[
        {
            "vector": embeddings.embed_query("Hello World"),
            "text": "Hello World",
            "id": "1",
        }
    ],
    mode="overwrite",
)
vectorstore = LanceDB.from_documents(documents, embeddings, connection=table)
with open("vectorstore-lance.pkl", "wb") as f:
    pickle.dump(vectorstore, f)
