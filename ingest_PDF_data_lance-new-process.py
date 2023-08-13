from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import LanceDB
from langchain.schema import Document

from dotenv import load_dotenv
import pytesseract
from PIL import Image
from io import BytesIO
import pypdfium2 as pdfium
import multiprocessing


import lancedb
import json
import pickle

from dotenv import load_dotenv

load_dotenv()

file_path = "KPM Offsite 2021 PDF .pdf"


def convert_pdf_to_images(path, scale=300 / 72):
    print("Converting PDF to images...")
    pdf_file = pdfium.PdfDocument(path)

    page_indices = [i for i in range(len(pdf_file))]

    renderer = pdf_file.render(
        pdfium.PdfBitmap.to_pil,
        page_indices=page_indices,
        scale=scale,
    )

    final_images = []

    for i, image in zip(page_indices, renderer):
        image_byte_array = BytesIO()
        image.save(image_byte_array, format="jpeg", optimize=True)
        image_byte_array = image_byte_array.getvalue()
        final_images.append(dict({i: image_byte_array}))

    return final_images


# 2. Extract text from images via pytesseract


def extract_text_from_img(list_dict_final_images):
    print("Extracting text from images...")
    image_list = [list(data.values())[0] for data in list_dict_final_images]
    image_content = []

    for index, image_bytes in enumerate(image_list):
        image = Image.open(BytesIO(image_bytes))
        raw_text = str(pytesseract.image_to_string(image))
        image_content.append(raw_text)

    return "\n".join(image_content)


def extract_content_from_pdf(file_path: str):
    images_list = convert_pdf_to_images(file_path)
    text_from_pdf = extract_text_from_img(images_list)
    print(text_from_pdf)

    return text_from_pdf


# Doctran QA
# print("Running Doctran QA...")
# qa_transformer = DoctranQATransformer(openai_api_model="gpt-3.5-turbo")
# transformed_document = qa_transformer.transform_documents(documents)
# print(json.dumps(transformed_document[0].metadata, indent=2))


# Split text
def split_text(text_from_pdf):
    print("Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    documents = text_splitter.split_text(text_from_pdf)
    split_documents = [Document(page_content=doc) for doc in documents]

    return split_documents


# Create vectorstore
def create_vectorstore(split_documents):
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
    vectorstore = LanceDB.from_documents(split_documents, embeddings, connection=table)
    with open("vectorstore-lance.pkl", "wb") as f:
        pickle.dump(vectorstore, f)


def main():
    text_from_pdf = extract_content_from_pdf(file_path)
    split_documents = split_text(text_from_pdf)
    create_vectorstore(split_documents)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
