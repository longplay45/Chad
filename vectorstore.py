# vectorstore.py

import os
from datetime import datetime

from langchain.document_loaders import PyMuPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

from config import get_logfile_path

DEFAULT_STORAGE_PATH = "vectorstore"
DEFAULT_EMBEDDING_MODEL_PATH = "all-MiniLM-L6-v2"
VECTORSTORE_INDEX_FILE = os.path.join(DEFAULT_STORAGE_PATH, "index.pkl")
LOG_DIR = DEFAULT_STORAGE_PATH
LOG_NAME = "pdf.log"
LOG_PATH = os.path.join(LOG_DIR, LOG_NAME)


def log_document(file_path):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_PATH, "a") as log_file:
        log_file.write(f"{timestamp} - {file_path}\n")

def count_all_documents_in_vectorstore():
    filenames = []

    with open(get_logfile_path()) as f:
        for line in f:
            filename = line.split(' - ')[-1].strip()
            filenames.append(filename)

    # Remove duplicates and keep the order of the first occurrence
    unique_filenames = list(dict.fromkeys(filenames))
    return len(unique_filenames)


def is_file_path_in_log(file_path):
    try:
        with open(LOG_PATH, "r") as log_file:
            return any(file_path in line for line in log_file)
    except FileNotFoundError:
        return False


def get_index_file_size(index_path=VECTORSTORE_INDEX_FILE):
    try:
        return os.path.getsize(index_path)
    except FileNotFoundError:
        return None


def load_embedding_model(embedding_model_path=DEFAULT_EMBEDDING_MODEL_PATH):
    return HuggingFaceEmbeddings(
        model_name=embedding_model_path,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_vectorstore(
    as_retriever=False, 
    embedding_model_path=DEFAULT_EMBEDDING_MODEL_PATH
):
    embedding_model = load_embedding_model(embedding_model_path=embedding_model_path)
    vs = FAISS.load_local(DEFAULT_STORAGE_PATH, embedding_model)
    if as_retriever:
        return vs.as_retriever()
    else:
        return vs


def add_document_to_vectorstore(
    file_path,
    storing_path=DEFAULT_STORAGE_PATH,
    embedding_model_path=DEFAULT_EMBEDDING_MODEL_PATH,
):
    embeddings = load_embedding_model(embedding_model_path)

    doc_exists = is_file_path_in_log(file_path)
    if doc_exists:
        new_vectorstore = FAISS.load_local(storing_path, embeddings)
    else:
        loader = PyMuPDFLoader(file_path)
        doc = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
        chunks = text_splitter.split_documents(doc)
        new_vectorstore = FAISS.from_documents(chunks, embeddings)

        index_exists = os.path.exists(f"{storing_path}/index.faiss")
        if index_exists:
            old_vectorstore = FAISS.load_local(storing_path, embeddings)
            new_vectorstore.merge_from(old_vectorstore)

        new_vectorstore.save_local(storing_path)

    log_document(file_path)

    return True


if __name__ == "__main__":
    # Example usage
    pdf_files = ["data/vw_financial_statements_2022.pdf", "data/paul_graham_essay.pdf"]

    for pdf_file in pdf_files:
        add_document_to_vectorstore(pdf_file)
