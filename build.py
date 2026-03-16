import os
import pickle
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from dotenv import load_dotenv
import config

load_dotenv()

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return None

# Main function to build the vector store
def build_vector_store():
    # Get all PDF files from the specified folder
    all_files = []
    for root, _dirs, files in os.walk(config.DATA_PATH):
        for file in files:
            if file.lower().endswith(".pdf"):
                all_files.append(os.path.join(root, file))

    # Extract text from all PDFs
    documents_text = [extract_text_from_pdf(file) for file in all_files]
    documents_text = [text for text in documents_text if text]  # Filter out None values

    if not documents_text:
        print("No text could be extracted from the PDF files. Aborting.")
        return

    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    texts = text_splitter.create_documents(documents_text)
    
    print(f"Number of text chunks: {len(texts)}")

    # Generate embeddings
    print("Generating embeddings...")
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    embeddings = model.encode([text.page_content for text in texts], show_progress_bar=True)
    
    # Create FAISS index
    print("Creating FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype("float32"))

    # Save the FAISS index and the texts
    faiss.write_index(index, config.INDEX_PATH)
    with open(config.METADATA_PATH, "wb") as f:
        pickle.dump(texts, f)

    print("Vector store built and saved successfully.")

if __name__ == "__main__":
    build_vector_store()
