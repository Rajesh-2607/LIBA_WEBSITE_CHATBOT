import os
from pathlib import Path
import pickle
import numpy as np
import faiss
import warnings
from pypdf import PdfReader
from pypdf.errors import PdfReadWarning
from sentence_transformers import SentenceTransformer

# ======================
# CONFIGURATION
# ======================
FOLDER_PATH = "college_pages_2"           # 📁 Folder containing PDFs (and subfolders)
INDEX_FILE = "college_index.faiss"   # 🗃️ Output FAISS index file
METADATA_FILE = "college_metadata.pkl"  # 🏷️ Output metadata file
CHUNK_SIZE = 500                     # 🧩 Characters per chunk
CHUNK_OVERLAP = 50                   # 🔗 Overlap between chunks
EMBEDDING_MODEL = "all-MiniLM-L6-v2" # 🧠 Embedding model

# ======================
# SUPPRESS PDF WARNINGS
# ======================
warnings.filterwarnings("ignore", category=PdfReadWarning)

# ======================
# CLEANUP OLD FILES
# ======================
if os.path.exists(INDEX_FILE):
    print(f"🗑️  Removing old FAISS index: {INDEX_FILE}")
    os.remove(INDEX_FILE)
if os.path.exists(METADATA_FILE):
    print(f"🗑️  Removing old meta {METADATA_FILE}")
    os.remove(METADATA_FILE)

# ======================
# TEXT EXTRACTION & CHUNKING
# ======================
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            text += f"\n--- Page {i+1} ---\n" + page_text
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    if not text.strip():
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
        if start >= len(text):
            break
    return chunks

# ======================
# MAIN PROCESSING
# ======================
print("🔍 Scanning for PDFs...")
pdf_files = list(Path(FOLDER_PATH).rglob("*.pdf"))
print(f"📄 Found {len(pdf_files)} PDF files")

if len(pdf_files) == 0:
    raise FileNotFoundError(f"No PDFs found in '{FOLDER_PATH}' or its subfolders.")

model = SentenceTransformer(EMBEDDING_MODEL)
print(f"🧠 Loaded embedding model: {EMBEDDING_MODEL}")

metadata = []
embeddings_list = []
chunk_id = 0

for pdf_path in pdf_files:
    print(f"📄 Processing: {pdf_path}")
    try:
        text = extract_text_from_pdf(pdf_path)
        if not text.strip():
            print(f"⚠️  Warning: No text extracted from {pdf_path}")
            continue

        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        print(f"  ➤ Split into {len(chunks)} chunks")

        for chunk in chunks:
            embedding = model.encode([chunk], show_progress_bar=False)[0]
            embeddings_list.append(embedding)

            metadata.append({
                "id": chunk_id,
                "file_path": str(pdf_path),
                "text": chunk,  # ✅ Full text for RAG/chatbot use
                "text_preview": chunk[:200] + ("..." if len(chunk) > 200 else ""),
                "chunk_index": len(metadata),
            })
            chunk_id += 1

    except Exception as e:
        print(f"❌ Error processing {pdf_path}: {e}")
        continue

# ======================
# VALIDATE & BUILD FAISS INDEX
# ======================
if len(embeddings_list) == 0:
    raise ValueError("⛔ No embeddings generated. Check your PDFs or text extraction.")

assert len(embeddings_list) == len(metadata), (
    f"Mismatch: {len(embeddings_list)} embeddings vs {len(metadata)} metadata entries"
)

embeddings = np.array(embeddings_list).astype('float32')
dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)
faiss.normalize_L2(embeddings)
index.add(embeddings)

print(f"✅ FAISS index built with {index.ntotal} vectors")
print(f"✅ Metadata built with {len(metadata)} entries")
assert index.ntotal == len(metadata), "FATAL: FAISS and metadata size mismatch after build!"

# ======================
# SAVE OUTPUTS
# ======================
faiss.write_index(index, INDEX_FILE)
with open(METADATA_FILE, "wb") as f:
    pickle.dump(metadata, f)

print(f"💾 Saved FAISS index to: {INDEX_FILE}")
print(f"💾 Saved metadata to: {METADATA_FILE}")
print("🎉 Index build completed successfully!")
