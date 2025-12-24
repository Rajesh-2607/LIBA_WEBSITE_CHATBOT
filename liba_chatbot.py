import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key
client = openai.OpenAI()

# Load SentenceTransformer model once
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index and metadata once at startup
INDEX_FILE = "college_index.faiss"
METADATA_FILE = "college_metadata.pkl"

print("Loading FAISS index and metadata...")
index = faiss.read_index(INDEX_FILE)
with open(METADATA_FILE, "rb") as f:
    metadata = pickle.load(f)

# Handle different formats of metadata
if isinstance(metadata, tuple) and len(metadata) == 2:
    # Old format: (documents, metadatas)
    documents, metadatas = metadata
elif isinstance(metadata, list) and isinstance(metadata[0], dict):
    # New format: list of dicts with filename, chunk, and content
    documents = [m["text"] for m in metadata]
    metadatas = metadata
else:
    raise ValueError("❌ Unexpected metadata format in pickle file")

print(f"FAISS index loaded with {index.ntotal} vectors")
print(f"Metadata loaded with {len(documents)} chunks")

def answer_query(query, k=10, model_name="gpt-4o"):
    # Generate query embedding and normalize
    q_embedding = model.encode([query], convert_to_numpy=True)
    q_embedding = normalize(q_embedding, axis=1)

    # Search FAISS index for top-k relevant chunks
    D, I = index.search(q_embedding, k)
    relevant_chunks = [documents[i] for i in I[0]]

    # Combine relevant chunks as context
    context = "\n\n".join(relevant_chunks)

    # Prepare prompt messages for OpenAI chat completion
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful, conversational and optimistic assistant for LIBA, a Jesuit Business School's website. "
                "Use the context provided to answer user queries positively and informatively but while responding do not mention refer to “the provided context”, “the documents”, or any source. "
                "If a document contains mixed or negative content, highlight only the constructive or commendable aspects. "
                "Reframe mildly negative details in a diplomatic and encouraging manner, emphasizing LIBA’s strengths, values, and continuous improvement. "
                "If the answer is not directly stated, infer it logically based on context. "
                "Do not say \"no information\" unless it's clearly and entirely absent. "
                "Give directions to contact LIBA incase for further details when needed.""Give me the answer in two or three lines."
            )
        },
        {
            "role": "user",
            "content": f"### CONTEXT:\n{context}\n\n### QUESTION:\n{query}"
        }
    ]

    # Call OpenAI chat completion
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.5
    )

    return response.choices[0].message.content.strip()

