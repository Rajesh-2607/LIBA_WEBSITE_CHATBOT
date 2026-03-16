import os
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import faiss
import config

load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load the FAISS index and metadata
try:
    index = faiss.read_index(config.INDEX_PATH)
    with open(config.METADATA_PATH, "rb") as f:
        texts = pickle.load(f)
except FileNotFoundError:
    print("Error: Index or metadata file not found.")
    print("Please run the `build.py` script first to create the vector store.")
    exit()

# Load the sentence transformer model
model = SentenceTransformer(config.EMBEDDING_MODEL)

def answer_query(user_query):
    # Encode the user's query
    query_embedding = model.encode([user_query])

    # Perform similarity search
    D, I = index.search(np.array(query_embedding).astype("float32"), k=config.FAISS_TOP_K)

    # Get the most relevant text chunks
    relevant_texts = [texts[i] for i in I[0]]
    
    # Create the prompt for the language model
    prompt = (
        "You are a helpful, conversational and optimistic assistant for LIBA, a Jesuit Business School's website. "
        "Use the context provided to answer user queries positively and informatively but while responding do not mention refer to “the provided context”, “the documents”, or any source. "
        "If a document contains mixed or negative content, highlight only the constructive or commendable aspects. "
        "Reframe mildly negative details in a diplomatic and encouraging manner, emphasizing LIBA’s strengths, values, and continuous improvement. "
        "If the answer is not directly stated, infer it logically based on context. "
        "Do not say \"no information\" unless it's clearly and entirely absent. "
        "Give directions to contact LIBA incase for further details when needed."
    )

    # Prepare prompt messages for OpenAI chat completion
    messages = [
        {
            "role": "system",
            "content": prompt
        },
        {
            "role": "user",
            "content": f"### CONTEXT:\n{relevant_texts}\n\n### QUESTION:\n{user_query}"
        }
    ]

    # Call OpenAI chat completion
    response = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=messages,
        temperature=0.5
    )

    return response.choices[0].message.content.strip()

