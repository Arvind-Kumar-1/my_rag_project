# procedural_test.py
import fitz
import faiss
import numpy as np
import requests
import ollama
from sentence_transformers import SentenceTransformer
from typing import List

def download_and_chunk_pdf(url: str) -> List[str]:
    print("Step 1: Downloading and chunking PDF...")
    try:
        response = requests.get(url)
        response.raise_for_status()
        doc = fitz.open(stream=response.content, filetype="pdf")
        chunks = [chunk.strip() for page in doc for chunk in page.get_text("text").strip().split('\n\n') if chunk.strip()]
        print(f"   Success: Found {len(chunks)} text chunks.")
        return chunks
    except Exception as e:
        print(f"   ERROR during PDF processing: {e}")
        return []

def generate_answer(question: str, context: str) -> str:
    print(f"Step 4: Generating answer for question: '{question[:30]}...'")
    prompt = f"Based ONLY on the provided context, provide a concise and direct answer to the following question.\n\nCONTEXT:\n---\n{context}\n---\n\nQUESTION: {question}\n\nANSWER:"
    response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': prompt}], options={'temperature': 0.0})
    answer = response['message']['content'].strip()
    print(f"   Success: Received answer.")
    return answer

def run_procedural_test():
    print("--- Starting Procedural Test ---")

    # Load the model into a simple variable, not a class attribute.
    print("Step A: Loading embedding model...")
    try:
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("   Success: Embedding model loaded.")
    except Exception as e:
        print(f"   CRITICAL FAILURE: Could not load SentenceTransformer model. Error: {e}")
        return

    doc_url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
    questions = [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?"
    ]

    text_chunks = download_and_chunk_pdf(doc_url)
    if not text_chunks:
        print("Test stopped because PDF processing failed.")
        return

    print("Step 2: Encoding all text chunks...")
    chunk_embeddings = embedding_model.encode(text_chunks)
    index = faiss.IndexFlatL2(chunk_embeddings.shape[1])
    index.add(np.array(chunk_embeddings, dtype=np.float32))
    print("   Success: FAISS index created.")

    final_answers = []
    for question in questions:
        print(f"\n--- Processing Question: '{question}' ---")
        print("Step 3: Encoding question...")
        question_embedding = embedding_model.encode([question])
        
        k = 5
        _, indices = index.search(np.array(question_embedding, dtype=np.float32), k)
        retrieved_chunks = [text_chunks[i] for i in indices[0]]
        context_text = "\n\n".join(retrieved_chunks)
        
        answer = generate_answer(question, context_text)
        final_answers.append(answer)

    print("\n\n--- FINAL RESULTS ---")
    for i, answer in enumerate(final_answers):
        print(f"  Answer to question {i+1}: {answer}")
    print("--- TEST COMPLETE ---")

if __name__ == "__main__":
    run_procedural_test()