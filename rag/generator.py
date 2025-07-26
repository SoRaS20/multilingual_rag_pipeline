import os
import google.generativeai as genai

# Configure the API key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable is required")
genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-2.0-flash")

# def generate_answer(query, contexts):
#     prompt = f"Answer the question based only on the context below.\n\nContext:\n{contexts}\n\nQuestion: {query}\nAnswer:"
#     return model.generate_content(prompt).text.strip()
def generate_answer(query: str, contexts: str) -> str:
    prompt = f"""
    Answer the question based **only** on the provided context. 
    Respond in the **same language** as the question (Bangla or English). 
    If the answer is not found, say "I don't know" (English) or "আমি জানি না" (Bangla).

    Context:
    {contexts}

    Question: {query}
    Answer: (in the language of the question)
    """
    return model.generate_content(prompt).text.strip()