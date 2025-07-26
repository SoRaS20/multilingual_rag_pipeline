from fastapi import FastAPI
from app.routes import router
import os
import sys
sys.path.append('.')

app = FastAPI(title="Multilingual RAG API")
app.include_router(router)


def test_embedder():
    try:
        from rag.embedder import embed_text
        result = embed_text(["test"])
        print("✓ Embedder working")
        return True
    except Exception as e:
        print(f"✗ Embedder error: {e}")
        return False

def test_vectorstore():
    try:
        from rag.embedder import embed_query
        from rag.vectorstore import retrieve_similar
        
        # Use actual embedding instead of dummy
        test_embedding = embed_query("test")
        result = retrieve_similar(test_embedding)
        print("✓ Vectorstore working")
        return True
    except Exception as e:
        print(f"✗ Vectorstore error: {e}")
        return False

def test_generator():
    try:
        from rag.generator import generate_answer
        result = generate_answer("test", "test context")
        print("✓ Generator working")
        return True
    except Exception as e:
        print(f"✗ Generator error: {e}")
        return False

def test_memory():
    try:
        from rag.memory import add_to_memory, get_memory
        add_to_memory("test", "test answer")
        memory = get_memory()
        print("✓ Memory working")
        return True
    except Exception as e:
        print(f"✗ Memory error: {e}")
        return False

if __name__ == "__main__":
    print("Testing components...")
    test_embedder()
    test_vectorstore()
    test_generator()
    test_memory()