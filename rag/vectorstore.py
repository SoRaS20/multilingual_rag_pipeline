import chromadb
import os
import json
from typing import List, Dict, Any

def get_client():
    try:
        db_path = "./db"
        if not os.path.exists(db_path):
            os.makedirs(db_path)
            
        client = chromadb.PersistentClient(path=db_path)
        return client
    except Exception as e:
        print(f"Error initializing ChromaDB client: {e}")
        raise

def store_embeddings(texts: List[str], embeddings: List[List[float]], content_types: List[str] = None, embedding_type: str = "gemini"):
    """
    Store text chunks with their embeddings and content type metadata
    
    Args:
        texts: List of text chunks
        embeddings: List of embedding vectors
        content_types: List of content types for each chunk
        embedding_type: Type of embedding model used ("clip", "gemini", "hybrid")
    """
    try:
        client = get_client()
        collection = client.get_or_create_collection(name="hsc_bangla")
        
        # Prepare metadata for each chunk
        metadatas = []
        embedding_dim = len(embeddings[0]) if embeddings else 0
        
        for i, text in enumerate(texts):
            content_type = content_types[i] if content_types else detect_content_type(text)
            metadata = {
                "content_type": content_type,
                "chunk_length": len(text),
                "chunk_index": i,
                "embedding_type": embedding_type,
                "embedding_dimension": embedding_dim
            }
            metadatas.append(metadata)
        
        # Generate unique IDs based on embedding type and index
        ids = [f"{embedding_type}_chunk_{i}" for i in range(len(texts))]
        
        # Add to collection
        collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"Stored {len(texts)} chunks successfully using {embedding_type} embeddings (dim: {embedding_dim})")
        
    except Exception as e:
        print(f"Error storing embeddings: {e}")
        raise

def retrieve_similar(embedding: List[float], n_results: int = 5, content_type_filter: str = None):
    """
    Retrieve similar chunks with optional content type filtering
    """
    try:
        client = get_client()
        collection = client.get_or_create_collection(name="hsc_bangla")
        
        # Build where clause for filtering
        where_clause = None
        if content_type_filter:
            where_clause = {"content_type": content_type_filter}
        
        results = collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            where=where_clause
        )
        
        return results
        
    except Exception as e:
        print(f"Error in retrieve_similar: {e}")
        raise

def detect_content_type(text: str) -> str:
    """
    Detect the type of content (story, mcq, vocabulary, etc.)
    """
    if any(marker in text for marker in ['ক)', 'খ)', 'গ)', 'ঘ)', 'উত্তর:']):
        return "mcq"
    elif '-' in text and len(text.split('-')) == 2 and len(text) < 200:
        return "vocabulary"
    elif any(word in text for word in ['রবীন্দ্রনাথ', 'ঠাকুর', 'জন্ম', 'মৃত্যু']):
        return "biography"
    else:
        return "story"

def get_collection_stats():
    """
    Get statistics about the stored collection
    """
    try:
        client = get_client()
        collection = client.get_or_create_collection(name="hsc_bangla")
        count = collection.count()
        return {"total_chunks": count}
    except Exception as e:
        print(f"Error getting collection stats: {e}")
        return {"total_chunks": 0}