from rag.embedder import embed_query
from rag.vectorstore import retrieve_similar
import re


def retrieve_chunks(query: str, n_results: int = 5, use_clip: bool = True):
    """
    Retrieve relevant chunks based on query with smart filtering and CLIP support
    
    Args:
        query: Search query in Bengali or English
        n_results: Number of results to return
        use_clip: Whether to use CLIP embeddings (faster and better for multimodal)
    """
    # Generate embedding for the query using CLIP or Gemini
    query_embedding = embed_query(query, use_clip=use_clip)

    # Detect query type and adjust retrieval strategy
    query_type = detect_query_type(query)

    # Retrieve with appropriate filtering
    if query_type == "vocabulary":
        results = retrieve_similar(query_embedding, n_results, content_type_filter="vocabulary")
    elif query_type == "mcq":
        results = retrieve_similar(query_embedding, n_results, content_type_filter="mcq")
    else:
        # For general questions, search all content types
        results = retrieve_similar(query_embedding, n_results)

    # Return the documents
    if results and 'documents' in results and results['documents']:
        return results['documents'][0]
    else:
        return []


def detect_query_type(query: str) -> str:
    """
    Detect the type of query to improve retrieval
    """
    vocabulary_keywords = ['অর্থ', 'মানে', 'শব্দার্থ', 'টীকা']
    mcq_keywords = ['বিকল্প', 'সঠিক উত্তর', 'কোনটি', 'কী', 'কে', 'কোথায়', 'কখন']

    if any(keyword in query for keyword in vocabulary_keywords):
        return "vocabulary"
    elif any(keyword in query for keyword in mcq_keywords):
        return "mcq"
    else:
        return "general"


def get_contextual_chunks(query: str, n_results: int = 3):
    """
    Get chunks with additional context for better answer generation
    """
    chunks = retrieve_chunks(query, n_results)

    # Add context information
    contextual_info = []
    for chunk in chunks:
        content_type = detect_content_type_from_text(chunk)
        contextual_info.append({
            "text": chunk,
            "type": content_type,
            "relevance": "high"  # Could implement scoring
        })

    return contextual_info


def detect_content_type_from_text(text: str) -> str:
    """Helper function to detect content type from text"""
    if any(marker in text for marker in ['ক)', 'খ)', 'গ)', 'ঘ)', 'উত্তর:']):
        return "mcq"
    elif '-' in text and len(text.split('-')) == 2 and len(text) < 200:
        return "vocabulary"
    else:
        return "story"
