import sys
sys.path.append('.')

from rag.loader import load_pdf, chunk_text_by_content_type
from rag.embedder import embed_text
from rag.vectorstore import store_embeddings, get_collection_stats
from app.config import PDF_PATH
import os

def process_bengali_pdf():
    """
    Process the Bengali PDF and store embeddings
    """
    print("🔄 Processing Bengali PDF...")
    
    # Check if PDF exists
    if not os.path.exists(PDF_PATH):
        print(f"❌ PDF not found at: {PDF_PATH}")
        return False
    
    # Step 1: Load PDF content
    print("📖 Loading PDF content...")
    text = load_pdf(PDF_PATH)
    print(f"✅ Loaded {len(text)} characters from PDF")
    
    # Step 2: Chunk the text intelligently
    print("✂️ Chunking text by content type...")
    chunks = chunk_text_by_content_type(text, chunk_size=300)
    print(f"✅ Created {len(chunks)} chunks")
    
    # Step 3: Generate embeddings with CLIP support
    print("🧠 Generating embeddings with CLIP...")
    embeddings = embed_text(chunks, use_clip=True)  # Use CLIP for better multilingual support
    print(f"✅ Generated {len(embeddings)} embeddings")
    
    # Step 4: Store in vector database with embedding type metadata
    print("💾 Storing in vector database...")
    
    # Detect embedding type based on dimension
    embedding_dim = len(embeddings[0]) if embeddings else 0
    if embedding_dim == 512:
        embedding_type = "clip"
    elif embedding_dim == 768:
        embedding_type = "gemini"
    else:
        embedding_type = "unknown"
    
    store_embeddings(chunks, embeddings, embedding_type=embedding_type)
    
    # Step 5: Verify storage
    stats = get_collection_stats()
    print(f"✅ Processing complete! Total chunks in database: {stats['total_chunks']}")
    print(f"📊 Using {embedding_type} embeddings (dimension: {embedding_dim})")
    
    return True

if __name__ == "__main__":
    success = process_bengali_pdf()
    if success:
        print("\n🎉 Your Bengali PDF is now ready for Q&A!")
        print("You can now ask questions like:")
        print("- অনুপমের অভিভাবক কে ছিলেন?")
        print("- অপরিচিতা গল্পের মূল বিষয় কী?")
        print("- ফল্গু শব্দের অর্থ কী?")
    else:
        print("\n❌ Processing failed. Please check the errors above.")
