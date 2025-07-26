import os
import google.generativeai as genai
import time
from typing import List

# Try to import CLIP for enhanced multimodal embeddings
try:
    import torch
    import clip
    from PIL import Image
    CLIP_AVAILABLE = True
    print("🎯 CLIP detected - Enhanced multimodal embeddings available!")
except ImportError:
    CLIP_AVAILABLE = False
    print("📝 CLIP not available - Using Gemini embeddings (install torch and clip for multimodal support)")

# Configure Gemini API
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable is required")
genai.configure(api_key=api_key)

# Global CLIP model (loaded on first use)
_clip_model = None
_clip_preprocess = None
_device = None

def _load_clip_model():
    """Load CLIP model on first use"""
    global _clip_model, _clip_preprocess, _device
    if _clip_model is None and CLIP_AVAILABLE:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Loading CLIP model on {_device}...")
        _clip_model, _clip_preprocess = clip.load("ViT-B/32", device=_device)
        _clip_model.eval()
        print("✅ CLIP model loaded successfully")

def embed_text_with_clip(texts: List[str]) -> List[List[float]]:
    """
    Embed text using CLIP (faster and better for multimodal content)
    """
    if not CLIP_AVAILABLE:
        raise ValueError("CLIP not available - install torch and clip packages")
    
    _load_clip_model()
    
    with torch.no_grad():
        # Tokenize and encode text
        text_tokens = clip.tokenize(texts, truncate=True).to(_device)
        text_embeddings = _clip_model.encode_text(text_tokens)
        
        # Normalize embeddings (important for similarity search)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        
        return text_embeddings.cpu().numpy().tolist()

def embed_text(texts: List[str], use_clip: bool = True) -> List[List[float]]:
    """
    Generate embeddings for Bengali/multilingual text with CLIP support
    
    Args:
        texts: List of text strings to embed
        use_clip: Whether to use CLIP (if available) or fallback to Gemini
    
    Returns:
        List of embeddings
    """
    # Try CLIP first if requested and available
    if use_clip and CLIP_AVAILABLE:
        try:
            print(f"🎯 Using CLIP embeddings for {len(texts)} texts...")
            embeddings = embed_text_with_clip(texts)
            print(f"✅ Generated {len(embeddings)} CLIP embeddings (dim: {len(embeddings[0])})")
            return embeddings
        except Exception as e:
            print(f"⚠️ CLIP embedding failed, falling back to Gemini: {e}")
    
    # Fallback to Gemini implementation
    print(f"📝 Using Gemini embeddings for {len(texts)} texts...")
    embeddings = []
    
    for i, text in enumerate(texts):
        try:
            # Add a small delay to avoid rate limiting
            if i > 0:
                time.sleep(0.1)
                
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_document"
            )
            embeddings.append(result["embedding"])
            
        except Exception as e:
            print(f"Error embedding text chunk {i}: {e}")
            # Return a zero vector as fallback
            embeddings.append([0.0] * 768)  # Default embedding size
    
    print(f"✅ Generated {len(embeddings)} Gemini embeddings")
    return embeddings

def embed_query(query: str, use_clip: bool = True) -> List[float]:
    """
    Generate embedding for a single query
    """
    embeddings = embed_text([query], use_clip=use_clip)
    return embeddings[0] if embeddings else [0.0] * 768

def embed_image(image_path: str) -> List[float]:
    """
    Embed image using CLIP (only available if CLIP is installed)
    
    Args:
        image_path: Path to image file
        
    Returns:
        Image embedding
    """
    if not CLIP_AVAILABLE:
        raise ValueError("Image embedding requires CLIP - install torch and clip packages")
    
    _load_clip_model()
    
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = _clip_preprocess(image).unsqueeze(0).to(_device)
        
        with torch.no_grad():
            # Encode image
            image_embedding = _clip_model.encode_image(image_tensor)
            
            # Normalize embedding
            image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
            
            return image_embedding.cpu().numpy()[0].tolist()
    except Exception as e:
        print(f"Error embedding image {image_path}: {e}")
        # Return zero vector as fallback
        return [0.0] * 512  # CLIP embedding dimension

def embed_multimodal_content(texts: List[str], 
                           image_paths: List[str] = None,
                           fusion_method: str = "average") -> List[List[float]]:
    """
    Embed content that contains both text and images (textbook content)
    Only works if CLIP is available
    
    Args:
        texts: List of text chunks
        image_paths: List of image paths (can be None for some chunks)
        fusion_method: How to combine text and image embeddings
                      - "average": Average text and image embeddings
                      - "text_only": Use only text embeddings
                      
    Returns:
        List of combined embeddings
    """
    if not CLIP_AVAILABLE:
        print("⚠️ Multimodal embedding requires CLIP - using text-only embeddings")
        return embed_text(texts, use_clip=False)
    
    # Get text embeddings
    text_embeddings = embed_text(texts, use_clip=True)
    
    # If no images, return text embeddings
    if not image_paths or fusion_method == "text_only":
        return text_embeddings
    
    # Process images and combine
    combined_embeddings = []
    for i, (text_emb, img_path) in enumerate(zip(text_embeddings, image_paths)):
        if img_path and os.path.exists(img_path):
            try:
                img_emb = embed_image(img_path)
                
                if fusion_method == "average":
                    # Average text and image embeddings (same dimension)
                    combined_emb = [(t + i) / 2 for t, i in zip(text_emb, img_emb)]
                else:
                    combined_emb = text_emb
                    
                combined_embeddings.append(combined_emb)
                print(f"✅ Combined text+image embedding for chunk {i}")
            except Exception as e:
                print(f"⚠️ Failed to process image for chunk {i}: {e}")
                combined_embeddings.append(text_emb)
        else:
            combined_embeddings.append(text_emb)
    
    return combined_embeddings

# Convenience functions for different use cases
def embed_bengali_textbook(texts: List[str], 
                          image_paths: List[str] = None) -> List[List[float]]:
    """
    Optimized embedding for Bengali textbook content
    Uses CLIP for better multimodal understanding
    """
    print("📚 Processing Bengali textbook content...")
    
    if image_paths:
        return embed_multimodal_content(texts, image_paths, fusion_method="average")
    else:
        return embed_text(texts, use_clip=True)