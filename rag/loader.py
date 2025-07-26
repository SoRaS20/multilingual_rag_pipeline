from PyPDF2 import PdfReader
import re

def load_pdf(path):
    reader = PdfReader(path)
    return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

def chunk_text_by_content_type(text, chunk_size=300):
    """
    Intelligently chunk text based on content type for Bengali content
    """
    chunks = []
    
    # Split by double newlines to get paragraphs
    paragraphs = text.split('\n\n')
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
            
        # Handle MCQ chunks (detect questions with options)
        if is_mcq_content(paragraph):
            chunks.append(paragraph)
        # Handle vocabulary entries
        elif is_vocabulary_entry(paragraph):
            chunks.append(paragraph)
        # Handle regular story content
        else:
            # For story content, split by sentences if too long
            if len(paragraph) > chunk_size:
                sentences = split_bengali_sentences(paragraph)
                current_chunk = ""
                
                for sentence in sentences:
                    if len(current_chunk + sentence) <= chunk_size:
                        current_chunk += sentence + " "
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence + " "
                
                if current_chunk:
                    chunks.append(current_chunk.strip())
            else:
                chunks.append(paragraph)
    
    return chunks

def is_mcq_content(text):
    """Detect if text contains MCQ format"""
    mcq_patterns = [
        r'ক\)',  # Bengali option markers
        r'খ\)',
        r'গ\)',
        r'ঘ\)',
        r'উত্তর:',  # Answer marker
        r'\?',  # Question mark
    ]
    return sum(1 for pattern in mcq_patterns if re.search(pattern, text)) >= 3

def is_vocabulary_entry(text):
    """Detect if text is a vocabulary entry"""
    return '-' in text and len(text.split('-')) == 2 and len(text) < 200

def split_bengali_sentences(text):
    """Split Bengali text into sentences"""
    # Bengali sentence endings
    sentence_endings = r'[।!?]'
    sentences = re.split(sentence_endings, text)
    return [s.strip() for s in sentences if s.strip()]
