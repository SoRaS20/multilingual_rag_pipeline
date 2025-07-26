import re
from nltk.tokenize import sent_tokenize

def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()

def chunk_text(text, chunk_size=300):
    sentences = sent_tokenize(text)
    chunks, chunk = [], ""
    for sent in sentences:
        if len(chunk) + len(sent) <= chunk_size:
            chunk += " " + sent
        else:
            chunks.append(chunk.strip())
            chunk = sent
    if chunk:
        chunks.append(chunk.strip())
    return chunks
