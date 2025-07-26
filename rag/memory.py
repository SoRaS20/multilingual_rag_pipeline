from collections import deque

short_term_memory = deque(maxlen=5)

def add_to_memory(query, answer):
    short_term_memory.append((query, answer))

def get_memory():
    return list(short_term_memory)
