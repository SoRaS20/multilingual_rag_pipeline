from pydantic import BaseModel
from typing import List, Tuple

class QueryInput(BaseModel):
    query: str

class QueryOutput(BaseModel):
    answer: str
    memory: List[Tuple[str, str]]
