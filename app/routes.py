from fastapi import APIRouter, Query, HTTPException
from rag.retriever import retrieve_chunks
from rag.generator import generate_answer
from rag.memory import add_to_memory, get_memory
from app.models import QueryOutput
import traceback

router = APIRouter()

@router.post(
    "/query",
    response_model=QueryOutput,
    description="Enter your question (in English or Bangla) directly into the input field. This will return a grounded answer with memory history."
)
def get_answer(query: str = Query(..., description="Your question")):
    try:
        print(f"Received query: {query}")
        
        # Step 1: Retrieve chunks
        chunks = retrieve_chunks(query)
        print(f"Retrieved {len(chunks)} chunks")
        
        # Step 2: Create context
        context = "\n".join(chunks)
        print(f"Context length: {len(context)}")
        
        # Step 3: Generate answer
        answer = generate_answer(query, context)
        print(f"Generated answer: {answer[:100]}...")
        
        # Step 4: Add to memory
        add_to_memory(query, answer)
        
        # Step 5: Return response
        memory = get_memory()
        return QueryOutput(answer=answer, memory=memory)
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")