# app/api.py
from fastapi import APIRouter, Header, HTTPException, Depends
from typing import Annotated, Optional
import time
from datetime import datetime

from .schemas import (
    QueryRequest, 
    SimpleQueryResponse, 
    DetailedQueryResponse, 
    HealthResponse,
    ErrorResponse
)
from .processor import RAGProcessor
from .config import settings

router = APIRouter()

# Singleton processor instance
_processor_instance = None

def get_processor() -> RAGProcessor:
    """Dependency to get processor instance"""
    global _processor_instance
    if _processor_instance is None:
        _processor_instance = RAGProcessor()
    return _processor_instance

def verify_authorization(authorization: Annotated[Optional[str], Header()] = None):
    """Verify Bearer token authorization"""
    if not authorization:
        raise HTTPException(
            status_code=401, 
            detail="Authorization header is required"
        )
    
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401, 
            detail="Authorization header must be Bearer token"
        )
    
    token = authorization.replace("Bearer ", "")
    if token != settings.AUTH_TOKEN:
        raise HTTPException(
            status_code=401, 
            detail="Invalid authorization token"
        )
    
    return token

@router.post(
    "/hackrx/run", 
    response_model=SimpleQueryResponse,
    summary="Process document and questions",
    description="Processes a document URL and questions, returning simple string answers",
    responses={
        200: {"description": "Successfully processed queries"},
        401: {"description": "Unauthorized - Invalid or missing Bearer token"},
        422: {"description": "Validation Error"},
        500: {"description": "Internal Server Error"}
    }
)
async def run_hackrx_query(
    request: QueryRequest, 
    authorization: Annotated[Optional[str], Header()] = None,
    processor: RAGProcessor = Depends(get_processor)
):
    """
    Main endpoint that processes a document URL and a list of questions.
    
    **Authentication Required**: Bearer token in Authorization header
    
    **Request Format**:
    ```json
    {
        "documents": "https://example.com/document.pdf",
        "questions": [
            "What is the grace period for premium payment?",
            "What are the medical expenses covered?"
        ]
    }
    ```
    
    **Response Format**:
    ```json
    {
        "answers": [
            "Grace period is 30 days from the due date...",
            "Medical expenses covered include..."
        ]
    }
    ```
    """
    # Verify authentication
    verify_authorization(authorization)
    
    try:
        start_time = time.time()
        
        answer_list = processor.process_document_and_questions(
            doc_url=str(request.documents), 
            questions=request.questions
        )
        
        processing_time = time.time() - start_time
        print(f"Request processed in {processing_time:.2f} seconds")
        
        return SimpleQueryResponse(answers=answer_list)
        
    except HTTPException:
        # Re-raise HTTP exceptions (like auth errors)
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )

@router.post(
    "/hackrx/run/detailed", 
    response_model=DetailedQueryResponse,
    summary="Process document and questions (detailed format)",
    description="Processes a document URL and questions, returning detailed answers with explainability"
)
async def run_detailed_query(
    request: QueryRequest, 
    authorization: Annotated[Optional[str], Header()] = None,
    processor: RAGProcessor = Depends(get_processor)
):
    """
    Enhanced endpoint that returns detailed answers with explainability,
    source information, and confidence scores.
    """
    # Verify authentication
    verify_authorization(authorization)
    
    try:
        start_time = time.time()
        
        detailed_answers = processor.process_document_and_questions_detailed(
            doc_url=str(request.documents), 
            questions=request.questions
        )
        
        processing_time = time.time() - start_time
        
        return DetailedQueryResponse(
            answers=detailed_answers,
            document_info={
                "url": str(request.documents),
                "questions_count": len(request.questions)
            },
            processing_time=processing_time
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions (like auth errors)
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check if the service is running and models are loaded"
)
async def health_check(processor: RAGProcessor = Depends(get_processor)):
    """Health check endpoint to verify service status"""
    try:
        # Test if the processor is working
        model_loaded = hasattr(processor, 'embedding_model') and processor.embedding_model is not None
        
        return HealthResponse(
            status="healthy" if model_loaded else "degraded",
            model_loaded=model_loaded,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            timestamp=datetime.now().isoformat()
        )

@router.get(
    "/models/info",
    summary="Get model information",
    description="Get information about loaded models and their capabilities"
)
async def get_model_info(processor: RAGProcessor = Depends(get_processor)):
    """Get information about the loaded models"""
    try:
        return {
            "embedding_model": "all-MiniLM-L6-v2",
            "embedding_dimensions": 384,
            "llm_model": "llama3",
            "chunk_size": processor.chunk_size,
            "overlap": processor.overlap,
            "supported_formats": ["pdf", "docx", "doc", "email", "text"],
            "features": [
                "semantic_search",
                "intelligent_chunking", 
                "query_intent_analysis",
                "explainable_answers",
                "confidence_scoring",
                "source_tracking"
            ]
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting model info: {str(e)}"
        )