# app/schemas.py
from pydantic import BaseModel, HttpUrl, Field
from typing import List, Dict, Any, Optional

class QueryRequest(BaseModel):
    documents: HttpUrl
    questions: List[str] = Field(..., min_items=1, description="List of questions to ask about the document")

class SimpleQueryResponse(BaseModel):
    """Simple response format for compatibility with existing API"""
    answers: List[str]

class SourceInfo(BaseModel):
    """Information about a source chunk used in the answer"""
    source_id: int
    section: Optional[str] = None
    similarity_score: float
    chunk_id: int

class DetailedAnswer(BaseModel):
    """Detailed answer with explainability"""
    answer: str = Field(..., description="The main answer to the question")
    evidence: Optional[str] = Field(None, description="Supporting evidence from the document")
    conditions: Optional[str] = Field(None, description="Any conditions or limitations mentioned")
    confidence: str = Field("Medium", description="Confidence level: High/Medium/Low")
    sources: List[SourceInfo] = Field(default_factory=list, description="Source chunks used")
    generation_time: float = Field(0.0, description="Time taken to generate the answer in seconds")
    raw_response: Optional[str] = Field(None, description="Raw LLM response for debugging")

class DetailedQueryResponse(BaseModel):
    """Enhanced response with detailed answers and explainability"""
    answers: List[DetailedAnswer]
    document_info: Optional[Dict[str, Any]] = Field(None, description="Information about the processed document")
    processing_time: float = Field(0.0, description="Total processing time in seconds")

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    timestamp: str

class ErrorResponse(BaseModel):
    """Error response format"""
    error: str
    detail: Optional[str] = None
    timestamp: str