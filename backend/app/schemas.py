from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class SourceDocument(BaseModel):
    content: str
    metadata: dict
    score: float


class QueryRequest(BaseModel):
    question: str = Field(..., description="User's compliance question")
    top_k: int = Field(default=5, description="Number of documents to retrieve")


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]
    confidence: float
    model: str
    provider: str = "openai-compatible"
    processing_time_ms: float
    reasoning: Optional[str] = None


class InputCheckRequest(BaseModel):
    product_name: str
    ingredients: List[str]
    intended_use: str = Field(default="crop_production")


class InputCheckResponse(BaseModel):
    product_name: str
    approved: bool
    status: str
    details: List[str]
    sources: List[SourceDocument]
