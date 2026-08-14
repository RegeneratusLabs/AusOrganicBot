from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from app.config import MAX_QUESTION_LENGTH, MAX_TOP_K, MAX_INGREDIENTS


class SourceDocument(BaseModel):
    content: str
    metadata: dict
    score: float


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=MAX_QUESTION_LENGTH, description="User's compliance question")
    top_k: int = Field(default=5, ge=1, le=MAX_TOP_K, description="Number of documents to retrieve")
    country: str = Field(default="au", pattern=r"^[a-z]{2}$", description="Country code whose standards to query")


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]
    confidence: float
    model: str
    provider: str = "openai-compatible"
    processing_time_ms: float
    reasoning: Optional[str] = None
    refused: Optional[bool] = None
    country: str = "au"


class InputCheckRequest(BaseModel):
    product_name: str = Field(..., min_length=1, max_length=200)
    ingredients: List[str] = Field(default_factory=list, max_length=MAX_INGREDIENTS)
    intended_use: str = Field(default="crop_production", max_length=100)
    country: str = Field(default="au", pattern=r"^[a-z]{2}$")


class InputCheckResponse(BaseModel):
    product_name: str
    approved: bool
    status: str
    details: List[str]
    sources: List[SourceDocument]
    country: str = "au"
