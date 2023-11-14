from typing import Optional, List
from pydantic import BaseModel, Field


class LLMResponse(BaseModel):
    response: str
    prompt_name: str


class Document(BaseModel):
    docid: int
    text: str
    title: str
    emb: Optional[List[float]] = Field(
        None,
        description="Optional document embedding"
    )