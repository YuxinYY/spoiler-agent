from __future__ import annotations

from pydantic import BaseModel, Field


class SpoilerClassification(BaseModel):
    has_spoiler: int = Field(..., ge=0, le=1, description="0 or 1")
    confidence: float = Field(..., ge=0.0, le=1.0, description="0.0-1.0")
    spoiler_type: str = Field(..., description="movie|novel|none")
    key_spoiler_sentence: str = Field(..., description="Key spoiler sentence or 'none'")


class SpoilerRequest(BaseModel):
    text: str = Field(..., description="Input text to classify")
    content_type: str = Field("unknown", description="movie|novel|game|book|unknown")


class SpoilerResponse(BaseModel):
    code: int = Field(200, description="HTTP-style status code")
    message: str = Field("success", description="Result message")
    data: SpoilerClassification
    content_type: str = Field("unknown", description="Echoed content type")


def normalize_spoiler_type(value: str | None) -> str:
    if not value:
        return "none"
    value = value.strip().lower()
    mapping = {
        "movie": "movie",
        "film": "movie",
        "novel": "novel",
        "book": "novel",
        "none": "none",
        "no": "none",
        "na": "none",
    }
    return mapping.get(value, value)
