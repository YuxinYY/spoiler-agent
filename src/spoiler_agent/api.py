from __future__ import annotations

import os

from fastapi import FastAPI

from .infer import classify_text
from .schema import SpoilerRequest, SpoilerResponse

app = FastAPI(title="spoiler-agent")


@app.post("/predict_spoiler", response_model=SpoilerResponse)
def predict_spoiler(input_data: SpoilerRequest) -> SpoilerResponse:
    result = classify_text(input_data.text, input_data.content_type)
    return SpoilerResponse(
        code=200,
        message="success",
        data=result,
        content_type=input_data.content_type,
    )


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
