import base64
import io
import os
import torch
import uvicorn

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

from PIL import Image
from transformers import AutoModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = AutoModel.from_pretrained(
    os.getenv("MODEL", "jinaai/jina-clip-v1"),
    torch_dtype="auto",
    trust_remote_code=True,
).to(device)

model.eval()

app = FastAPI(
    title="LLM Platform Embeddings"
)

class InputItem(BaseModel):
    text: Optional[str] = None
    image: Optional[str] = None

class EmbeddingRequest(BaseModel):
    input: List[InputItem]

@app.post("/embeddings")
async def rerank(request: EmbeddingRequest):
    input = request.input

    data = []
    
    for i, input in enumerate(input):
        if input.text:
            embedding = model.encode_text(input.text)

            data.append({
                "object": "embedding",
                "index": i,
                "embedding": embedding.tolist()
            })
        
        if input.image:
            if input.image.startswith("http://") or input.image.startswith("https://"):
                embedding = model.encode_image(input.image)
            else:
                image_data = base64.b64decode(input.image)

                image = Image.open(io.BytesIO(image_data))
                image = image.convert("RGB")

                embedding = model.encode_image(image)

            data.append({
                "object": "embedding",
                "index": i,
                "embedding": embedding.tolist()
            })
    
    return {
        "object": "list",
        "model": model.name_or_path,
        "data": data
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)