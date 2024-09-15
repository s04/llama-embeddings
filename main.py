import torch
import uvicorn

from fastapi import FastAPI
from typing import List, Optional
from pydantic import BaseModel

from transformers import AutoModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = AutoModel.from_pretrained(
    'jinaai/jina-clip-v1',
    torch_dtype="auto",
    trust_remote_code=True,
)

model.to(device)
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
            embedding = model.encode_image(input.image)

            data.append({
                "object": "embedding",
                "index": i,
                "embedding": embedding.tolist()
            })
    
    return {
        "data": data
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)