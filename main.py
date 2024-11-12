import os
import io
import base64
import uvicorn
import asyncio
from concurrent.futures import ThreadPoolExecutor
import functools
import logging

from fastapi import FastAPI
from pydantic import BaseModel, field_validator
from typing import List, Optional

from PIL import Image
from transformers import AutoModel
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create thread pool for model inference
thread_pool = ThreadPoolExecutor(max_workers=1)

# Check if MPS is available
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Load the model
model = AutoModel.from_pretrained(
    os.getenv("MODEL", "jinaai/jina-embeddings-v3"),
    trust_remote_code=True
)

# Move the model to the appropriate device
model = model.to(device)
model.eval()

app = FastAPI(
    title="LLM Platform Embeddings"
)

class InputItem(BaseModel):
    text: Optional[str] = None
    image: Optional[str] = None
    max_length: Optional[int] = 8192
    truncate_dim: Optional[int] = 512

class EmbeddingRequest(BaseModel):
    input: List[InputItem]

    @field_validator('input', mode='before')
    def parse_input(cls, v):
        if isinstance(v, str):
            v = [v]
        elif not isinstance(v, list):
            raise ValueError('Input must be a string or a list')
        
        input = []

        for item in v:
            if isinstance(item, str):
                input.append({'text': item})
            elif isinstance(item, dict):
                input.append(item)
            else:
                raise ValueError('Each item in input must be a string or a dictionary')
            
        return input

async def run_in_executor(func, *args, **kwargs):
    """Run a blocking function in the thread pool."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(thread_pool, functools.partial(func, *args, **kwargs))

@app.post("/embeddings")
@app.post("/v1/embeddings")
async def embed(request: EmbeddingRequest):
    input = request.input
    data = []
    
    for i, input in enumerate(input):
        if input.text:
            logger.info(f"Processing text input #{i}: {input.text[:100]}...")
            encode = getattr(model, "encode_text", None)
            
            if encode == None:
                encode = getattr(model, "encode", None)
                
            if encode == None:
                raise ValueError('Model does not have a method to encode text')
            
            # Run model inference in thread pool
            embedding = await run_in_executor(encode, sentences=input.text, truncate_dim=input.truncate_dim)
            logger.info(f"Successfully embedded text #{i}")

            data.append({
                "object": "embedding",
                "index": i,
                "embedding": embedding.tolist()
            })
            
        if input.image:
            logger.info(f"Processing image input #{i}")
            encode = getattr(model, "encode_image", None)
            
            if encode == None:
                raise ValueError('Model does not have a method to encode image')
            
            if input.image.startswith("http://") or input.image.startswith("https://"):
                logger.info(f"Embedding image from URL: {input.image[:100]}...")
                embedding = await run_in_executor(encode, input.image)
            else:
                logger.info("Embedding image from base64 data...")
                image_data = base64.b64decode(input.image)
                image = Image.open(io.BytesIO(image_data))
                image = image.convert("RGB")
                embedding = await run_in_executor(encode, image)
            
            logger.info(f"Successfully embedded image #{i}")
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
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        timeout_keep_alive=120,
        limit_concurrency=5,
        workers=1  # Important: keep single worker when using thread pool
    )