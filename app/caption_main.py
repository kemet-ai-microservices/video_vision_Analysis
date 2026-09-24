"""FastAPI service for MiniCPM-V keyframe captioning."""

from __future__ import annotations

import os
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from loguru import logger
from transformers import AutoModelForImageTextToText, AutoProcessor


class MiniCPMCaptionEngine:
    def __init__(self) -> None:
        self.model_path = os.environ.get(
            "CAPTION_MODEL_PATH", "openbmb/MiniCPM-V-4.6"
        )
        self.processor = None
        self.model = None
        self.lock = threading.Lock()

    def load(self) -> None:
        logger.info("Loading MiniCPM caption model: {}", self.model_path)
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            dtype="auto",
            device_map=os.environ.get("CAPTION_DEVICE_MAP", "auto"),
        )
        self.model.eval()
        logger.success("MiniCPM caption model loaded on {}", self.model.device)

    def caption(self, image_path: str, prompt: str, max_new_tokens: int) -> str:
        if self.model is None or self.processor is None:
            raise RuntimeError("MiniCPM caption model is not loaded")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        with self.lock:
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                max_slice_nums=36,
            ).to(self.model.device, dtype=self.model.dtype)

            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs, max_new_tokens=max_new_tokens
                )

            trimmed = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(
                    inputs.input_ids, generated_ids
                )
            ]
            output = self.processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        return output[0].strip()

    def unload(self) -> None:
        self.model = None
        self.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


engine = MiniCPMCaptionEngine()


@asynccontextmanager
async def lifespan(_: FastAPI):
    engine.load()
    yield
    engine.unload()


app = FastAPI(
    title="MiniCPM Caption Service",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict:
    device = str(engine.model.device) if engine.model is not None else None
    return {
        "status": "ok",
        "service": "minicpm-caption",
        "model": engine.model_path,
        "device": device,
        "ready": engine.model is not None,
    }


@app.post("/v1/caption")
async def caption(
    image: UploadFile = File(...),
    prompt: str = Form("Describe the image in detail."),
    max_new_tokens: int = Form(2000),
) -> dict:
    if not prompt.strip():
        raise HTTPException(400, "prompt cannot be empty")
    if not 1 <= max_new_tokens <= 4096:
        raise HTTPException(400, "max_new_tokens must be between 1 and 4096")

    suffix = Path(image.filename or "image.jpg").suffix or ".jpg"
    raw = await image.read()
    if not raw:
        raise HTTPException(400, "Uploaded image is empty")

    temporary_path = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary:
            temporary.write(raw)
            temporary_path = temporary.name
        result = await run_in_threadpool(
            engine.caption, temporary_path, prompt.strip(), max_new_tokens
        )
    finally:
        if temporary_path:
            Path(temporary_path).unlink(missing_ok=True)

    return {"caption": result, "model": engine.model_path}
