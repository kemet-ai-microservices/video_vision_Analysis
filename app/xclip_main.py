"""FastAPI service for XCLIP zero-shot video-frame classification."""

from __future__ import annotations

import json
import os
import threading
from contextlib import asynccontextmanager
from io import BytesIO

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from loguru import logger
from PIL import Image, UnidentifiedImageError
from transformers import XCLIPModel, XCLIPProcessor


class XCLIPEngine:
    def __init__(self) -> None:
        self.model_path = os.environ.get(
            "XCLIP_MODEL_PATH", "microsoft/xclip-base-patch32"
        )
        self.device = os.environ.get("XCLIP_DEVICE", "cuda")
        self.processor: XCLIPProcessor | None = None
        self.model: XCLIPModel | None = None
        self.lock = threading.Lock()

    def load(self) -> None:
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "XCLIP_DEVICE requests CUDA, but PyTorch cannot see an NVIDIA GPU"
            )

        logger.info("Loading XCLIP model {} on {}", self.model_path, self.device)
        self.processor = XCLIPProcessor.from_pretrained(self.model_path)
        self.model = XCLIPModel.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()
        logger.success("XCLIP model loaded")

    def classify(
        self, frames: list[Image.Image], labels: list[str]
    ) -> tuple[str, dict[str, float]]:
        if self.model is None or self.processor is None:
            raise RuntimeError("XCLIP model is not loaded")
        if not frames:
            raise ValueError("At least one frame is required")
        if not labels:
            raise ValueError("At least one classification label is required")

        with self.lock:
            video_inputs = self.processor.image_processor(
                [frames], return_tensors="pt"
            )
            text_inputs = self.processor.tokenizer(
                labels, return_tensors="pt", padding=True
            )
            inputs = {
                **{key: value.to(self.device) for key, value in video_inputs.items()},
                **{key: value.to(self.device) for key, value in text_inputs.items()},
            }
            with torch.inference_mode():
                outputs = self.model(**inputs)
            probabilities = outputs.logits_per_video.softmax(dim=1)[0].cpu()
        best_index = int(probabilities.argmax().item())
        scores = {
            label: round(float(probabilities[index].item()), 4)
            for index, label in enumerate(labels)
        }
        return labels[best_index], scores

    def unload(self) -> None:
        self.model = None
        self.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


engine = XCLIPEngine()


@asynccontextmanager
async def lifespan(_: FastAPI):
    engine.load()
    yield
    engine.unload()


app = FastAPI(
    title="XCLIP Classification Service",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "service": "xclip",
        "model": engine.model_path,
        "device": engine.device,
        "ready": engine.model is not None,
    }


@app.post("/v1/classify")
async def classify(
    frames: list[UploadFile] = File(...),
    labels: str = Form(...),
) -> dict:
    if not 1 <= len(frames) <= 32:
        raise HTTPException(400, "Send between 1 and 32 sampled frames")

    try:
        parsed_labels = json.loads(labels)
        if (
            not isinstance(parsed_labels, list)
            or not parsed_labels
            or not all(isinstance(label, str) and label for label in parsed_labels)
        ):
            raise ValueError
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        raise HTTPException(
            400, "labels must be a non-empty JSON array of strings"
        ) from exc

    decoded_frames: list[Image.Image] = []
    try:
        for upload in frames:
            raw = await upload.read()
            image = Image.open(BytesIO(raw)).convert("RGB")
            decoded_frames.append(image)
    except (UnidentifiedImageError, OSError) as exc:
        for decoded_image in decoded_frames:
            decoded_image.close()
        raise HTTPException(400, f"Invalid frame image: {exc}") from exc

    try:
        label, scores = await run_in_threadpool(
            engine.classify, decoded_frames, parsed_labels
        )
    finally:
        for image in decoded_frames:
            image.close()

    return {"label": label, "scores": scores, "frames_processed": len(frames)}
