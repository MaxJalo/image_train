# (debug)
import logging
import time
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from core.config import settings
from services import model_loader

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/debug", tags=["debug"])


@router.get("/model-status")
async def model_status(request: Request) -> Dict[str, Any]:
    app = request.app
    status = getattr(app.state, "model_status", {})
    versions = {"python": __import__("sys").version.split()[0]}
    try:
        import torch

        versions["torch"] = torch.__version__
    except Exception:
        versions["torch"] = None
    try:
        import ultralytics

        versions["ultralytics"] = ultralytics.__version__
    except Exception:
        versions["ultralytics"] = None
    except Exception:
        versions["tensorflow"] = None

    return {
        "models": status,
        "versions": versions,
        "model_paths": {"model1": settings.model1_path, "model2": settings.model2_path},
    }


class TestPhotoResponse(BaseModel):
    model: str
    inference_time: float | None
    error: str | None
    output_summary: str | None


@router.post("/test-photo")
async def run_test_photo(photo_path: str):
    p = photo_path
    if not p:
        raise HTTPException(status_code=400, detail="photo_path required")
    if not __import__("pathlib").Path(p).exists():
        raise HTTPException(status_code=404, detail=f"Photo not found: {p}")

    results = {}
    # Try to load models from cache
    cached = model_loader.get_cached_models()
    for key, model_path in [("model1", settings.model1_path), ("model2", settings.model2_path)]:
        info = {"loaded": False, "error": None, "inference_time": None}
        try:
            if model_path in cached:
                model = cached[model_path]
            else:
                model = model_loader.load_model(model_path)

            t0 = time.time()
            # Try running inference depending on model capabilities
            try:
                model(__import__("PIL").Image.open(p).convert("RGB"))
                t1 = time.time()
                info["inference_time"] = round(t1 - t0, 4)
                info["loaded"] = True
            except Exception:
                try:
                    model.predict(p)
                    t1 = time.time()
                    info["inference_time"] = round(t1 - t0, 4)
                    info["loaded"] = True
                except Exception as e:
                    info["error"] = str(e)
        except Exception as e:
            info["error"] = str(e)
        results[key] = info

    return results


@router.post("/test-model1")
async def test_model1(photo_path: str):
    p = photo_path
    if not p:
        raise HTTPException(status_code=400, detail="photo_path required")

    p_obj = __import__("pathlib").Path(p)
    if not p_obj.exists():
        raise HTTPException(status_code=404, detail=f"Photo not found: {p}")

    try:
        # Load the image
        import torch
        from PIL import Image

        logger.info(f"Testing Model-1 with photo: {p}")
        image = Image.open(p).convert("RGB")
        logger.info(f"Image loaded: size={image.size}, mode={image.mode}")

        # Get or load model
        cached = model_loader.get_cached_models()
        if settings.model1_path in cached:
            model = cached[settings.model1_path]
            logger.debug("Model loaded from cache")
        else:
            logger.debug("Loading model from disk")
            model = model_loader.load_model(settings.model1_path)

        # Get device from model
        device = next(model.parameters()).device
        logger.debug(f"Model device: {device}")

        # Prepare image as tensor using same preprocessing as train.py
        logger.debug("Converting image to tensor")
        from torchvision import transforms

        transform = transforms.Compose(
            [
                transforms.Resize((300, 300)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        tensor = transform(image).unsqueeze(0)
        tensor = tensor.to(device)
        logger.debug(f"Tensor shape: {tensor.shape}, device: {device}")

        # Run inference
        logger.info("Running inference")
        t0 = time.time()
        with torch.no_grad():
            output = model(tensor)
        t1 = time.time()
        inference_time = round(t1 - t0, 4)

        logger.info(f"Inference complete: output type={type(output)},\
             shape={output.shape if hasattr(output, 'shape') else 'N/A'}")

        # Parse output
        result = {
            "photo_path": p,
            "model": "MobileNet v3",
            "inference_time": inference_time,
            "output_type": str(type(output).__name__),
            "output_shape": str(output.shape) if hasattr(output, "shape") else None,
            "raw_output": str(output),
            "error": None,
        }

        # Try to interpret output
        if isinstance(output, torch.Tensor):
            if output.shape[-1] == 2:  # Binary classification
                probabilities = torch.softmax(output, dim=-1)[0]
                result["is_valid"] = bool(probabilities[1] > probabilities[0])
                result["confidence"] = float(probabilities[1].item())
                result["interpretation"] = "Binary classification (2 classes)"
            else:  # Single output or other
                result["interpretation"] = f"Output shape: {output.shape}"

        return result

    except Exception as e:
        logger.error(f"Error testing Model-1: {type(e).__name__}: {str(e)}")
        return {
            "photo_path": p,
            "model": "MobileNet v3",
            "error": str(e),
            "error_type": type(e).__name__,
        }
