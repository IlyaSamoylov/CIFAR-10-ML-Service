import os
from typing import List
from fastapi import APIRouter, Request, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image

from .services.inference import predict_image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")
# UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
# os.makedirs(UPLOADS_DIR, exist_ok=True)

templates = Jinja2Templates(directory=TEMPLATES_DIR)

router = APIRouter()

@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

@router.post("/predict")
async def predict(request: Request, files: List[UploadFile] = File(...), model_name: str = Form(...)):
    if model_name not in ("cnn", "resnet"):
        raise HTTPException(status_code=400, detail="Model must be 'cnn' or 'resnet'")

    results = []
    for file in files:
        try:
            image = Image.open(file.file).convert("RGB")
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": "invalid image"
            })
            continue

        pred = predict_image(image, model_name)
        results.append({
            "filename": file.filename,
            "model": model_name,
            "predicted_index": pred["index"],
            "predicted_label": pred["label"]
        })

    return {"results": results}
