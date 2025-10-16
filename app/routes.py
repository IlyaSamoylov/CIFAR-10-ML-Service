from typing import List
from fastapi import APIRouter, Request, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image

from .services.inference import predict_image
from app.config import TEMPLATES_DIR
from app.model_registry import MODEL_REGISTRY

templates = Jinja2Templates(directory=TEMPLATES_DIR)

router = APIRouter()

@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

@router.post("/predict")
async def predict(files: List[UploadFile] = File(...), model_name: str = Form(...)):
    if model_name not in MODEL_REGISTRY:
        raise HTTPException(status_code=400, detail="Неизвестное имя модели")

    results = []
    for file in files:
        try:
            image = Image.open(file.file).convert("RGB")
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": "неправильный файл"
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
