import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from .routes import router

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
# UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
# os.makedirs(UPLOADS_DIR, exist_ok=True)

app = FastAPI(title="CIFAR-10 Classifier (CNN & ResNet18)")

# Монтаж статики и подключение роутера
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.include_router(router)

# Запуск в IDE
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)