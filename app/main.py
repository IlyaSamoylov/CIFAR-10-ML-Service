from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.routes import router
from app.config import STATIC_DIR

app = FastAPI(title="CIFAR-10 Classifier (CNN & ResNet18)")

# Монтаж статики и подключение роутера
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.include_router(router)

# Запуск в IDE
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)