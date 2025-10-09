import os
import uuid
from typing import List, Literal

from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import uvicorn

# устройство
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# классы CIFAR-10
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


# директория текущего скрипта (app/main.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# директория моделей внутри app
MODELS_DIR = os.path.join(BASE_DIR, "models")

# пути к файлам моделей
cnn_path = os.path.join(MODELS_DIR, "cnn.pth")
resnet_path = os.path.join(MODELS_DIR, "resnet18.pth")

# CNN
class BetterCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# загрузка моделей
# CNN
cnn = BetterCNN().to(device)
cnn.load_state_dict(torch.load(cnn_path, map_location=device))
cnn.eval()

# ResNET18
resnet = models.resnet18(weights=None)
resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
resnet.maxpool = nn.Identity()
num_ftrs = resnet.fc.in_features
resnet.fc = nn.Linear(num_ftrs, 10)
resnet.load_state_dict(torch.load(resnet_path, map_location=device))
resnet = resnet.to(device)
resnet.eval()

# трансформации
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616))
])


# предсказание
def predict_image(image: Image.Image, model_name: Literal["cnn", "resnet"]) -> dict:
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        if model_name == "cnn":
            outputs = cnn(image_tensor)
        else:
            outputs = resnet(image_tensor)
        _, predicted = torch.max(outputs, 1)
        pred_idx = int(predicted.item())
        pred_label = CIFAR10_CLASSES[pred_idx]
    return {"index": pred_idx, "label": pred_label}

# FastAPI App
app = FastAPI(title="CIFAR-10 Classifier (CNN & ResNet18)")

# 📁 Пути к статике и шаблонам
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")  # пригодится, если будешь сохранять

# Проверяем, что папки существуют (на всякий случай)
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Подключаем статику и шаблоны по абсолютным путям
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


# главная
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

# прредсказание (единый эндпойнт)
@app.post("/predict")
async def predict(
    files: List[UploadFile] = File(...),
    model_name: str = Form(...)
):
    """
    📌 Отправьте одно или несколько изображений и укажите модель ("cnn" или "resnet").
    В ответ получите предсказания с индексом и названием класса.
    """
    results = []
    for file in files:
        image = Image.open(file.file).convert("RGB")
        pred = predict_image(image, model_name)
        results.append({
            "filename": file.filename,
            "model": model_name,
            "predicted_index": pred["index"],
            "predicted_label": pred["label"]
        })
    return {"results": results}

# запуск
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
