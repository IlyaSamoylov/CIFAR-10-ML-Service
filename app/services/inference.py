from PIL import Image
from typing import Dict
import torch
from torchvision import transforms

from .model_loader import get_model
from .preprocess import clean_image
from app.config import DEVICE, CIFAR10_CLASSES

# Трансформации
_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616))
])

def predict_image(image: Image.Image, model_name: str) -> Dict:
    # Базовая очистка
    image = clean_image(image)

    # Трансформация, преобразование в тензор и батч
    image_tensor = _transform(image).unsqueeze(0).to(DEVICE)

    # Получить одну нужную модель из доступных
    model = get_model(model_name)

    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        pred_idx = int(predicted.item())
        pred_label = CIFAR10_CLASSES[pred_idx]

    return {"index": pred_idx, "label": pred_label}
