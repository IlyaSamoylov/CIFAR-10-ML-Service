import torch
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # путь к корню проекта
STATIC_DIR = os.path.join(BASE_DIR, "static") # путь к static
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates") # путь к корню шаблону
MODELS_DIR = os.path.join(BASE_DIR, "models") # путь к весам моделей

# Метки классов CIFAR-10
CIFAR10_CLASSES = [
    "airplane", "car", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]