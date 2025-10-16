from typing import Dict, Type
from torch.nn import Module

from app.services.models_classes.cnn_model import BetterCNN
from app.services.models_classes.resnet18_model import ResNET18

# Поддерживаемые модели = название модели: класс
MODEL_REGISTRY: Dict[str, Type[Module]] = {
    "cnn": BetterCNN,
    "resnet": ResNET18,
}