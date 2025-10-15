from typing import Dict
import torch.nn as nn
from app.model_registry import MODEL_REGISTRY
from app.config import DEVICE

_models: Dict[str, nn.Module] = {}

def get_model(name: str) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Неизвестная модель: {name}")

    # при первом использовании
    if name not in _models:
        model_class = MODEL_REGISTRY[name]
        model = model_class().load_weights()
        model.to(DEVICE)
        model.eval()
        _models[name] = model

    return _models[name]
