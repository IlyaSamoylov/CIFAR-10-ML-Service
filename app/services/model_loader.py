from typing import Dict
import torch.nn as nn

from .models_classes.cnn_model import BetterCNN
from .models_classes.resnet18_model import ResNET18

# Ленивое создание и кэш моделей, чтобы не загружать веса при каждой загрузке файла
_models: Dict[str, nn.Module] = {}

def get_models() -> Dict[str, nn.Module]:
    global _models
    if not _models:
        cnn_model = BetterCNN().load_weights()
        resnet_model = ResNET18().load_weights()
        _models["cnn"] = cnn_model
        _models["resnet"] = resnet_model
    return _models
