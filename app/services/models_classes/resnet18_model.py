import torch
import torch.nn as nn
import os
from torchvision.models.resnet import ResNet, BasicBlock

from app.config import DEVICE, MODELS_DIR

class ResNET18(ResNet):
    def __init__(self):
        # Инициализация оригинального ResNet18 - Стандартный блок = Conv-BN-ReLU, в каждом из 4 слоев по два блока
        super().__init__(block=BasicBlock, layers=[2, 2, 2, 2])

        # Подгон слоев под обученную модель
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.maxpool = nn.Identity()
        num_ftrs = self.fc.in_features
        self.fc = nn.Linear(num_ftrs, 10)

    def load_weights(self):
        path = self._model_path("resnet18")
        print(f"Веса для ResNET18 загружены из директории {path}")
        if os.path.exists(path):
            state_dict = torch.load(path, map_location=DEVICE)
            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                print("Missing:", missing) # если в обученной модели веса, которых не достает в инференсной
                print("Unexpected:", unexpected) # если в инференсной модели веса, которых не хватает в обученной
        else:
            print("Веса не найдены по пути:", path)

        self.to(DEVICE)
        self.eval()
        return self

    def _model_path(self, name: str) -> str:
        return os.path.join(MODELS_DIR, f"{name}.pth")
