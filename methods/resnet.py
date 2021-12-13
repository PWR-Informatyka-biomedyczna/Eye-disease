from typing import Dict
import torch
from torch import nn
from torchvision.models import resnet18, resnet50

from methods.base_model import BaseModel


class ResNet18Model(BaseModel):

    def __init__(self, num_classes: int):
        super(ResNet18Model, self).__init__(num_classes)
        self._feature_extractor = resnet18(True)
        net_fc = self._feature_extractor.fc.in_features
        self._feature_extractor.fc = nn.Linear(net_fc, self.num_classes)
        self.input_size = (224, 224)
        self.last_layer = 'fc'

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        out = self._feature_extractor(x['input'])
        return out


class ResNet50Model(BaseModel):

    def __init__(self, num_classes: int):
        super(ResNet50Model, self).__init__(num_classes)
        self._feature_extractor = resnet50(True)
        net_fc = self._feature_extractor.fc.in_features
        self._feature_extractor.fc = nn.Linear(net_fc, self.num_classes)
        self.input_size = (224, 224)
        self.last_layer = 'fc'

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        out = self._feature_extractor(x['input'])
        return out