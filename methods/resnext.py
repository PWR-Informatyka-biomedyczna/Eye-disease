from typing import Dict
import torch
from torch import nn
from torchvision.models import resnext101_32x8d, resnext50_32x4d

from methods.base_model import BaseModel


class ResNext101(BaseModel):

    def __init__(self, num_classes: int):
        super(ResNext101, self).__init__(num_classes)
        self._feature_extractor = resnext101_32x8d(True)
        net_fc = self._feature_extractor.fc.in_features
        self._feature_extractor.fc = nn.Linear(net_fc, self.num_classes)
        self.input_size = (224, 224)

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        out = self._feature_extractor(x['input'])
        return out

class ResNext50(BaseModel):

    def __init__(self, num_classes: int):
        super(ResNext50, self).__init__(num_classes)
        self._feature_extractor = resnext50_32x4d(True)
        net_fc = self._feature_extractor.fc.in_features
        self._feature_extractor.fc = nn.Linear(net_fc, self.num_classes)
        self.input_size = (224, 224)

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        out = self._feature_extractor(x['input'])
        return out
