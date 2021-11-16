from typing import Dict
import torch
from torch import nn
from torchvision.models import resnet18

from methods.base_model import BaseModel


class ResNetModel(BaseModel):

    def __init__(self, num_classes: int):
        super(ResNetModel, self).__init__(num_classes)
        self._feature_extractor = resnet18(True)
        net_fc = self._feature_extractor.fc.in_features
        self._feature_extractor.fc = nn.Linear(net_fc, self.num_classes)

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        out = self._feature_extractor(x['input'])
        return out
