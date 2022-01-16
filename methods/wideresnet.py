from typing import Dict
import torch
from torch import nn
from torchvision.models import wide_resnet50_2, wide_resnet101_2

from methods.base_model import BaseModel

class WideResNet50(BaseModel):

    def __init__(self, num_classes: int):
        super(WideResNet50, self).__init__(num_classes)
        self._feature_extractor = wide_resnet50_2(True)
        net_fc = self._feature_extractor.fc.in_features
        self._feature_extractor.fc = nn.Linear(net_fc, self.num_classes, bias=True)
        self.input_size = (224, 224)
        
    def set_last_layer(self, layer: torch.nn.Linear):
        self._feature_extractor.classifier = layer
    
    def get_last_layer(self):
        return self._feature_extractor.classifier

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        out = self._feature_extractor(x['input'])
        return out

    
class WideResNet101(BaseModel):

    def __init__(self, num_classes: int):
        super(WideResNet101, self).__init__(num_classes)
        self._feature_extractor = wide_resnet101_2(True)
        net_fc = self._feature_extractor.fc.in_features
        self._feature_extractor.fc = nn.Linear(net_fc, self.num_classes, bias=True)
        self.input_size = (224, 224)

    def set_last_layer(self, layer: torch.nn.Linear):
        self._feature_extractor.classifier = layer

    def get_last_layer(self):
        return self._feature_extractor.classifier

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        out = self._feature_extractor(x['input'])
        return out
