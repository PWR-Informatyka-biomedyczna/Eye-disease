from typing import Dict
import torch
from torch import nn
from torchvision.models import densenet161

from methods.base_model import BaseModel


class DenseNet161(BaseModel):

    def __init__(self, num_classes: int):
        super(DenseNet161, self).__init__(num_classes)
        self._feature_extractor = densenet161(True)
        net_fc = self._feature_extractor.classifier.in_features
        self._feature_extractor.classifier = nn.Linear(net_fc, self.num_classes)
        self.input_size = (224, 244)
        
    def set_last_layer(self, layer: torch.nn.Linear):
        self._feature_extractor.classifier = layer
    
    def get_last_layer(self):
        return self._feature_extractor.classifier

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        out = self._feature_extractor(x['input'])
        return out
