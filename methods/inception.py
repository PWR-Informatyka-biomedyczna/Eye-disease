from typing import Dict
import torch
from torch import nn
from torchvision.models import inception_v3

from methods.base_model import BaseModel


class InceptionV3(BaseModel):

    def __init__(self, num_classes: int):
        super(InceptionV3, self).__init__(num_classes)
        self._feature_extractor = inception_v3(True, aux_logits=False)
        net_fc = self._feature_extractor.fc.in_features
        self._feature_extractor.fc = nn.Linear(net_fc, self.num_classes, bias=True)
        self.input_size = (299, 299)
        
    def set_last_layer(self, layer: torch.nn.Linear):
        self._feature_extractor.fc = layer
    
    def get_last_layer(self):
        return self._feature_extractor.fc

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        out = self._feature_extractor(x['input'])
        return out
