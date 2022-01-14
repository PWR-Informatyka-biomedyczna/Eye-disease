from typing import Dict
import torch
from torch import nn
from torchvision.models import mnasnet0_5,mnasnet1_0, mnasnet1_3

from methods.base_model import BaseModel


class MNASNet0_5(BaseModel):

    def __init__(self, num_classes: int):
        super(MNASNet0_5, self).__init__(num_classes)
        self._feature_extractor = mnasnet0_5(True)
        net_fc = self._feature_extractor.classifier[1].in_features
        self._feature_extractor.classifier[1] = nn.Linear(net_fc, self.num_classes, bias=True)
        self.input_size = (224, 224)
        
    def set_last_layer(self, layer: torch.nn.Linear):
        self._feature_extractor.classifier[1] = layer
    
    def get_last_layer(self):
        return self._feature_extractor.classifier[1]

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        out = self._feature_extractor(x['input'])
        return out

    
class MNASNet1_0(BaseModel):

    def __init__(self, num_classes: int):
        super(MNASNet1_0, self).__init__(num_classes)
        self._feature_extractor = mnasnet1_0(True)
        net_fc = self._feature_extractor.classifier[1].in_features
        self._feature_extractor.classifier[1] = nn.Linear(net_fc, self.num_classes, bias=True)
        self.input_size = (224, 224)

    def set_last_layer(self, layer: torch.nn.Linear):
        self._feature_extractor.classifier[1] = layer

    def get_last_layer(self):
        return self._feature_extractor.classifier[1]

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        out = self._feature_extractor(x['input'])
        return out
    

class MNASNet1_3(BaseModel):

    def __init__(self, num_classes: int):
        super(MNASNet1_3, self).__init__(num_classes)
        self._feature_extractor = mnasnet1_3(False)
        net_fc = self._feature_extractor.classifier[1].in_features
        self._feature_extractor.classifier[1] = nn.Linear(net_fc, self.num_classes, bias=True)
        self.input_size = (224, 224)

    def set_last_layer(self, layer: torch.nn.Linear):
        self._feature_extractor.classifier[1] = layer

    def get_last_layer(self):
        return self._feature_extractor.classifier[1]

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        out = self._feature_extractor(x['input'])
        return out
