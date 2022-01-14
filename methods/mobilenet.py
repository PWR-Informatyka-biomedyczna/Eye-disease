from typing import Dict
import torch
from torch import nn
from torchvision.models import mobilenet_v3_large,mobilenet_v3_small


class MobileNetV3Large(BaseModel):

    def __init__(self, num_classes: int):
        super(MobileNetV3Large, self).__init__(num_classes)
        self._feature_extractor = mobilenet_v3_large(True)
        net_fc = self._feature_extractor.classifier[3].in_features
        self._feature_extractor.classifier[3] = nn.Linear(net_fc, self.num_classes, bias=True)
        self.input_size = (224, 224)
        
    def set_last_layer(self, layer: torch.nn.Linear):
        self._feature_extractor.classifier[3] = layer
    
    def get_last_layer(self):
        return self._feature_extractor.classifier[3]

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        out = self._feature_extractor(x['input'])
        return out

    
class MobileNetV3Small(BaseModel):

    def __init__(self, num_classes: int):
        super(MobileNetV3Small, self).__init__(num_classes)
        self._feature_extractor = mobilenet_v3_small(True)
        net_fc = self._feature_extractor.classifier[3].in_features
        self._feature_extractor.classifier[3] = nn.Linear(net_fc, self.num_classes, bias=True)
        self.input_size = (224, 224)

    def set_last_layer(self, layer: torch.nn.Linear):
        self._feature_extractor.classifier[3] = layer
    
    def get_last_layer(self):
        return self._feature_extractor.classifier[3]

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        out = self._feature_extractor(x['input'])
        return out
