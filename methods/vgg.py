from typing import Dict
import torch
from torch import nn
from torchvision.models import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn

from methods.base_model import BaseModel


class VGG11(BaseModel):

    def __init__(self, num_classes: int):
        super(VGG11, self).__init__(num_classes)
        self._feature_extractor = vgg11_bn(True)
        net_fc = self._feature_extractor.classifier[6].in_features
        self._feature_extractor.classifier[6] = nn.Linear(net_fc, self.num_classes, bias=True)
        self.input_size = (224, 224)
        
    def set_last_layer(self, layer: torch.nn.Linear):
        self._feature_extractor.classifier = layer
    
    def get_last_layer(self):
        return self._feature_extractor.classifier

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        out = self._feature_extractor(x['input'])
        return out
    

class VGG13(BaseModel):

    def __init__(self, num_classes: int):
        super(VGG13, self).__init__(num_classes)
        self._feature_extractor = vgg13_bn(True)
        net_fc = self._feature_extractor.classifier[6].in_features
        self._feature_extractor.classifier[6] = nn.Linear(net_fc, self.num_classes, bias=True)
        self.input_size = (224, 224)
        
    def set_last_layer(self, layer: torch.nn.Linear):
        self._feature_extractor.classifier = layer
    
    def get_last_layer(self):
        return self._feature_extractor.classifier

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        out = self._feature_extractor(x['input'])
        return out

    
class VGG16(BaseModel):

    def __init__(self, num_classes: int):
        super(VGG16, self).__init__(num_classes)
        self._feature_extractor = vgg16_bn(True)
        net_fc = self._feature_extractor.classifier[6].in_features
        self._feature_extractor.classifier[6] = nn.Linear(net_fc, self.num_classes, bias=True)
        self.input_size = (224, 224)

    def set_last_layer(self, layer: torch.nn.Linear):
        self._feature_extractor.classifier = layer

    def get_last_layer(self):
        return self._feature_extractor.classifier

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        out = self._feature_extractor(x['input'])
        return out


class VGG19(BaseModel):

    def __init__(self, num_classes: int):
        super(VGG19, self).__init__(num_classes)
        self._feature_extractor = vgg19_bn(True)
        net_fc = self._feature_extractor.classifier[6].in_features
        self._feature_extractor.classifier[6] = nn.Linear(net_fc, self.num_classes, bias=True)
        self.input_size = (224, 224)
        
    def set_last_layer(self, layer: torch.nn.Linear):
        self._feature_extractor.classifier = layer
    
    def get_last_layer(self):
        return self._feature_extractor.classifier

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        out = self._feature_extractor(x['input'])
        return out
