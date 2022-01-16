from typing import Dict
import torch
from torch import nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101

from methods.base_model import BaseModel


class ResNet18Model(BaseModel):

    def __init__(self, num_classes: int):
        super(ResNet18Model, self).__init__(num_classes)
        self._feature_extractor = resnet18(True)
        net_fc = self._feature_extractor.fc.in_features
        self._feature_extractor.fc = nn.Linear(net_fc, self.num_classes)
        self.input_size = (224, 224)
        self.last_layer = 'fc'
        
    def set_last_layer(self, layer: torch.nn.Linear):
        self._feature_extractor.fc = layer
    
    def get_last_layer(self):
        return self._feature_extractor.fc

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        out = self._feature_extractor(x['input'])
        return out


class ResNet34Model(BaseModel):

    def __init__(self, num_classes: int):
        super(ResNet34Model, self).__init__(num_classes)
        self._feature_extractor = resnet34(True)
        net_fc = self._feature_extractor.fc.in_features
        self._feature_extractor.fc = nn.Linear(net_fc, self.num_classes)
        self.input_size = (224, 224)

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        out = self._feature_extractor(x['input'])
        return out
    
    def set_last_layer(self, layer: torch.nn.Linear):
        self._feature_extractor.fc = layer
    
    def get_last_layer(self):
        return self._feature_extractor.fc


class ResNet50Model(BaseModel):

    def __init__(self, num_classes: int):
        super(ResNet50Model, self).__init__(num_classes)
        self._feature_extractor = resnet50(True)
        net_fc = self._feature_extractor.fc.in_features
        self._feature_extractor.fc = nn.Linear(net_fc, self.num_classes)
        self.input_size = (224, 224)
        
    def set_last_layer(self, layer: torch.nn.Linear):
        self._feature_extractor.fc = layer
    
    def get_last_layer(self):
        return self._feature_extractor.fc
    
    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        out = self._feature_extractor(x['input'])
        return out


class ResNet101Model(BaseModel):

    def __init__(self, num_classes: int):
        super(ResNet101Model, self).__init__(num_classes)
        self._feature_extractor = resnet101(True)
        net_fc = self._feature_extractor.fc.in_features
        self._feature_extractor.fc = nn.Linear(net_fc, self.num_classes)
        self.input_size = (224, 224)

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        out = self._feature_extractor(x['input'])
        return out
    
    def set_last_layer(self, layer: torch.nn.Linear):
        self._feature_extractor.fc = layer
    
    def get_last_layer(self):
        return self._feature_extractor.fc