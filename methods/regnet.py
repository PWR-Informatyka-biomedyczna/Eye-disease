from typing import Dict
import torch
from torch import nn
from torchvision.models import regnet_y_8gf, regnet_y_400mf, regnet_y_800mf, regnet_x_800mf,regnet_x_400mf, regnet_y_1_6gf, regnet_x_1_6gf, regnet_y_16gf, regnet_y_32gf
from torchvision.models import regnet_y_3_2gf, regnet_x_3_2gf, regnet_x_8gf

from methods.base_model import BaseModel


class RegNetY8gf(BaseModel):

    def __init__(self, num_classes: int):
        super(RegNetY8gf, self).__init__(num_classes)
        self._feature_extractor = regnet_y_8gf(True)
        net_fc = self._feature_extractor.fc.in_features
        self._feature_extractor.fc = nn.Linear(net_fc, self.num_classes, bias=True)
        self.input_size = (224, 224)

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        out = self._feature_extractor(x['input'])
        return out
    
    def set_last_layer(self, layer: torch.nn.Linear):
        self._feature_extractor.fc = layer
    
    def get_last_layer(self):
        return self._feature_extractor.fc

    
class RegNetX8gf(BaseModel):

    def __init__(self, num_classes: int):
        super(RegNetX8gf, self).__init__(num_classes)
        self._feature_extractor = regnet_x_8gf(True)
        net_fc = self._feature_extractor.fc.in_features
        self._feature_extractor.fc = nn.Linear(net_fc, self.num_classes, bias=True)
        self.input_size = (224, 224)

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        out = self._feature_extractor(x['input'])
        return out
    
    def set_last_layer(self, layer: torch.nn.Linear):
        self._feature_extractor.fc = layer
    
    def get_last_layer(self):
        return self._feature_extractor.fc
    
    
class RegNetY400MF(BaseModel):

    def __init__(self, num_classes: int):
        super(RegNetY400MF, self).__init__(num_classes)
        self._feature_extractor = regnet_y_400mf(True)
        net_fc = self._feature_extractor.fc.in_features
        self._feature_extractor.fc = nn.Linear(net_fc, self.num_classes, bias=True)
        self.input_size = (224, 224)

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        out = self._feature_extractor(x['input'])
        return out
    
    def set_last_layer(self, layer: torch.nn.Linear):
        self._feature_extractor.fc = layer
    
    def get_last_layer(self):
        return self._feature_extractor.fc


class RegNetX400MF(BaseModel):

    def __init__(self, num_classes: int):
        super(RegNetX400MF, self).__init__(num_classes)
        self._feature_extractor = regnet_x_400mf(True)
        net_fc = self._feature_extractor.fc.in_features
        self._feature_extractor.fc = nn.Linear(net_fc, self.num_classes, bias=True)
        self.input_size = (224, 224)

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        out = self._feature_extractor(x['input'])
        return out
    
    def set_last_layer(self, layer: torch.nn.Linear):
        self._feature_extractor.fc = layer
    
    def get_last_layer(self):
        return self._feature_extractor.fc


class RegNetY800MF(BaseModel):

    def __init__(self, num_classes: int):
        super(RegNetY800MF, self).__init__(num_classes)
        self._feature_extractor = regnet_y_800mf(True)
        net_fc = self._feature_extractor.fc.in_features
        self._feature_extractor.fc = nn.Linear(net_fc, self.num_classes, bias=True)
        self.input_size = (224, 224)

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        out = self._feature_extractor(x['input'])
        return out
    
    def set_last_layer(self, layer: torch.nn.Linear):
        self._feature_extractor.fc = layer
    
    def get_last_layer(self):
        return self._feature_extractor.fc


class RegNetX800MF(BaseModel):

    def __init__(self, num_classes: int):
        super(RegNetX800MF, self).__init__(num_classes)
        self._feature_extractor = regnet_x_800mf(True)
        net_fc = self._feature_extractor.fc.in_features
        self._feature_extractor.fc = nn.Linear(net_fc, self.num_classes, bias=True)
        self.input_size = (224, 224)

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        out = self._feature_extractor(x['input'])
        return out
    
    def set_last_layer(self, layer: torch.nn.Linear):
        self._feature_extractor.fc = layer
    
    def get_last_layer(self):
        return self._feature_extractor.fc

    
class RegNetY1_6gf(BaseModel):

    def __init__(self, num_classes: int):
        super(RegNetY1_6gf, self).__init__(num_classes)
        self._feature_extractor = regnet_y_1_6gf(True)
        net_fc = self._feature_extractor.fc.in_features
        self._feature_extractor.fc = nn.Linear(net_fc, self.num_classes, bias=True)
        self.input_size = (224, 224)

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        out = self._feature_extractor(x['input'])
        return out
    
    def set_last_layer(self, layer: torch.nn.Linear):
        self._feature_extractor.fc = layer
    
    def get_last_layer(self):
        return self._feature_extractor.fc

    
class RegNetX1_6gf(BaseModel):

    def __init__(self, num_classes: int):
        super(RegNetX1_6gf, self).__init__(num_classes)
        self._feature_extractor = regnet_x_1_6gf(True)
        net_fc = self._feature_extractor.fc.in_features
        self._feature_extractor.fc = nn.Linear(net_fc, self.num_classes, bias=True)
        self.input_size = (224, 224)

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        out = self._feature_extractor(x['input'])
        return out
    
    def set_last_layer(self, layer: torch.nn.Linear):
        self._feature_extractor.fc = layer
    
    def get_last_layer(self):
        return self._feature_extractor.fc
    
    
class RegNetY3_2gf(BaseModel):

    def __init__(self, num_classes: int):
        super(RegNetY3_2gf, self).__init__(num_classes)
        self._feature_extractor = regnet_y_3_2gf(True)
        net_fc = self._feature_extractor.fc.in_features
        self._feature_extractor.fc = nn.Linear(net_fc, self.num_classes, bias=True)
        self.input_size = (224, 224)

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        out = self._feature_extractor(x['input'])
        return out
    
    def set_last_layer(self, layer: torch.nn.Linear):
        self._feature_extractor.fc = layer
    
    def get_last_layer(self):
        return self._feature_extractor.fc
    

class RegNetX3_2gf(BaseModel):

    def __init__(self, num_classes: int):
        super(RegNetX3_2gf, self).__init__(num_classes)
        self._feature_extractor = regnet_x_3_2gf(True)
        net_fc = self._feature_extractor.fc.in_features
        self._feature_extractor.fc = nn.Linear(net_fc, self.num_classes, bias=True)
        self.input_size = (224, 224)

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        out = self._feature_extractor(x['input'])
        return out
    
    def set_last_layer(self, layer: torch.nn.Linear):
        self._feature_extractor.fc = layer
    
    def get_last_layer(self):
        return self._feature_extractor.fc
    
    
class RegNetY16gf(BaseModel):

    def __init__(self, num_classes: int):
        super(RegNetY16gf, self).__init__(num_classes)
        self._feature_extractor = regnet_y_16gf(True)
        net_fc = self._feature_extractor.fc.in_features
        self._feature_extractor.fc = nn.Linear(net_fc, self.num_classes, bias=True)
        self.input_size = (224, 224)

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        out = self._feature_extractor(x['input'])
        return out

    def set_last_layer(self, layer: torch.nn.Linear):
        self._feature_extractor.fc = layer

    def get_last_layer(self):
        return self._feature_extractor.fc


class RegNetY32gf(BaseModel):

    def __init__(self, num_classes: int):
        super(RegNetY32gf, self).__init__(num_classes)
        self._feature_extractor = regnet_y_32gf(True)
        net_fc = self._feature_extractor.fc.in_features
        self._feature_extractor.fc = nn.Linear(net_fc, self.num_classes, bias=True)
        self.input_size = (224, 224)

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        out = self._feature_extractor(x['input'])
        return out

    def set_last_layer(self, layer: torch.nn.Linear):
        self._feature_extractor.fc = layer

    def get_last_layer(self):
        return self._feature_extractor.fc
