from typing import Dict
import torch
from torch import nn
from torchvision.models import efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b4

from methods.base_model import BaseModel


class EfficientNetB0(BaseModel):

    def __init__(self, num_classes: int):
        super(EfficientNetB0, self).__init__(num_classes)
        self._feature_extractor = efficientnet_b0(True)
        net_fc = self._feature_extractor.classifier[1].in_features
        self._feature_extractor.classifier[1] = nn.Linear(net_fc, self.num_classes)
        self.input_size = (224, 224)
        self.last_layer = 'classifier[1]'

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        out = self._feature_extractor(x['input'])
        return out

class EfficientNetB1(BaseModel):

    def __init__(self, num_classes: int):
        super(EfficientNetB1, self).__init__(num_classes)
        self._feature_extractor = efficientnet_b1(True)
        net_fc = self._feature_extractor.classifier[1].in_features
        self._feature_extractor.classifier[1] = nn.Linear(net_fc, self.num_classes)
        self.input_size = (240, 240)
        self.last_layer = 'classifier[1]'

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        out = self._feature_extractor(x['input'])
        return out

class EfficientNetB2(BaseModel):

    def __init__(self, num_classes: int):
        super(EfficientNetB2, self).__init__(num_classes)
        self._feature_extractor = efficientnet_b2(True)
        net_fc = self._feature_extractor.classifier[1].in_features
        self._feature_extractor.classifier[1] = nn.Linear(net_fc, self.num_classes)
        self.input_size = (260, 260)
        self.last_layer = 'classifier[1]'

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        out = self._feature_extractor(x['input'])
        return out

class EfficientNetB3(BaseModel):

    def __init__(self, num_classes: int):
        super(EfficientNetB3, self).__init__(num_classes)
        self._feature_extractor = efficientnet_b3(True)
        net_fc = self._feature_extractor.classifier[1].in_features
        self._feature_extractor.classifier[1] = nn.Linear(net_fc, self.num_classes)
        self.input_size = (300, 300)
        self.last_layer = 'classifier[1]'

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        out = self._feature_extractor(x['input'])
        return out

class EfficientNetB4(BaseModel):

    def __init__(self, num_classes: int):
        super(EfficientNetB4, self).__init__(num_classes)
        self._feature_extractor = efficientnet_b4(True)
        net_fc = self._feature_extractor.classifier[1].in_features
        self._feature_extractor.classifier[1] = nn.Linear(net_fc, self.num_classes)
        self.input_size = (380, 380)
        self.last_layer = 'classifier[1]'

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        out = self._feature_extractor(x['input'])
        return out
