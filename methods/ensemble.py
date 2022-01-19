from typing import Dict
import torch
from torch import nn
from torchvision.models import regnet_y_3_2gf

from methods.base_model import BaseModel


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


class MLP(nn.Module):
    def __init__(self, in_features: int, num_classes, depth: int = 1):
        super(MLP, self).__init__()
        self.layers = nn.Sequential()
        features = in_features
        for _ in range(depth + 1):
            self.layers.add_module(nn.Linear(features, features / 2))
            features = features / 2
        self.layers.add_module(nn.Linear(features, num_classes))
        
    def forward(self, x):
        return self.layers(x)


class RegNetY3_2gfEnsemble(BaseModel):

    def __init__(self, num_classes: int, n_classifiers: int, depth: int = 1):
        super(RegNetY3_2gfEnsemble, self).__init__(num_classes)
        self.n_classifiers = n_classifiers
        self._feature_extractor = regnet_y_3_2gf(True)
        in_features = self._feature_extractor.fc.in_features
        self._feature_extractor.fc = Identity()
        self.fc_nets = []
        for _ in range(n_classifiers):
            self.fc_nets.append(MLP(in_features, num_classes, depth))
        self.input_size = (224, 224)

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        features = self._feature_extractor(x['input'])
        logits = 0
        for fc in self.fc_nets:
            logits += fc(features)
        
        return out
    
    def set_last_layer(self, layer: torch.nn.Linear):
        self._feature_extractor.fc = layer
    
    def get_last_layer(self):
        return self._feature_extractor.fc

print(RegNetY3_2gfEnsemble(4))