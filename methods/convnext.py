from typing import Dict
import torch
import timm
from torch import nn
from torch_model.methods.base_model import BaseModel

class ConvNextTiny(BaseModel):
    def __init__(self, num_classes: int):
        super(ConvNextTiny, self).__init__(num_classes)
        self._feature_extractor = timm.create_model(
            "convnext_tiny", pretrained=True, num_classes=num_classes
        )
        self.input_size = (224, 224)
        self.num_classes = num_classes

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        if isinstance(x, dict):
            out = self._feature_extractor(x["input"])
        else:
            out = self._feature_extractor(x)
        return out

    def set_last_layer(self, layer: torch.nn.Linear):
        self._feature_extractor.head.fc = layer

    def get_last_layer(self):
        return self._feature_extractor.head.fc

    def get_feature_extractor(self) -> torch.nn.Module:
        return self._feature_extractor

    def get_layers_count(self):
        return len(self.get_layers())

    def get_layers(self):
        children = []
        for module in self._feature_extractor.children():
            children.append(module)
        subchildren = self._get_sub_children(children)
        subchildren = self._get_sub_children(subchildren)
        subchildren = self._get_sub_children(subchildren)
        subchildren = self._get_sub_children(subchildren)
        subchildren = self._get_sub_children(subchildren)
        subchildren = self._get_sub_children(subchildren)
        return subchildren

    def get_last_conv_layer(self):
        return self.get_layers()[-15]

    def reset_classifier(self, num_classes, global_pool="avg"):
        self._feature_extractor.reset_classifier(num_classes, global_pool)

    def forward_features(self, x):
        if isinstance(x, dict):
            out = self._feature_extractor.forward_features(x["input"])
        else:
            out = self._feature_extractor.forward_features(x)
        return out

    def forward_head(self, x):
        if isinstance(x, dict):
            out = self._feature_extractor.forward_head(x["input"])
        else:
            out = self._feature_extractor.forward_head(x)
        return out
