import abc
import torch
from torch import nn


class BaseModel(nn.Module):

    def __init__(self, num_classes: int):
        super(BaseModel, self).__init__()
        self.num_classes = num_classes

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
