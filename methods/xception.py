import abc
import math
import typing
from typing import Dict
import torch
from torch import nn

from methods import BaseModel


class XceptionModel(BaseModel):

    def __init__(self, num_classes: int, fully_connected_layer: nn.Module = None):
        super(XceptionModel, self).__init__(num_classes)
        middle_flow_modules = []
        for i in range(8):
            middle_flow_modules.append(
                ResidualConnection(
                    nn.Sequential(
                        nn.ReLU(),
                        SeparableConvolution(728, 728, (3, 3)),
                        nn.BatchNorm2d(728),
                        nn.ReLU(),
                        SeparableConvolution(728, 728, (3, 3)),
                        nn.BatchNorm2d(728),
                        nn.ReLU(),
                        SeparableConvolution(728, 728, (3, 3)),
                        nn.BatchNorm2d(728)
                    )
                )
            )

        self.model = nn.Sequential(
            nn.Conv2d(3, 32, (3, 3), stride=(2, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResidualConnection(
                nn.Sequential(
                    SeparableConvolution(64, 128, (3, 3)),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    SeparableConvolution(128, 128, (3, 3)),
                    nn.BatchNorm2d(128),
                    nn.MaxPool2d((3, 3), stride=(2, 2), padding=1)
                ),
                nn.Sequential(
                    nn.Conv2d(64, 128, (1, 1), stride=(2, 2)),
                    nn.BatchNorm2d(128)
                )
            ),
            ResidualConnection(
                nn.Sequential(
                    nn.ReLU(),
                    SeparableConvolution(128, 256, (3, 3)),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    SeparableConvolution(256, 256, (3, 3)),
                    nn.BatchNorm2d(256),
                    nn.MaxPool2d((3, 3), stride=(2, 2), padding=1)
                ),
                nn.Sequential(
                    nn.Conv2d(128, 256, (1, 1), stride=(2, 2)),
                    nn.BatchNorm2d(256)
                )
            ),
            ResidualConnection(
                nn.Sequential(
                    nn.ReLU(),
                    SeparableConvolution(256, 728, (3, 3)),
                    nn.BatchNorm2d(728),
                    nn.ReLU(),
                    SeparableConvolution(728, 728, (3, 3)),
                    nn.BatchNorm2d(728),
                    nn.MaxPool2d((3, 3), stride=(2, 2), padding=1)
                ),
                nn.Sequential(
                    nn.Conv2d(256, 728, (1, 1), stride=(2, 2)),
                    nn.BatchNorm2d(728)
                )
            ),

            middle_flow_modules[0],
            middle_flow_modules[1],
            middle_flow_modules[2],
            middle_flow_modules[3],
            middle_flow_modules[4],
            middle_flow_modules[5],
            middle_flow_modules[6],
            middle_flow_modules[7],

            ResidualConnection(
                nn.Sequential(
                    nn.ReLU(),
                    SeparableConvolution(728, 728, (3, 3)),
                    nn.BatchNorm2d(728),
                    nn.ReLU(),
                    SeparableConvolution(728, 1024, (3, 3)),
                    nn.BatchNorm2d(1024),
                    nn.MaxPool2d((3, 3), stride=(2, 2), padding=1)
                ),
                nn.Sequential(
                    nn.Conv2d(728, 1024, (1, 1), stride=(2, 2)),
                    nn.BatchNorm2d(1024)
                )
            ),

            SeparableConvolution(1024, 1536, (3, 3)),
            nn.BatchNorm2d(1536),
            nn.ReLU(),

            SeparableConvolution(1536, 2048, (3, 3)),
            nn.BatchNorm2d(2048),
            nn.ReLU(),

            GlobalAveragePooling()
        )
        if fully_connected_layer is not None:
            self.model.add_module("Fully Connected Layer", module=fully_connected_layer)

        self.model.add_module("Last Layer", module=LogisticRegression(2048, num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    @abc.abstractmethod
    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.model(x['input'])


class SeparableConvolution(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int or tuple, bias=False):
        super(SeparableConvolution, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, groups=in_channels, bias=bias,
                                   padding=1)
        self.pointwise = nn.Conv2d(in_channels, out_channels, (1, 1), bias=bias)

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class ResidualConnection(nn.Module):
    def __init__(self, sequention: nn.Sequential, residual_sequention: nn.Sequential or typing.Callable = lambda x: x):
        super(ResidualConnection, self).__init__()
        self.sequention = sequention
        self.residual_sequention = residual_sequention

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sequention(x) + self.residual_sequention(x)


class GlobalAveragePooling(nn.Module):
    def __init__(self):
        super(GlobalAveragePooling, self).__init__()

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.mean(x, (2, 3))
        return out


class LogisticRegression(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.sigmoid(self.linear(x))
        return out
