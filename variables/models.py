from typing import Literal, Tuple, Union

import torch as t
from torch import nn
from torchvision.models.resnet import (resnet18, resnet34, resnet50, resnet101,
                                       resnet152)

from perturb.variables.module import ExtendedModule


class FCN(ExtendedModule):
    """A simple FCN classifier with a variable number of hidden layers."""

    n_hidden: Tuple[int, ...]
    n_units: Tuple[int, ...]

    def __init__(self, n_hidden: Union[int, Tuple[int]], n_in=784, n_out=10, **kwargs):
        if isinstance(n_hidden, int):
            n_hidden = (n_hidden,)

        hyperparams = {"n_hidden": n_hidden}

        super().__init__(hyperparams=hyperparams, **kwargs)  # type: ignore

        self.n_hidden = n_hidden
        self.n_units = n_units = (n_in, *n_hidden, n_out)

        fc_layers = [
            nn.Linear(n_units[i], n_units[i + 1]) for i in range(len(n_units) - 1)
        ]
        activations = [nn.ReLU() for _ in range(len(n_units) - 1)]
        hidden_layers = [item for pair in zip(fc_layers, activations) for item in pair]

        self.model = nn.Sequential(
            nn.Flatten(),
            *hidden_layers[:-1],  # Skip the last ReLU
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.model(x)


class Lenet5(ExtendedModule):
    def __init__(self, **kwargs):
        super().__init__(hyperparams={}, **kwargs)  # type: ignore

        self.model = nn.Sequential(
            nn.Conv2d(1, 6, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.model(x)


class ResNet(ExtendedModule):
    def __init__(
        self,
        n_layers: Literal[18, 34, 50, 101, 152],
        **kwargs,
    ):
        self.n_layers = n_layers
        super().__init__(hyperparams={"n_layers": n_layers}, **kwargs)

        if n_layers == 18:
            self.model = resnet18(weights=None)
        elif n_layers == 34:
            self.model = resnet34(weights=None)
        elif n_layers == 50:
            self.model = resnet50(weights=None)
        elif n_layers == 101:
            self.model = resnet101(weights=None)
        elif n_layers == 152:
            self.model = resnet152(weights=None)

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.model(x)
