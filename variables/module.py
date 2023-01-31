import math
import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict
from functools import wraps
from pathlib import Path
from typing import (TYPE_CHECKING, Callable, Collection, Container, Iterable,
                    List, Literal, Optional, Tuple, Type, TypedDict, Union)

import torch as t
from torch import nn
from torchvision.models.resnet import (resnet18, resnet34, resnet50, resnet101,
                                       resnet152)

from perturb.constants import DEVICE
from perturb.types_ import ParameterOrTensor
from perturb.utils import get_parameters
from perturb.variables.weight_init import FlattenedKaimingHe

if TYPE_CHECKING:
    from perturb.variables.weight_init import WeightInitializer


def dot_parameters(
    p0s: Iterable[ParameterOrTensor], p1s: Iterable[ParameterOrTensor]
) -> t.Tensor:
    """Compute the dot product of two sets of parameters."""
    value = t.zeros(1, device=DEVICE)

    for p0, p1 in zip(p0s, p1s):
        if p0.shape != p1.shape:
            raise ValueError("Parameters have different shapes")

        value += t.sum(p0 * p1)

    return value


def distance_bw_parameters(
    p0s: Iterable[ParameterOrTensor], p1s: Iterable[ParameterOrTensor], p="fro"
) -> t.Tensor:
    """Compute the distance between two sets of parameters."""
    value = t.zeros(1, device=DEVICE)

    for p0, p1 in zip(p0s, p1s):
        if p0.shape != p1.shape:
            raise ValueError("Parameters have different shapes")

        value += t.norm(p0 - p1, p=p) ** 2

    return value.sqrt()


def parameters_norm(ps: Iterable[ParameterOrTensor], p="fro") -> t.Tensor:
    """Compute the norm of a set of parameters."""
    value = t.zeros(1, device=DEVICE)

    for param in ps:
        value += t.norm(param, p=p) ** 2

    return value.sqrt()


def cosine_similarity_bw_parameters(
    p0s: Iterable[ParameterOrTensor], p1s: Iterable[ParameterOrTensor]
) -> t.Tensor:
    """Compute the cosine similarity between two sets of parameters."""
    return dot_parameters(p0s, p1s) / (parameters_norm(p0s) * parameters_norm(p1s))


def extract_parameters(
    model: Union["ExtendedModule", Iterable[ParameterOrTensor], "WeightInitializer"]
) -> Iterable[ParameterOrTensor]:
    """Extract the parameters of a model, weight initalizier, etc."""
    if hasattr(model, "model"):  # Trial
        return model.model.parameters()  # type: ignore
    elif hasattr(model, "initial_weights"):  # WeightInitializer
        if model.initial_weights is None:  # type: ignore
            raise ValueError("Model has not been initialized")
        return model.initial_weights  # type: ignore
    elif hasattr(model, "parameters"):
        return model.parameters()  # type: ignore

    return model  # type: ignore


ParameterExtractable = Union[
    "ExtendedModule", Iterable[ParameterOrTensor], "WeightInitializer"
]


def wrap_extractor(fn: Callable[..., t.Tensor]):
    """Wrap a function that takes parameters into a function that takes models."""

    @wraps(fn)
    def wrapper(
        self,
        other: ParameterExtractable,
        *args,
        **kwargs,
    ) -> t.Tensor:
        return fn(self, extract_parameters(other), *args, **kwargs)

    return wrapper


class ExtendedModule(nn.Module):
    """
    An extended module with an associated weight initializer object, hyperparams dict, and 
    some methods for computing distances between models.
    """
    weight_initializer_cls = FlattenedKaimingHe

    def __init__(
        self,
        hyperparams: Optional[dict] = None,
        weight_initializer: Optional["WeightInitializer"] = None,
        **kwargs,
    ):
        self.hyperparams = hyperparams or {}
        self.weight_initializer = weight_initializer or self.weight_initializer_cls()

        super().__init__(**kwargs)

        self.weight_initializer.apply(self)

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @wrap_extractor
    def dot(self, other: Iterable[ParameterOrTensor]) -> t.Tensor:
        if isinstance(other, ExtendedModule):
            other = other.parameters()

        return dot_parameters(self.parameters(), other)

    def __matmul__(self, other: ParameterExtractable) -> t.Tensor:
        return self.dot(other)

    @wrap_extractor
    def lp_distance(
        self,
        other: Iterable[ParameterOrTensor],
        p: str = "fro",
    ) -> t.Tensor:
        return distance_bw_parameters(self.parameters(), other, p=p)

    def norm(self) -> t.Tensor:
        return parameters_norm(self.parameters())

    @wrap_extractor
    def cosine_similarity(self, other: Iterable[ParameterOrTensor]) -> t.Tensor:
        return cosine_similarity_bw_parameters(self.parameters(), other)

    @property
    def device(self) -> t.device:
        return next(self.parameters()).device

    @property
    def shape(self) -> OrderedDict:
        return OrderedDict((name, p.shape) for name, p in self.state_dict().items())


def extend_module(module: Type[nn.Module]) -> ExtendedModule:
    """Extends an optimizer to include a hyperparams property."""

    if issubclass(module, ExtendedModule):
        # Insert ExtendedOptimizer into the class hierarchy
        mro = module.mro()
        mro.insert(mro.index(nn.Module), ExtendedModule)

        dict_ = module.__dict__.copy()

        return type(optimizer.__name__, tuple(mro), dict_)  # type: ignore
    else:
        raise TypeError("module must be a subclass of nn.Module")
