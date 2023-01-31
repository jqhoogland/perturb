import hashlib
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import (Any, Callable, Collection, Container, Dict, Generic,
                    Iterable, List, Optional, Protocol, Sequence, Tuple, Type,
                    TypeVar, Union)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
# Reexport
# from tqdm import tqdm, trange
from tqdm.notebook import tqdm, trange

from perturb.types_ import OptionalTuple, S, T


def setup() -> str:
    device = "cuda" if t.cuda.is_available() else "cpu"

    mpl.rcParams["text.usetex"] = True
    # mpl.rcParams[
    #     "text.latex.preamble"
    # ] = [r"\usepackage{amsmath}"]  # for \text command
    mpl.rcParams["figure.dpi"] = 300
    plt.style.use("ggplot")
    t.device(device)

    # wandb.login()

    return device


def stable_hash(x: Any) -> str:
    return hashlib.sha256(str(x).encode("utf-8")).hexdigest()[:32]


def to_tuple(x: OptionalTuple[T], default: S):
    if isinstance(x, tuple):
        return x
    elif isinstance(x, list):
        return tuple(x)

    return (x, default)


def dict_to_str(d: Dict[str, Any]) -> str:
    return ", ".join(f"{k}={v}" for k, v in d.items())


def get_parameters(model: nn.Module) -> t.Tensor:
    """Get a flattened tensor of all parameters in a model."""
    warnings.warn("parameters_vector is expensive for deep models.", DeprecationWarning)
    return t.cat([p.view(-1) for p in model.parameters()])


def tensor_map(f: Callable[..., float], *args):
    """Map a function that returns a float over a (collection of) iterables,
    then wrap the result in a tensor."""
    return t.tensor([f(*arg) for arg in zip(*args)])


def calculate_n_params(widths: Tuple[int, ...]) -> int:
    return (
        sum((widths[i] + 1) * widths[i + 1] for i in range(len(widths) - 1))
        + widths[-1]
    )


def divide_params(
    n_params: int, n_layers: int, h_initial: int, h_final: int
) -> Tuple[int, ...]:
    """
    Divide parameters into n_layers layers and returns the number of units
    in each layer. Assumes each layer has c times as many parameters as the
    next layer, where c is a constant.
    """

    def _calculate_widths(c: float) -> Tuple[int, ...]:
        return (h_initial, *(int((c**i) * h_final) for i in range(n_layers, -1, -1)))

    def _calculate_n_params(c: float):
        return calculate_n_params(_calculate_widths(c))

    # Find the value of c that gives the closest number of parameters
    c = scipy.optimize.brentq(lambda c: _calculate_n_params(c) - n_params, 0.1, 10)

    return _calculate_widths(c)
