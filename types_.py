from typing import (TYPE_CHECKING, Any, Dict, Iterable, List, Tuple, TypeVar,
                    Union)

import torch as t
from torch import nn

if TYPE_CHECKING:
    from perturb.interventions.base import InterventionGroup

T = TypeVar("T")
S = TypeVar("S")

OptionalTuple = Union[T, Tuple[T, ...], List[T]]
WithOptions = Union[T, Tuple[T, Dict[str, Any]]]

ParameterOrTensor = Union[nn.parameter.Parameter, t.Tensor]
