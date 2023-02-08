import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch as t
import torch.nn as nn


@dataclass
class WeightInitializer(ABC):
    """A class that, when called, jointly initializes a sequence of models"""

    seed_weights: Any = 0
    initial_weights: Optional[Tuple[t.Tensor]] = None

    @abstractmethod
    def _apply(self, model: nn.Module):
        raise NotImplementedError

    def apply(self, model: nn.Module):
        t.manual_seed(self.seed_weights)
        self._apply(model)
        self.update(model)

    def update(self, model: nn.Module):
        if self.initial_weights is not None:
            for initial_weight, param in zip(self.initial_weights, model.parameters()):
                initial_weight.copy_(param.data)
        else:
            self.initial_weights = tuple(
                param.data.clone() for param in model.parameters()
            )

    @property
    @abstractmethod
    def hyperparams(self) -> dict:
        raise NotImplementedError
    
    def norm(self) -> t.Tensor:
        from perturb.variables.module import parameters_norm

        assert self.initial_weights is not None, "Must call update() before calling norm()"
        return parameters_norm(self.initial_weights)
    

@dataclass
class FlattenedKaimingHe(WeightInitializer):
    """A class that, when called, jointly initializes a sequence of models"""

    gain_multiplier: float = 1.0

    def _apply(self, model: nn.Module):
        model.apply(self._apply_to_layer)

    def _apply_to_layer(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Kaiming normal initialization (written by hand so we can constrain the norm of the matrix)
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(module.weight)
            n_params = fan_in * fan_out
            gain = nn.init.calculate_gain("relu") * self.gain_multiplier
            std = gain / math.sqrt(fan_in)

            # For a matrix whose elements are sampled from a normal distribution with mean 0 and standard deviation std,
            # the norm of the matrix is sqrt(fan_in) * std

            # Make sure the weights are perfectly normalized
            t.nn.init.normal_(module.weight.data)
            module.weight.data *= math.sqrt(n_params) * std / t.norm(module.weight.data)

            if module.bias is not None:
                # TODO: Not sure exactly how the bias should be initialized
                # t.nn.init.normal_(module.bias.data)
                # module.bias.data *= math.sqrt(fan_out) * std / t.norm(module.bias.data)
                t.nn.init.zeros_(module.bias.data)

    @property
    def hyperparams(self) -> dict:
        return {
            "seed_weights": self.seed_weights,
            "gain_multiplier": self.gain_multiplier,
        }
