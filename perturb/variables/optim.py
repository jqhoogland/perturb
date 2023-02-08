from typing import Type

from torch import optim


class ExtendedOptimizer(optim.Optimizer):
    @property
    def hyperparams(self) -> dict:
        """Assumes that the optimizer has only a single set of hyperparams
        (and not a separate set for each parameter group)"""
        param_group = self.param_groups[0]

        hyperparams = {}

        for k, v in param_group.items():
            if k not in ["params", "foreach", "maximize", "capturable", "fused"]:
                hyperparams[k] = v

        return hyperparams


def extend_optimizer(optimizer: Type[optim.Optimizer]) -> ExtendedOptimizer:
    """Extends an optimizer to include a hyperparams property."""

    if issubclass(optimizer, optim.Optimizer):
        # Insert ExtendedOptimizer into the class hierarchy
        mro = optimizer.mro()
        mro.insert(mro.index(optim.Optimizer), ExtendedOptimizer)

        dict_ = optimizer.__dict__.copy()

        return type(optimizer.__name__, tuple(mro), dict_)  # type: ignore
    else:
        raise TypeError("optimizer must be a subclass of optim.Optimizer")


ExtendedSGD = extend_optimizer(optim.SGD)
ExtendedAdam = extend_optimizer(optim.Adam)
