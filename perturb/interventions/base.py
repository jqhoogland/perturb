from abc import ABC, abstractmethod
from typing import Iterable, Tuple, Union

from perturb.trials import Trial

# Apply (1) at the start, (2) at a specific step, (3) for a range of steps
When = Union[None, int, Tuple[int, int]]

class Intervention(ABC):
    def __init__(self, when: When = None):
        self.when = when or 0

    @classmethod
    def make_variations(cls, *args, **kwargs) -> Iterable["Intervention"]:
        for i, arg in enumerate(args):
            if isinstance(arg, Iterable):
                return [
                    variation
                    for a in arg
                    for variation in cls.make_variations(
                        *args[:i], a, *args[i + 1 :], **kwargs
                    )
                ]
            
        for k, v in kwargs.items():
            if isinstance(v, Iterable):
                if k == "when" and not (isinstance(v, tuple) and len(v) == 2):
                    continue

                return [
                    variation
                    for a in v
                    for variation in cls.make_variations(*args, **{**kwargs, k: a})
                ]

        return [cls(*args, **kwargs)]

    @abstractmethod
    def apply(self, trial: Trial):
        raise NotImplementedError

    def revert(self, trial: Trial):
        raise ValueError(
            f"Not supported for this intervention ({self.__class__.__name__})"
        )

    def toggle_if_needed(self, trial: Trial):
        step = trial.step

        if isinstance(self.when, int):
            if step == self.when:
                self.apply(trial)
        else:
            start, end = self.when
            if step == start:
                self.apply(trial)
            elif step == end:
                self.revert(trial)

    def __enter__(self, trial: Trial):
        self.apply(trial)

    def __exit__(self, trial: Trial):
        self.revert(trial)

    @property
    @abstractmethod
    def hyperparams(self) -> dict:
        raise NotImplementedError

InterventionGroup = Union[
    Intervention, Iterable[Intervention], Iterable[Iterable[Intervention]]
]