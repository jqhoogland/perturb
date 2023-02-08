from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type

import pandas as pd
import torch as t
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from perturb.types_ import WithOptions
from perturb.utils import stable_hash, to_tuple

if TYPE_CHECKING:
    from perturb.interventions.base import Intervention
    from perturb.variables.dataloaders import ExtendedDataLoader
    from perturb.variables.module import ExtendedModule
    from perturb.variables.optim import ExtendedOptimizer


class Trial:
    model_hyperparams: Dict[str, Any]
    opt_hyperparams: Dict[str, Any]

    def __init__(
        self,
        model: WithOptions[Type["ExtendedModule"]],
        opt: WithOptions[Type["ExtendedOptimizer"]],
        dl: Tuple[Type["ExtendedDataLoader"], Dict[str, Any], Dataset],
        interventions: Optional[List["Intervention"]] = None,
        control: Optional["Trial"] = None,
        device: str = "cpu"
    ):
        # TODO: What if model_hyperparams or opt_hyperparams are missing defaults?
        self.model_cls, self.model_hyperparams = to_tuple(model, {})
        self.model = None

        self.control: Optional["Trial"] = control

        self.opt_cls, self.opt_hyperparams = to_tuple(opt, {})
        self.opt = None

        self.dl_cls, self.dl_hyperparams, self.dataset = dl
        self.dl = None

        self.interventions = interventions or []

        self.logs = {}
        self.active = False

        self.step = 0
        self.device = device

    def activate(self):
        self.active = True

        self.model = self.model or self.model_cls(**self.model_hyperparams)  # type: ignore
        self.opt = self.opt or self.opt_cls(self.model.parameters(), **self.opt_hyperparams)  # type: ignore
        self.dl = self.dl or self.dl_cls(self.dataset, **self.dl_hyperparams)  # type: ignore

        self.model.to(self.device)

        if self.control is not None:
            self.control.activate()

    def deactivate(self):
        """
        Deactivate the trial to free up memory.
        """
        self.active = False

        self.model = None
        self.opt = None

        if self.control is not None:
            self.control.deactivate()

    def __enter__(self):
        self.activate()
        return self

    def __exit__(self, *args):
        self.deactivate()

    def loss(
        self, output: t.Tensor, target: t.Tensor, reduction: str = "mean"
    ) -> t.Tensor:
        return F.nll_loss(output, target, reduction=reduction)

    def run(self, n_epochs: int = 1, start: int = 0, **kwargs):
        with self:
            for epoch_idx in range(start, n_epochs):
                yield epoch_idx, self.run_epoch(epoch_idx, **kwargs)

    def run_epoch(self, epoch_idx: int, reset: bool = False, **kwargs):
        self.dl.step = epoch_idx * len(self.dl)  # To ensure consistent seeds for each epoch

        if self.dl.step < self.step and not reset:
            return

        for batch_idx, batch in enumerate(self.dl):
            if self.dl.step < self.step and not reset:
                continue

            # Seems a bit wasteful, but it's only a constant factor (relative to the training time)
            for intervention in self.interventions:
                intervention.toggle_if_needed(self)

            yield batch_idx, self.dl.step, batch, self.run_batch(batch, **kwargs)


    def run_batch(self, batch: Tuple[t.Tensor, t.Tensor]):
        # assert self.active and self.model is not None and self.opt is not None, "Trial not active"

        x, y = batch

        self.opt.zero_grad()  # type: ignore
        output = self.model(x)  # type: ignore
        loss = self.loss(output, y)
        loss.backward()
        self.opt.step()  # type: ignore

        self.step += 1

        return loss.item()

    def log(self, step: Optional[int] = None, **kwargs):
        if step is None:
            step = self.step

        self.logs[step] = self.logs.get(step, {})
        self.logs[step].update(kwargs)

    @property
    def unique_name(self):
        return f"{self.model_cls.__name__}_{self.hash}"

    @property
    def hash(self):
        return stable_hash(self.extra_repr)

    @property
    def intervention_hyperparams(self):
        hyperparams = {
            k: v
            for intervention in self.interventions
            for k, v in intervention.hyperparams.items()
        }

        return {k: hyperparams[k] for k in sorted(hyperparams.keys())}

    @property
    def hyperparams(self):

        hyperparams = {
            **self.model_hyperparams,
            **self.opt_hyperparams,
            **self.dl_hyperparams,
            **self.intervention_hyperparams,
        }

        # Consistent ordering
        return {k: hyperparams[k] for k in sorted(hyperparams.keys())}

    @property
    def extra_repr(self):
        return ", ".join(f"{k}={v}" for k, v in self.hyperparams.items())

    def df(self, full=True):
        """DataFrame of logs (add on hyperparams as cols)"""
        df = pd.DataFrame.from_dict(self.logs, orient="index")
        # df["step"] = df.index
        df = df.rename_axis("step").reset_index()

        if full:
            hyperparams = self.hyperparams.copy()

            for k, v in hyperparams.items():
                if isinstance(v, (list, tuple)):
                    hyperparams[k] = [v] * len(df)

            df = df.assign(**hyperparams)

        return df

    def __call__(self, *args, **kwargs) -> t.Tensor:
        assert self.active and self.model is not None, "Trial not active"
        return self.model(*args, **kwargs)

    def __getattribute__(self, __name: str) -> Any:
        """Forward attributes to model"""
        try:
            return super().__getattribute__(__name)
        except AttributeError:
            if self.model is not None:
                return getattr(self.model, __name)

    def with_interventions(self, interventions: List["Intervention"]):
        return self.__class__(
            model=(self.model_cls, self.model_hyperparams),
            opt=(self.opt_cls, self.opt_hyperparams),
            dl=(self.dl_cls, self.dl_hyperparams, self.dataset),
            interventions=[*self.interventions, *interventions],
            control=self,
            device=self.device
        )

    @property
    def init(self):
        return self.model.weight_initializer
    
    def __repr__(self) -> str:
        return f"{self.unique_name}({self.extra_repr})"