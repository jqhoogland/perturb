# Weight Space measurements

import functools
from abc import ABC
from dataclasses import dataclass
from functools import wraps
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Union

import torch as t
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

from perturb.variables.module import parameters_norm

if TYPE_CHECKING:
    from perturb.experiment import Trial


def metric(a: t.Tensor, b: t.Tensor, p="fro", **kwargs):
    return t.norm(a - b, p=p, **kwargs)


class AbstractMetrics(ABC):
    def __init__(self, dls: Optional[Dict[str, DataLoader]] = None, ivl=1000) -> None:
        self.dls = dls or {}
        self.ivl = ivl

        # Method name: (metric name(s), latex name(s), requires data)
        self.metrics: Dict[
            str, Tuple[Union[str, Tuple[str, ...]], Union[str, Tuple[str, ...]], bool]
        ] = {}  # TODO This would make more sense as a class attribute via __new__

    def add_dl(self, name: str, dl: DataLoader) -> None:
        self.dls[name] = dl

    def add_dls(self, dls: Dict[str, DataLoader]) -> None:
        self.dls.update(dls)

    def register_metric(
        self,
        name: Union[str, Tuple[str, ...]],
        fn: Callable,
        latex: Optional[Union[str, Tuple[str]]] = None,
        requires_data: bool = True,
    ):
        latex = latex or name
        self.metrics[fn.__name__] = (name, latex, requires_data) 

    @property
    def data_metrics(self):
        return {
            fn_name: (metrics, latex, requires_data)
            for fn_name, (metrics, latex, requires_data) in self.metrics.items()
            if requires_data
        }

    @property
    def data_free_metrics(self):
        return {
            fn_name: (metrics, latex, requires_data)
            for fn_name, (metrics, latex, requires_data) in self.metrics.items()
            if not requires_data
        }

    def measure(
        self,
        epoch_idx: int,
        batch_idx: int,
        step: int,
        trial: "Trial",
        **kwargs,
    ) -> Dict[str, float]:

        measurements = {}

        for fn_name, (metrics, _, _) in self.data_free_metrics.items():
            values = getattr(self, fn_name)(trial, **kwargs)

            if not isinstance(metrics, tuple):
                measurements[metrics] = values

            for metric, value in zip(metrics, values):
                measurements[metric] = value

        if not self.dls:
            return measurements

        for dl_name, dl in self.dls.items():
            for fn_name, (metrics, _, _) in self.data_metrics.items():
                if not isinstance(metrics, tuple):
                    measurements[f"{metrics}_{dl_name}"] = t.zeros(1)
                else:
                    for metric in metrics:
                        measurements[f"{metric}_{dl_name}"] = t.zeros(1)

            with t.no_grad():
                for data, target in dl:    
                    for fn_name, (metrics, _, _) in self.data_metrics.items():
                        values = getattr(self, fn_name)(trial, data, target, **kwargs)

                        # Sorry this is gross

                        if not isinstance(metrics, tuple):
                            if values is None:
                                measurements[f"{metrics}_{dl_name}"] = None
                            else:
                                measurements[f"{metrics}_{dl_name}"] += values
                        else:
                            for metric, value in zip(metrics, values):
                                if value is None:
                                    measurements[f"{metric}_{dl_name}"] = None
                                else:
                                    measurements[f"{metric}_{dl_name}"] += value
                        
            for full_metric_name, value in measurements.items():
                if isinstance(value, t.Tensor):
                    measurements[full_metric_name] = value.item()

                if full_metric_name.endswith(f"_{dl_name}") and value is not None:
                    measurements[full_metric_name] /= len(dl)

        trial.log(step=step, epoch=epoch_idx, batch=batch_idx, **measurements)

        return measurements


class Metrics(AbstractMetrics):
    def __init__(self, dls: Optional[Dict[str, DataLoader]] = None, ivl=200) -> None:
        super().__init__(dls, ivl)

        self.register_metric(("L", "acc"), self.loss_and_accuracy, requires_data=True)

    def loss_and_accuracy(
        self, trial: "Trial", data: t.Tensor, target: t.Tensor, **kwargs
    ) -> Tuple:
        output = trial.model(data)  # type: ignore
        loss = F.nll_loss(output, target, reduction="sum")
        pred = output.argmax(dim=1, keepdim=True)
        acc = pred.eq(target.view_as(pred)).sum()

        return loss, acc


class FullPanelMetrics(AbstractMetrics):
    def __init__(self, dls: Optional[Dict[str, DataLoader]] = None, ivl=1000) -> None:
        super().__init__(dls, ivl)

        self.register_metric(
            ("L", "acc", "L_cf", "acc_cf"),
            self.loss_and_accuracy_with_cf,
            requires_data=True,
        )
        self.register_metric(
            ("w_norm", "w_norm_init", "w_norm_cf"),
            self.ws,
            requires_data=False,
        )

        self.register_metric(
            ("dw_init", "dw_cf", "dw_control_normed"),
            self.dws,
            requires_data=False,
        )

        self.register_metric(
            ("cos_sim_init", "cos_sim_control"),
            self.cos_sims,
            requires_data=False,
        )

    def loss_and_accuracy_with_cf(
        self, trial: "Trial", data: t.Tensor, target: t.Tensor, **kwargs
    ) -> Tuple:
        output = trial.model(data)  # type: ignore
        loss = F.nll_loss(output, target, reduction="sum")
        pred = output.argmax(dim=1, keepdim=True)
        acc = pred.eq(target.view_as(pred)).sum()

        if trial.control is None:
            return loss, acc, None, None

        control_output = trial.control.model(data)  # type: ignore
        loss_cf = F.cross_entropy(control_output, target, reduction="sum")
        pred_cf = control_output.argmax(dim=1, keepdim=True)
        acc_cf = pred_cf.eq(target.view_as(pred_cf)).sum()

        return loss, acc, loss_cf, acc_cf

    @staticmethod
    def ws(trial: "Trial"):
        w_norm = trial.norm()
        w_norm_init = trial.init.norm()

        if trial.control is None:
            return w_norm, w_norm / w_norm_init, None
        
        w_norm_cf = trial.control.norm()

        return w_norm, w_norm / w_norm_init, w_norm / w_norm_cf

    def dws(self, trial: "Trial"):
        epsilon = self.epsilon(trial)
        dw_init = trial.lp_distance(trial.init)

        if trial.control is None:
            return dw_init, None, None
        
        dw_control = trial.lp_distance(trial.control)
        dw_control_normed = self.normalize_wrt(dw_control, epsilon)

        return dw_init, dw_control, dw_control_normed

    def cos_sims(self, trial: "Trial"):
        cos_sim_init = trial.cosine_similarity(trial.init)
        
        if trial.control is None:
            return cos_sim_init, None
        
        cos_sim_control = trial.cosine_similarity(trial.control)

        return cos_sim_init, cos_sim_control

    def normalize_wrt(self, n: t.Tensor, d) -> t.Tensor:
        if d == 0:
            return t.tensor(0)

        return n / d

    @staticmethod
    def epsilon(trial: "Trial") -> t.Tensor:
        # TODO: These two are not the same for small epsilon?
        # return trial.weight_initializer.epsilon
        norm = t.zeros(1, device=trial.device)

        if trial.control is None:
            return norm

        if (
            trial.weight_initializer.initial_weights is None
            or trial.control.init.initial_weights is None
        ):
            raise ValueError("Initial weights not set")

        for p1, p2 in zip(
            trial.weight_initializer.initial_weights,
            trial.control.init.initial_weights,
        ):
            norm += t.norm(p1 - p2, p="fro") ** 2

        return norm.sqrt()
