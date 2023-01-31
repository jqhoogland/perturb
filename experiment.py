import itertools
from typing import Dict, Iterable, List, Optional, Tuple, Type, TypeVar, Union

import pandas as pd
from torch import optim
from torch.utils.data import Dataset

from perturb.interventions.base import Intervention, InterventionGroup
from perturb.observations.checkpoints import Checkpointer
from perturb.observations.metrics import AbstractMetrics, Metrics
from perturb.observations.plots import Plotter
from perturb.trials import Trial
from perturb.types_ import WithOptions
from perturb.utils import stable_hash, to_tuple, tqdm
from perturb.variables.dataloaders import ExtendedDataLoader
from perturb.variables.module import ExtendedModule


def _fix_intervention_group(
    intervention_group: InterventionGroup,
) -> List[List[Intervention]]:

    if isinstance(intervention_group, Intervention):
        return [[intervention_group]]

    elif isinstance(intervention_group, Iterable):
        intervention_group = list(intervention_group)

        if isinstance(intervention_group[0], Intervention):
            return [intervention_group]

        return [list(i) for i in intervention_group]

    raise TypeError(
        "intervention_group must be an Intervention, Iterable[Intervention], or Iterable[Iterable[Intervention]]"
    )


class Experiment:
    def __init__(
        self,
        model: WithOptions[Type[ExtendedModule]],
        datasets: Union[Tuple[Dataset, ...], Dict[str, Dataset]],
        interventions: InterventionGroup,
        opt: WithOptions[Type[optim.Optimizer]] = optim.SGD,
        dl: WithOptions[Type[ExtendedDataLoader]] = ExtendedDataLoader,
        variations: Optional[InterventionGroup] = None,
        plotter: Optional[Plotter] = None,
        checkpointer: Optional[Checkpointer] = None,
        metrics: Optional[AbstractMetrics] = None,
        name: Optional[str] = None,
    ):
        # Datasets

        if isinstance(datasets, tuple):
            if len(datasets) != 2:
                raise ValueError("datasets must be a tuple of length 2")

            datasets = {"train": datasets[0], "test": datasets[1]}

        # Data loaders

        dl_cls, dl_kwargs = to_tuple(dl, {"batch_size": 64, "shuffle": True})  # type: ignore
        self.dls: Dict[str, ExtendedDataLoader] = {}

        for name, dataset in datasets.items():
            self.dls[name] = dl_cls(dataset, **dl_kwargs)  # type: ignore

        # Metrics

        self.metrics = metrics or Metrics()
        self.metrics.add_dls(self.dls)

        # Plotter

        self.plotter = plotter

        if self.plotter is not None:
            self.plotter.register(self)

        # Checkpointer

        self.checkpointer = checkpointer or Checkpointer()

        # Controls

        if variations is None:
            control = Trial(model, opt, (dl_cls, dl_kwargs, datasets["train"]))  # type: ignore
            self.variations = [control]
        else:
            self.variations = [
                Trial(model, opt, (dl_cls, dl_kwargs, datasets["train"]), variation_combo)  # type: ignore
                for variation_combo in itertools.product(
                    *_fix_intervention_group(variations)
                )
            ]

        # Interventions

        self.interventions = [
            trial.with_interventions(intervention_combo)
            for trial in self.variations
            for intervention_combo in itertools.product(
                *_fix_intervention_group(interventions)
            )
        ]

        # Trials

        self.trials = [
            *self.variations,
            *self.interventions,
        ]

        self.name = name or stable_hash(self.hyperparams)

    def run(self, n_epochs: int, reset: bool = False, n_epochs_at_a_time: Optional[int] = None): 
        epoch_idx, batch_idx, step = 0, 0, 0
        epoch_len = len(self.dls["train"]) / self.dls["train"].batch_size

        n_epochs_at_a_time = n_epochs_at_a_time or n_epochs

        for round_idx in tqdm(range(n_epochs // n_epochs_at_a_time), desc="Trials"):

            for trial in tqdm(self.trials, desc="Trials"):
                trial = self.checkpointer.load(trial)
                
                print(trial)
                for epoch_idx, epoch in tqdm(
                    trial.run(n_epochs_at_a_time, start=round_idx * n_epochs_at_a_time, reset=reset),
                    desc="Epochs", total=n_epochs_at_a_time
                ):
                    for batch_idx, step, _, loss in epoch:
                        if step % self.metrics.ivl == 0:
                            self.metrics.measure(
                                epoch_idx=epoch_idx,
                                batch_idx=batch_idx,
                                step=step,
                                trial=trial,
                            )

                        if step % self.checkpointer.ivl == 0:
                            self.checkpointer.save(
                                epoch_idx=epoch_idx,
                                batch_idx=batch_idx,
                                step=step,
                                trial=trial,
                            )

            if self.plotter:
                self.plotter.plot(
                    epoch_idx=epoch_idx,
                    batch_idx=batch_idx,
                    step=step,
                )

    def df(self):
        """Returns a dataframe with the logs of all trials"""
        return pd.concat([trial.df() for trial in self.trials])

    @property
    def hyperparams(self):
        return [trial.hyperparams for trial in self.trials]

    def __getitem__(self, i):
        return self.trials[i]

    def __len__(self):
        return len(self.trials)

    def get_interventions(self, control: Trial):
        return [trial for trial in self.interventions if trial.control == control]

    def __repr__(self) -> str:
        return f"Experiment_{self.name}({self.trials})"
