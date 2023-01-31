from typing import Optional

from perturb.interventions.base import Intervention
from perturb.trials import Trial
from perturb.variables.dataloaders import ExtendedDataLoader


class ChangeTrainLoader(Intervention):
    hyperparams: dict
    prev_train_loader: Optional[ExtendedDataLoader] = None

    def __init__(self, when=None, **kwargs):
        super().__init__(when=when)

        self.hyperparams = kwargs

    def apply(self, trial: Trial):
        hyperparams = {**trial.train_loader.hyperparams, **self.hyperparams}

        self.prev_train_loader = trial.train_loader
        trial.train_loader = trial.train_loader.__class__(**hyperparams)

    def revert(self, trial: Trial):
        assert (
            self.prev_train_loader is not None
        ), "Cannot revert to previous train loader if it was not saved"
        trial.train_loader = self.prev_train_loader