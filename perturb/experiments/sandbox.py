# %%

import torch as t
from torchvision import datasets, transforms

from perturb.experiment import Experiment
from perturb.interventions.weights import PerturbWeights
from perturb.observations.metrics import FullPanelMetrics
from perturb.observations.plots import FullPanelPlotter, Plotter
from perturb.utils import setup
from perturb.variables.models import FCN

device = setup()


def get_mnist_data():
    train_ds = datasets.MNIST(
        root="data", train=True, download=True, transform=transforms.ToTensor()
    )
    test_ds = datasets.MNIST(
        root="data", train=False, download=True, transform=transforms.ToTensor()
    )

    return train_ds, test_ds


def get_cifar10_data():
    train_ds = datasets.CIFAR10(
        root="data", train=True, download=True, transform=transforms.ToTensor()
    )
    test_ds = datasets.CIFAR10(
        root="data", train=False, download=True, transform=transforms.ToTensor()
    )

    return train_ds, test_ds


def get_imagenet_data():
    train_ds = datasets.ImageFolder(
        root="data/imagenet/train", transform=transforms.ToTensor()
    )
    test_ds = datasets.ImageFolder(
        root="data/imagenet/val", transform=transforms.ToTensor()
    )

    return train_ds, test_ds


def get_data(dataset: str):
    if dataset == "mnist":
        return get_mnist_data()
    elif dataset == "cifar10":
        return get_cifar10_data()
    elif dataset == "imagenet":
        return get_imagenet_data()
    else:
        raise ValueError(f"Unknown dataset {dataset}")


DEFAULT_MODEL_HYPERPARAMS = dict(
    n_hidden=100,
)

DEFAULT_SGD_HYPERPARAMS = dict(
    lr=0.01,
    momentum=0.0,
    weight_decay=0.0,
)

exp = Experiment(
    model=(FCN, DEFAULT_MODEL_HYPERPARAMS),
    opt=(t.optim.SGD, DEFAULT_SGD_HYPERPARAMS),
    datasets=get_data("mnist"),
    interventions=[
        PerturbWeights.make_variations(
            epsilon=[0.001, 0.01, .1],
            seed_perturbation=range(3),
        )
    ],
    plotter=FullPanelPlotter(
        average_over=["seed_perturbation"],
        dir="plots/vanilla"
    ),
    metrics=FullPanelMetrics(ivl=200),
    name="vanilla",
    device=device
)

exp.run(n_epochs=5, n_epochs_at_a_time=2)                                                                                                                                                                                                                                                                                                                                                                                                                             # %%
# %%