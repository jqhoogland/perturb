# Perturb

Some personal tools for running ML experiments:

- Run **variations** of anything and everything (weight initializations, architecture, hyperpararmeters, optimizer choices, etc.).
- Fine-grained **interventions** (perturb the weights, gradients, activations, hyperparameters, etc.).
- Take **checkpoints** any time.
- Custom **metrics** and **plotters** (record any metric you can imagine — on the models themselves, between models, the test sets, training sets, etc.).
- Train in **parallel** on a cluster (ok, not yet, but, you know, eventually) or in **serial** on your local machine. 
- **Reuse** past results when running a new intervention if you've already tested a particular condition/control.
- **Consistent** seed management. 

The library is organized as follows:
- `Experiment` is a collection of `Learners` differentiated by `Intervention`.
- `Trial` is a (`Model`, `Optimizer`, `DataLoader`, `Metrics`, `Intervention`) tuple. It is the basic unit of training.
- `Intervention` is a class that perturbs the model, optimizer, or data loader. 
- `Metrics` is a class that records metrics and logs them. 
- `Plotter` is a class that plots metrics.

> **Note:** This library is still in development. The API is subject to change. I definitely don't recommend using it yet, as I haven't ironed out all the kinks.

## Examples

Suppose I want to test the effect of a small weight perturbation at initialization. It's as simple as the following:

```python
from torchvision import datasets, transforms

from perturb.experiments import Experiment
from perturb.interventions import PerturbWeights
from perturb.metrics import Metrics
from perturb.plotter import Plotter
from perturb.models import Lenet5

train_set = datasets.MNIST('data', train=True, download=True,transform=transforms.ToTensor())
test_set = datasets.MNIST('data', train=False, download=True,transform=transforms.ToTensor())

exp = Experiment(
    model=Lenet5(),
    datasets=(train_set, test_set),
    interventions=[
        PerturbWeights.make_variations(
            epsilon=(0.001, 0.01, 0.1),  
            seed_weights=range(10)
        )
    ]
)

# 1 control + 3 x 10 interventions = 31 trials

exp.run(n_epochs=10)
```

Or maybe I want to compare the behavior of different optimizers.

```python
from perturb.interventions import ReinitWeights, ChangeOptimizer

exp = Experiment(
    model=Lenet5(),
    dataset=(train_set, test_set),
    variations=[
        ReinitWeights.make_variations(
            seed_weights=range(5)
        ),
        ChangeOptimizer.make_variations(
            optimizer=torch.optim.SGD,
            lr=(0.001, 0.01, 0.1),
        )
    ],
    interventions=[
        ChangeOptimizer.make_variations(
            optimizer=torch.optim.Adam,
        )
    ],
)

# (5 x 3 variations) x (1 control + 1 intervention) = 30 trials
```

The distinction between `variations` and `interventions` is for measuring purposes: it allows us to write metrics that are relative to a control condition. This way we don't have to keep multiple large models in memory at once.

Maybe instead I'd like to vary the batch size.

```python
exp = Experiment(
    model=Lenet5(),
    dataset=(train_set, test_set),
    variations=[
        ReinitWeights.make_variations(
            seed_weights=range(5)
        ),
        ChangeTrainLoader.make_variations(
            batch_size=32,
            seed_shuffle=(.1, .2, .3)
        )
    ],
    interventions=[
        ChangeTrainLoader.make_variations(
            batch_size=(64, 128, 256, 512, 1024),
        )
    ]
)

# (5 x 3 variations) x (1 control + 5 interventions) = 90 trials

```

Or maybe I want to test the effect of a temporary perturbation to momentum, depending on when it is applied during training.

```python
exp = Experiment(
    model=Lenet5(),
    dataset=(train_set, test_set),
    variations=[
        ReinitWeights.make_variations(
            seed_weights=range(10)
        ),
        ChangeOptimizer.make_variations(
            momentum=(0.9, 0.99, 0.999),
        ),
    ],
    interventions=[
        ChangeOptimizer.make_variations(
            momentum=lambda m: (m * (1 + epsilon) for epsilon in (0.001, -0.001, 0.01, -0.01, 0.1, -0.1)),
            when=((0, 100), (1000, 1100), (5000, 5100)),  # Step ranges to maintain perturbation
        )
    ]
)

# (10 x 3 variations) x (1 control + 6 interventions) = 210 trials

```

That's a *lot* of trials. My computer will take several days to run all of them.

So I can get rid of the `ReinitWeights`, which leaves me with a more reasonable 21 trials. 
After I've validated for a fixed choice of weight initialization, I can add it back in, and run the experiment again. Best of all, it'll automatically skip the trials that have already been run.

Alternatively, I can train for only a few epochs, validate the results, and then train for more epochs, picking up where I left off from the checkpoints.

This allows for a more iterative experimentation loop so you can explore more ground faster.

## Logging

By default, the `Metrics` will record the loss and accuracy on the training and test sets at the end of each epoch. `Plotting` is disabled by default, but you can enable it by passing `plot=True` to `Experiment.run`.


Often, you'll want to compute performance relative to some control condition (e.g., cross entropy relative to an unperturbed model). The `Trial` class has a `control` attribute that points to the control condition. You can use this to compute any metric you want by subclassing `Metrics`.

```python

class CustomMetrics(Metrics):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_metric('w', self.weight_norm)
        self.register_metric('dw_control', self.weight_distance_from_control)

    def weight_norm(self, trial):
        norm = t.zeros(1)

        for p in trial.model.parameters():
            norm += torch.norm(p)
        
        return norm
    
    def weight_distance_from_control(self, trial):
        distance = t.zeros(1)

        for p1, p2 in zip(trial.model.parameters(), trial.control.model.parameters()):
            distance += torch.norm(p1-p2) ** 2
        
        return distance ** 0.5

```


## Plotting

Let's take the example of the batch size experiment. 

```python

exp = Experiment(
    model=Lenet5(),
    dataset=(train_set, test_set),
    variations=[
        ReinitWeights.make_variations(
            seed_weights=range(5)
        ),
        ChangeTrainLoader.make_variations(
            batch_size=32,
            batch_order_seed_weights=(.1, .2, .3)
        )
    ],
    interventions=[
        ChangeTrainLoader.make_variations(
            batch_size=(64, 128, 256, 512, 1024),
        )
    ],
    metrics=CustomMetrics(),
    plotter=Plotter(
        average_over=('seed_shuffle', 'seed'),
    )
)

```