import itertools
from pathlib import Path
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from perturb.observations.checkpoints import Pathy
from perturb.utils import dict_to_str, tqdm

if TYPE_CHECKING:
    from perturb.experiment import Experiment


# def plot_metric_scaling(
#     experiment: "Experiment",
#     df: pd.DataFrame,
#     metric: Tuple[CallableWithLatex, ...],
#     step: Optional[int] = None,
#     comparison: str = "epsilon",
#     sample_axis: str = "seed_perturbation",
#     include_baseline: bool = True,  # Whether to plot the first value of comparison
#     baseline: dict = {"epsilon": 0.0},
#     **kwargs,
# ) -> Tuple[Figure, List[List[plt.Axes]]]:
#     if step:
#         df = df.loc[df["step"] <= step]

#     metric_labels = tuple(m.__latex__[0] for m in metric)

#     comparison_label = var_to_latex(comparison)
#     comparison_values = df[comparison].unique().tolist()

#     if not include_baseline:
#         comparison_values = comparison_values[1:]

#     if "perturbation" in kwargs:
#         del kwargs["perturbation"]

#     details = dict_to_latex(kwargs)

#     averages = df

#     if not include_baseline:
#         for k, v in baseline.items():
#             averages = averages.loc[averages[k] != v]

#     averages = averages.groupby([comparison, "step"]).mean(numeric_only=True)
#     averages.reset_index(inplace=True)

#     steps = df["step"].unique()

#     # Figure is as wide as there are metrics and
#     # as tall as there are choices of `comparison` (+ 1 to compare averages)
#     fig, axes = plt.subplots(
#         len(comparison_values) + 1,
#         len(metric),
#         figsize=(len(metric) * 5, len(comparison_values) * 5),
#     )

#     # This works better with escaped LaTeX than f-strings. I think.
#     title = (
#         "$"
#         + " ? "
#         # str(metric_labels) +
#         "$ for $"
#         +
#         # str(comparison_label) +
#         "("
#         + ", ".join(map(lambda s: str(s), comparison_values))
#         + " )$\n($"
#         + details
#         + "$)"
#     )

#     fig.suptitle(title)
#     fig.tight_layout(pad=4.0)

#     for c, (m_fn, m_label) in enumerate(zip(metric, metric_labels)):
#         m = m_fn.__name__

#         for r, v in enumerate(comparison_values):
#             # Get the data for this comparison value

#             data = df.loc[df[comparison] == v]
#             sample_values = data[sample_axis].unique()

#             # Plot the data
#             for sample in sample_values:
#                 axes[r][c].plot(
#                     steps,
#                     data.loc[data[sample_axis] == sample][m].values,
#                     alpha=0.75,
#                     linewidth=0.5,
#                 )

#             # Plot the average across samples
#             axes[r][c].plot(
#                 steps,
#                 averages.loc[averages[comparison] == v][m].values,
#                 color="black",
#                 linestyle="--",
#             )

#             axes[r][c].set_title(f"${m_label}$ for ${comparison_label} = {v}$")
#             axes[r][c].set_xlabel("Step $t$")
#             axes[r][c].set_ylabel(f"${m_label}$")

#     # Plot the comparison of averages
#     for c, (m_fn, m_label) in enumerate(zip(metric, metric_labels)):
#         m = m_fn.__name__

#         # Plot the averages across comparison values
#         for v in comparison_values:
#             axes[-1][c].plot(
#                 steps,
#                 averages.loc[averages[comparison] == v][m].values,
#                 label=f"${v}$",
#             )

#         axes[-1][c].set_title(f"${m_label}$ across ${comparison_label}$")
#         axes[-1][c].set_xlabel("Step $t$")
#         axes[-1][c].set_ylabel(f"$\\overline{{{m_label}}}$")
#         axes[-1][c].legend()

#     return fig, axes


def slugify(s: str) -> str:
    return s.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "").lower()


class Plotter:
    def __init__(
        self,
        ivl=5,
        experiment: Optional["Experiment"] = None,
        average_over: Optional[List[str]] = None,
        dir: Optional[Pathy] = None,
    ):
        self.experiment = experiment
        self.ivl = ivl
        self.average_over = average_over or []
        self.plot_fns: List[Tuple[Callable, str]] = [
            (self.plot_performance, "Performance")
        ]
        self.dir = dir

    def register_plot_fn(self, plot_fn: Callable, name: Optional[str] = None):
        name = name or plot_fn.__name__
        self.plot_fns.append((plot_fn, name))

    def register(self, experiment: "Experiment"):
        self.experiment = experiment

    def plot(self, epoch_idx: int, batch_idx: int, step: int, overwrite: bool = True):
        assert self.experiment is not None, "Experiment not registered with Plotter"

        df = self.experiment.df()
        df = df.loc[df["step"] <= step]

        if self.dir:
            path = Path(self.dir)
            path.mkdir(parents=True, exist_ok=True)

        for plot_fn, name in tqdm(self.plot_fns, "Plotting..."):
            slug = slugify(name)

            fig, axes = plot_fn(df)
            fig.suptitle(f"{name} at step {step}")

            if overwrite and path:
                fig.savefig(
                    str(path / f"{slug}.png"),  # type: ignore
                )
            else:
                print("WTF", path, name)

    def _plot(self, df: pd.DataFrame, metrics: List[str], **kwargs):
        assert self.experiment is not None, "Experiment not registered with Plotter"

        variations = self.experiment.variations
        variation_hyperparams = [
            {k: v for k, v in var.hyperparams.items() if k not in self.average_over}
            for var in variations
        ]

        constant_hyperparams = {}
        variable_hyperparams = set()

        for hyperparams in variation_hyperparams:
            for k, v in hyperparams.items():
                if k in variable_hyperparams:
                    continue
                if k not in constant_hyperparams:
                    constant_hyperparams[k] = v
                elif constant_hyperparams[k] != v:
                    del constant_hyperparams[k]
                    variable_hyperparams.add(k)

        # Plot a grid with rows for each variation (modulo average_over) and columns for each metric
        fig, axes = plt.subplots(
            len(variation_hyperparams),
            len(metrics),
            figsize=(len(metrics) * 5, len(variation_hyperparams) * 5),
        )

        fig.subplots_adjust(top=0.95, bottom=0.1)

        if len(variation_hyperparams) == 1:
            axes = [axes]

        for i, hyperparams in enumerate(variation_hyperparams):
            data = df.copy()

            for k, v in hyperparams.items():
                data = data.loc[data[k] == v]

            full_steps = data["step"].unique()
            average = data.groupby("step").mean(numeric_only=True)
            average = average.reset_index()

            for j, metric in enumerate(metrics):
                # Plot the data
                key_values_to_average_over = zip(self.average_over, itertools.product(
                    *[
                        data[k].unique() for k in self.average_over
                    ]
                ))

                for k, v in key_values_to_average_over:
                    trial = data.loc[data[k] == v]
                    axes[i][j].plot(
                        trial["step"].values,
                        trial[metric].values,
                        alpha=0.5,
                        linewidth=0.25,
                    )

                # Plot the average across samples
                axes[i][j].plot(
                    full_steps,
                    average[metric].values,
                    color="black",
                    linestyle="--",
                )

                unique_variation_hyperparams = {k: v for k, v in hyperparams.items() if k in variable_hyperparams}

                axes[i][j].set_title(f"{metric}\n{dict_to_str(unique_variation_hyperparams)}")
                axes[i][j].set_xlabel("Step $t$")
                axes[i][j].set_ylabel(f"{metric}")

        # Plot constant hyperparams at bottom in center
        fig.text(
            0.5,
            0.05,
            dict_to_str(constant_hyperparams),
            ha="center",
            va="center",
        )

        return fig, axes

    def plot_performance(self, df: pd.DataFrame, **kwargs):
        return self._plot(
            df,
            metrics=["L_train", "acc_train", "L_test", "acc_test"],
        )


class FullPanelPlotter(Plotter):
    def __init__(self, ivl=1000, experiment: Optional["Experiment"] = None, average_over: Optional[List[str]] = None, dir: Optional[Pathy] = None):
        super().__init__(ivl, experiment, average_over, dir)

        self.plot_fns.extend([
            (self.plot_rel_performance, "Relative Performance"),
            (self.plot_weight_metrics, "Weights"),
        ])

    def plot_rel_performance(self, df: pd.DataFrame, **kwargs):
        return self._plot(
            df,
            metrics=["L_cf_train", "acc_cf_train", "L_cf_test", "acc_cf_test"],
        )
    
    def plot_weight_metrics(self, df: pd.DataFrame, **kwargs):
        return self._plot(
            df,
            metrics=["w_norm", "w_norm_init", "w_norm_cf", "dw_init", "dw_cf", "dw_control_normed", "cos_sim_init", "cos_sim_control"],
        )
