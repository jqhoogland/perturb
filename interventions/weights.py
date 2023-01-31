from dataclasses import dataclass
from typing import Callable, Literal

import torch as t

from perturb.interventions.base import Intervention
from perturb.utils import tqdm
from perturb.variables.module import ExtendedModule


def apply_householder_matrix_from_vertical_(x: t.Tensor, vs: t.Tensor):
    """Applies the above for the case that y = (0, ..., 0, 1)
    without having to compute the matrix explicitly.

    The Householder matrix maps a vector |x> onto |y>

    First, you normalize both vectors.

    Then you define |c> = |x> + |y>

    Then you define the Householder matrix:

    H = 2 |c><c| / (<c|c>) - I

    That is:

    H|v> = 2 |c><c|v> / <c|c> - |v>

    Parameters
    ----------
    x : t.Tensor shape (d,)
        Vector which defines the rotation. This is where (0, ..., 0, 1) is mapped.
    vs : t.Tensor shape (n, d)
        n Vectors to rotate

    """
    x_norm = t.norm(x)
    x /= x_norm
    x[-1] += 1

    vs -= (2 * t.inner(x, vs).view(-1, 1) / t.dot(x, x)) * x.view(1, -1)
    x[-1] -= 1
    x *= x_norm


def sample_from_hypersphere_intersection(
    r: t.Tensor,
    epsilon: float,
    n_samples: int,
):
    """Sample points from the intersection of two hyperspheres.

    Parameters
    ----------
    r : np.ndarray
        Vector that determines the radius of the larger hypersphere
    epsilon : float
        Radius of the smaller hypersphere, centered at r
    n_samples : int
        Number of samples to take

    Returns
    -------
    np.ndarray
        Array of shape (n_samples, r.shape[0])
    """

    d = r.shape[0]
    r_norm = t.norm(r)

    # Get the angle of the cone which goes through the center of the sphere and the intersection
    cone_angle = t.arccos(1 - epsilon**2 / (2 * r_norm**2))

    # Get the perp distance from r to the intersection
    epsilon_inner = r_norm * t.sin(cone_angle)

    # Sample a perturbation from the d-1 dimensional hypersphere of intersection
    perturbations = t.empty(n_samples, d)
    t.nn.init.normal_(perturbations)
    perturbations *= epsilon_inner / t.norm(perturbations[:, :-1], dim=1, keepdim=True)
    perturbations[:, -1] = 0

    # Apply the rotation
    apply_householder_matrix_from_vertical_(r, perturbations)

    # Shift the perturbations
    perturbations += r * t.cos(cone_angle)

    return perturbations



class PerturbWeights(Intervention):
    """Perturbs the weights of a model, while preserving the norm (i.e., performs a rotation on the sphere)."""

    def __init__(self,
        seed_perturbation: int,
        epsilon: float = 0.0,
        mode: Literal["absolute", "relative"] = "relative",
        norm: Callable = t.norm,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.seed_perturbation = seed_perturbation
        self.epsilon = epsilon
        self.mode = mode
        self.norm = norm

    def __post_init__(self):
        if self.mode == "absolute":
            return

        if self.epsilon > 2.0:
            raise ValueError(f"epsilon must be less than 2.0, but got {self.epsilon}")
        elif self.epsilon < 0.0:
            raise ValueError(
                f"epsilon must be greater than 0.0, but got {self.epsilon}"
            )

    @property
    def hyperparams(self) -> dict:
        return {
            "seed_perturbation": self.seed_perturbation,
            "epsilon": self.epsilon,
            "mode": self.mode,
            # "norm": self.norm,
        }

    def apply(self, model: ExtendedModule, verbose: bool = False):
        with t.no_grad():
            t.manual_seed(self.seed_perturbation)

            self.init_norm = model.norm().item()
            self.delta_norm = self.epsilon * self.init_norm

            if self.epsilon == 0.0:
                return
            if self.epsilon == 2.0:
                for p in model.parameters():
                    p.data = -p.data
                return
            
            progress = tqdm(
                model.parameters(),
                desc="Perturbing weights",
            ) if verbose else model.parameters()

            for p in progress:
                p_vec = p.data.view(-1)
                epsilon = (
                    self.epsilon * self.norm(p_vec).item()
                    if self.mode == "relative"
                    else self.epsilon
                )

                p.data = sample_from_hypersphere_intersection(
                    p_vec,
                    epsilon,
                    n_samples=1,
                ).view_as(p)

        # Register the change on the model's weight initializer
        # (so we can compute metrics relative to the starting point)
        # if self.when == 0:
        #     model.weight_initializer.update(model.parameters())


class ReinitWeights(Intervention):
    """Reinitializes the weights of a model."""

    seed_weights: int

    def __post_init__(self):
        if self.seed_weights is None:
            raise ValueError("seed_weights must be provided")
        if self.when != 0:
            raise ValueError("ReinitWeights must be applied at the beginning")

    @property
    def hyperparams(self) -> dict:
        return {
            "seed_weights": self.seed_weights,
        }

    def apply(self, model: ExtendedModule):
        with t.no_grad():
            model.weight_initializer.seed_weights = self.seed_weights
            model.weight_initializer.apply(model)


# ------------------------------------------------------------------------------

# def test_seed_weights():
#     from serimats.paths.variables.models import FCN

#     m0 = FCN(dict(n_hidden=1000))
#     m1 = FCN(dict(n_hidden=1000))

#     perturbation = RelativePerturbationInitializer(
#         seed_weights=0,
#         seed_perturbation=0,
#         epsilon=0.1,
#     )

#     perturbation.initialize_weights(m0)
#     perturbation.initialize_weights(m1)

#     for p0, p1 in zip(m0.parameters(), m1.parameters()):
#         assert t.allclose(p0, p1)


# def test_seed_perturbation():
#     from serimats.paths.variables.models import FCN

#     m0 = FCN(dict(n_hidden=1000))
#     m1 = FCN(dict(n_hidden=1000))

#     perturbation = RelativePerturbationInitializer(
#         seed_weights=0,
#         seed_perturbation=0,
#         epsilon=0.1,
#     )

#     perturbation(m0)
#     perturbation(m1)

#     for p0, p1 in zip(m0.parameters(), m1.parameters()):
#         assert t.allclose(p0, p1)


# def test_relative_perturbation():
#     from serimats.paths.variables.models import FCN

#     for epsilon in [0.0, 0.001, 0.01, 0.1, 0.5, 0.9, 1.0, 1.5, 2.0]:
#         m0 = FCN(dict(n_hidden=1000))
#         ms = [FCN(dict(n_hidden=1000)) for _ in range(10)]

#         norms = []
#         deltas = []

#         for j in range(3):
#             for i, m in enumerate(ms):
#                 perturbation = RelativePerturbationInitializer(
#                     seed_weights=j,
#                     seed_perturbation=i,
#                     epsilon=epsilon,
#                 )
#                 perturbation.initialize_weights(m0)
#                 perturbation(m)

#                 m_norm = m.norm()
#                 m0_norm = m0.norm().item()

#                 distance = m.lp_distance(m0).item()

#                 norms.append(m0_norm)
#                 deltas.append(distance / m_norm)

#                 assert np.allclose(m_norm, m0_norm, atol=1e-2, rtol=1e-2)
#                 assert np.allclose(
#                     distance,
#                     (perturbation.epsilon * m0_norm),
#                     atol=1e-2,
#                     rtol=1e-2,
#                 )
