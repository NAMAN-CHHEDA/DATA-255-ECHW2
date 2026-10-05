"""Model classes for the four homework experiments.

Importing this package imports torch via the model modules. The command-line
scripts import it only when a training implementation asks for a class.
"""

from src.models.gru_baseline import GRUBaseline
from src.models.kan_neural_ode import KANNeuralODE, ODEFuncKAN
from src.models.liquid_model import LiquidModel, LiquidRiskModel
from src.models.mlp_neural_ode import MLPNeuralODE, ODEFuncMLP
from src.models.neural_ode_common import NeuralODEBase

MODEL_REGISTRY: dict[str, type] = {
    "gru_baseline": GRUBaseline,
    "mlp_neural_ode": MLPNeuralODE,
    "kan_neural_ode": KANNeuralODE,
    "liquid_model": LiquidModel,
}

__all__ = [
    "GRUBaseline",
    "KANNeuralODE",
    "LiquidModel",
    "LiquidRiskModel",
    "MLPNeuralODE",
    "MODEL_REGISTRY",
    "NeuralODEBase",
    "ODEFuncKAN",
    "ODEFuncMLP",
]
