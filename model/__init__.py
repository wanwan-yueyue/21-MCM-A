# model/__init__.py
from .decomposition import DecompositionDynamics
from .growth import FungalGrowthModel
from .interaction import InteractionModel

__all__ = ["DecompositionDynamics", "FungalGrowthModel", "InteractionModel"]