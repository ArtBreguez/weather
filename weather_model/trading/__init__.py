"""Trading subpackage for edge detection, position sizing, and execution."""
from .edge import EdgeDetector
from .sizing import KellyCriterion, RiskParity, PositionSizer
from .execution import TradeExecutor

__all__ = [
    "EdgeDetector",
    "KellyCriterion",
    "RiskParity",
    "PositionSizer",
    "TradeExecutor",
]
