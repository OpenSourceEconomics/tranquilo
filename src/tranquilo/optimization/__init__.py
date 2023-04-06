import inspect

from tranquilo.optimization import (
    scipy_optimizers,
)
from tranquilo.optimization.tranquilo import tranquilo

MODULES = [
    scipy_optimizers,
    tranquilo,
]

ALL_ALGORITHMS = {}
AVAILABLE_ALGORITHMS = {}
for module in MODULES:
    func_dict = dict(inspect.getmembers(module, inspect.isfunction))
    for name, func in func_dict.items():
        if hasattr(func, "_algorithm_info"):
            ALL_ALGORITHMS[name] = func
            if func._algorithm_info.is_available:
                AVAILABLE_ALGORITHMS[name] = func
