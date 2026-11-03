"""Tool conditions A-D. Each module exposes TOOLS and dispatch(name, args)."""
import importlib

CONDITIONS = ("A", "B", "C", "D")


def load(condition):
    return importlib.import_module(f"tools.condition_{condition.lower()}")
