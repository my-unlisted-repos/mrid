import importlib.util

TORCH_INSTALLED = importlib.util.find_spec('torch') is not None