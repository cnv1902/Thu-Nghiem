import numpy as np
from importlib import import_module


def _load_cupy():
    try:
        return import_module("cupy")
    except Exception:
        return None


def get_array_module(use_gpu=False):
    """
    Return (xp, gpu_enabled) where xp is either numpy or cupy.
    Falls back to numpy if cupy is unavailable.
    """
    if use_gpu:
        try:
            cp = _load_cupy()
            if cp is not None:
                return cp, True
        except Exception:
            pass
    return np, False


def to_backend_array(x, xp):
    return xp.asarray(x)


def to_numpy(x):
    cp = _load_cupy()
    if cp is not None and isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    return np.asarray(x)


def cleanup_gpu_memory():
    """
    Release cached GPU memory blocks when running CuPy.
    Safe no-op on CPU-only environments.
    """
    try:
        cp = _load_cupy()
        if cp is None:
            return
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass
