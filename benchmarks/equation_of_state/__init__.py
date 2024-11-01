import math
import importlib
import functools
import sys


@functools.cache
def generate_inputs(size):
    return generate_inputs_uncached(size)


def generate_inputs_uncached(size):
    import numpy as np
    import gc

    np.random.seed(17)
    use_simple_shapes = True
    gc.enable()
    gc.collect()

    if use_simple_shapes:
        if not getattr(generate_inputs, "_printed_shape_warning", False):
            print("", file=sys.stderr)
            print("#========================================", file=sys.stderr)
            print("# USES SIMPLE SHAPE", file=sys.stderr)
            print("#========================================", file=sys.stderr)
            generate_inputs._printed_shape_warning = True
        shape = (
            math.ceil(2 * size ** (1 / 3)),
            math.ceil(2 * size ** (1 / 3)),
            math.ceil(0.25 * size ** (1 / 3)),
        )
        shape_p = shape

    else:
        shape = (
            math.ceil(2 * size ** (1 / 3)),
            math.ceil(2 * size ** (1 / 3)),
            math.ceil(0.25 * size ** (1 / 3)),
        )
        shape_p = (1, 1, shape[-1])

    s = np.random.uniform(1e-2, 10, size=shape)
    t = np.random.uniform(-12, 20, size=shape)
    p = np.random.uniform(0, 1000, size=shape_p)
    return s, t, p

def try_import(backend):
    try:
        return importlib.import_module(f".eos_{backend}", __name__)
    except ImportError:
        return None


def get_callable(backend, size, device="cpu"):
    backend_module = try_import(backend)
    inputs = generate_inputs(size)
    if hasattr(backend_module, "prepare_inputs"):
        inputs = backend_module.prepare_inputs(*inputs, device=device)
    return functools.partial(backend_module.run, *inputs, device=device)


__implementations__ = (
    "aesara",
    "cupy",
    "jax",
    "numba",
    "numpy",
    "pytorch",
    "tensorflow",
    "taichi",
    "jace",
)
