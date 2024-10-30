from backends import (
    __backends__ as setup_functions,
    check_backend_conflicts,
    convert_to_numpy,
    BackendNotSupported,
    BackendConflict,
)
from utilities import (
    Timer,
    estimate_repetitions,
    format_output,
    compute_statistics,
    get_benchmark_module,
    check_consistency,
)

import gc
import time
import jace

try:
    import cupy as cp
except ImportError:
    cp = None

def cleanup(device):
    jace.util.translation_cache.clear_translation_cache()
    gc.enable()
    gc.collect()

    if(device == "gpu"):
        mempool = cp.get_default_memory_pool()
        mempoolpin = cp.get_default_pinned_memory_pool()

        print(f"Def Memory Pool used: {mempool.used_bytes()}")
        print(f"Def Memory Pool total: {mempool.total_bytes()}")
        mempool.free_all_blocks()
        mempoolpin.free_all_blocks()
        cp.cuda.stream.get_current_stream().synchronize()
        print(f"Def Memory Pool used (A): {mempool.used_bytes()}")
        print(f"Def Memory Pool tota (A): {mempool.total_bytes()}")
    return


def main():
    benchmark = "benchmarks/equation_of_state"
    device = "gpu"
    bm_module, bm_identifier = get_benchmark_module(benchmark)
    jace_be = bm_module.try_import("jace")

    DEFAULT_SIZE = tuple( x * (1024**2) for x in [100, 200, 300, 400, 500])

    for size in DEFAULT_SIZE:
        print(f"START WITH SIZE: {size}")
        sa_np, ct_np, p_np = bm_module.generate_inputs_uncached(size)
        sa, ct, p = jace_be.prepare_inputs(sa_np, ct_np, p_np, device)
        del sa_np, ct_np, p_np

        cleanup(device)
        timings = []
        for rep in range(10):
            start = time.time()
            jace_be.run(sa, ct, p, device=device)
            end = time.time()
            timings.append(end - start)

        del sa, ct, p
        cleanup(device)

    print(timings)


if __name__ == "__main__":
    main()


