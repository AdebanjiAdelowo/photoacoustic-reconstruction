"""Bound the compiled code that repeated j-Wave simulations leave behind.

Every call of j-Wave's `simulate_wave_propagation` compiles a new XLA program, and JAX keeps all of
them: about 80 memory mappings per simulation (measured). Linux limits a process to 65,530 mappings
(vm.max_map_count), so after roughly 800 simulations the next compilation fails with
"LLVM compilation error: Cannot allocate memory". The expanded evaluation runs more than 3,000.

`simulation_done` is called after each simulation and empties JAX's compilation caches every
CLEAR_EVERY calls. This releases memory only: reconstructions are bitwise identical with and
without it. The cost is one recompilation (about 2 s) per CLEAR_EVERY simulations.
"""

import jax

CLEAR_EVERY = 50
_simulations = 0


def simulation_done(value=None):
    """Count one finished simulation, clearing JAX's caches when due. Returns `value` unchanged, so
    it can wrap a call inside an expression."""
    global _simulations
    _simulations += 1
    if _simulations % CLEAR_EVERY == 0:
        jax.clear_caches()
    return value
