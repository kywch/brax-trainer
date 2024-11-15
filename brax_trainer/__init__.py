import os

# See https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html
# By default, JAX preallocates 75% of the total GPU memory. Reduce this to 25%
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.25"
