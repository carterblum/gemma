from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import linen as nn


@dataclass
class MutableState:
    counter: int = 0
    values: list = None

    def __post_init__(self):
        if self.values is None:
            self.values = []


class DummyModule(nn.Module):
    mutable_attr: MutableState

    def __call__(self, x):
        # Does nothing, just returns input
        return x


# Create the module
mutable_state = MutableState(counter=5, values=[1, 2, 3])
model = DummyModule(mutable_attr=mutable_state)

# JIT the apply method, with model as static argument
jitted_apply = jax.jit(model.apply, static_argnums=(0,))

# Initialize and apply
key = jax.random.PRNGKey(0)
x = jnp.ones((4, 8))
params = model.init(key, x)
output = jitted_apply(params, x)

print("First call successful")

# Now mutate and see recompilation
model.mutable_attr.counter = 10
output2 = jitted_apply(params, x)  # Should recompile
print("Second call after mutation")
