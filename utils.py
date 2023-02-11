import jax
import jax.numpy as jnp

def apply_label_smoothing(labels: jnp.ndarray, label_smoothing: int) -> jnp.ndarray:
    """Applies label smoothing to one-hot targets."""

    num_classes = labels.shape[-1]
    one = jax.lax.convert_element_type(1, labels.dtype)
    label_smoothing = jax.lax.convert_element_type(label_smoothing, labels.dtype)
    return (one - label_smoothing) * labels + label_smoothing / num_classes