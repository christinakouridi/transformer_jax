import jax
import jax.numpy as jnp
import optax

def apply_label_smoothing(labels: jnp.ndarray, label_smoothing: int) -> jnp.ndarray:
    """Applies label smoothing to one-hot targets."""

    num_classes = labels.shape[-1]
    one = jax.lax.convert_element_type(1, labels.dtype)
    label_smoothing = jax.lax.convert_element_type(label_smoothing, labels.dtype)
    return (one - label_smoothing) * labels + label_smoothing / num_classes

def get_optimizer(
    name: str, learning_rate: float, grad_clip_value: float
) -> optax.GradientTransformation:
    """Set up the optimizer."""
    # TODO(CK): add learning rate scheduler
    if name == 'adam':
        optimizer = optax.chain(
            optax.clip_by_global_norm(grad_clip_value),
            optax.adam(learning_rate),
        )
    elif name == 'adafactor':
        optimizer = optax.chain(
            optax.clip_by_global_norm(grad_clip_value),
            optax.adafactor(learning_rate),
        )
    else:
        raise NotImplementedError(
            "This optimizer is not supported. Valid options: 'adam', 'adafactor'"
        )
    return optimizer