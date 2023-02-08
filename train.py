import logging
import os
import time
from pathlib import Path
from typing import NamedTuple

import haiku as hk
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
from datasets import load_dataset
from omegaconf import DictConfig
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

from dataset import (encode_test_data, get_test_batch_iterator,
                     get_train_batch_iterator)
from model import Transformer, TransformerConfig

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"

log = logging.getLogger(__name__)
logging.getLogger("datasets_modules").setLevel(logging.ERROR)


class TrainingState(NamedTuple):
    """Container for the training state."""

    params: hk.Params
    opt_state: optax.OptState
    key: jnp.DeviceArray
    step: jnp.DeviceArray


@hydra.main(
    config_path="configs", config_name="transformer.yaml", version_base=None
)
def main(cfg: DictConfig):
    # ===== Set up =====
    base_dir = Path.cwd()
    jax.config.update("jax_log_compiles", cfg.debug.jax_log_compiles)
    jax.config.update("jax_disable_jit", cfg.debug.jax_disable_jit)

    log.info(f"Working directory: {base_dir}.")
    log.info(f"Devices: {jax.devices()}")
    log.debug(f"Config: {cfg}")

    # ===== Pretrained tokenizer =====
    tokenizer = Tokenizer.from_file(cfg.tokenizer.save_path)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        cls_token="<cls>",
        sep_token="<sep>",
        mask_token="<mask>",
        padding_side="right",
        truncation_side="right",
    )

    # ===== Data batch loader =====
    # At the moment this optimises for memory and not speed.
    rng_train = np.random.default_rng(cfg.exp.train_seed)
    train_data_iterator = load_dataset(
        path="wmt14", name="de-en", split="train", streaming=True
    )
    train_batch_iterator = get_train_batch_iterator(
        train_data_iterator,
        tokenizer,
        cfg.exp.batch_size,
        cfg.model.sequence_len,
        cfg.exp.num_shuffle_batches,
        rng=rng_train,
    )

    rng_test = np.random.default_rng(cfg.exp.test_seed)
    test_data_iterator = load_dataset(
        path="wmt14", name="de-en", split="test"
    )
    test_data = encode_test_data(
        test_data_iterator, tokenizer, cfg.model.sequence_len
    )

    # ===== Training set up =====
    transformer_cfg = TransformerConfig(
        key_size=cfg.model.key_size,
        ff_size=cfg.model.ff_size,
        model_size=cfg.model.model_size,
        vocab_size=cfg.model.vocab_size,
        num_heads=cfg.model.num_heads,
        num_layers=cfg.model.num_layers,
        dropout_rate=cfg.model.dropout_rate,
        pad_token=int(tokenizer.pad_token_id),
    )
    optimiser = optax.chain(
        optax.clip_by_global_norm(cfg.exp.grad_clip_value),
        optax.adam(
            cfg.exp.learning_rate
        ),  # TODO(CK): test if adafactor can reserve memory
    )

    def forward(
        source_tokens: jnp.ndarray,
        target_tokens: jnp.ndarray,
        is_training: bool,
        config: TransformerConfig,
    ) -> jnp.ndarray:
        """Performs a forward pass through the model.
        The source and target masks that indicate padding are recomputed on the
        fly to reserve memory.
        """
        model = Transformer(config=config)
        return model(
            source_tokens, target_tokens, is_training=is_training,
        )

    @hk.transform
    def loss_fn(
        source_tokens, target_tokens, is_training: bool = True
    ) -> jnp.ndarray:
        """Computes the loss of the language model with respect to its
        parameters."""
        # TODO(CK): differences in indexing with other implementations
        logits = forward(
            source_tokens, target_tokens, is_training, transformer_cfg
        )  # [B, T, V]
        targets = jax.nn.one_hot(
            target_tokens, cfg.model.vocab_size
        )  # [B, T, V]

        mask = jnp.greater(source_tokens, 0)  # TODO(CK): needed?
        log_likelihood = jnp.sum(targets * jax.nn.log_softmax(logits), axis=-1)
        return -jnp.sum(log_likelihood * mask) / jnp.sum(mask)

    @jax.jit
    def update(state: TrainingState, source_tokens, target_tokens):
        """Performs an sgd step and returns training metrics."""
        key, sub_key = jax.random.split(state.key)
        loss_and_grad_fn = jax.value_and_grad(loss_fn.apply)
        loss, gradients = loss_and_grad_fn(
            state.params, sub_key, source_tokens, target_tokens
        )

        updates, new_opt_state = optimiser.update(gradients, state.opt_state)
        new_params = optax.apply_updates(state.params, updates)
        new_state = TrainingState(
            params=new_params,
            opt_state=new_opt_state,
            key=key,
            step=state.step + 1,
        )
        metrics = {"step": state.step, "loss": loss}
        return new_state, metrics

    @jax.jit
    def init(key, source_tokens, target_tokens) -> TrainingState:
        key, sub_key = jax.random.split(key)
        initial_params = loss_fn.init(sub_key, source_tokens, target_tokens)
        initial_opt_state = optimiser.init(initial_params)
        return TrainingState(
            params=initial_params,
            opt_state=initial_opt_state,
            key=key,
            step=np.array(0),
        )

    # ===== Evaluation =====
    @jax.jit
    def _eval_step(state, source_tokens, target_tokens):
        key, sub_key = jax.random.split(state.key)
        loss_and_grad_fn = jax.value_and_grad(loss_fn.apply)
        loss, _ = loss_and_grad_fn(
            state.params,
            sub_key,
            source_tokens,
            target_tokens,
            is_training=False
        )
        return loss

    def evaluate(state: TrainingState, batch_iterator):
        losses = []
        for source_tokens, target_tokens in batch_iterator:
            losses.append(
                jax.vmap(_eval_step, in_axes=(None, 1, 1))(
                    state, source_tokens[:, None, :], target_tokens[:, None, :]
                )
            )
        return jnp.mean(jnp.asarray(losses))

    # ===== Initialisation =====
    key = jax.random.PRNGKey(cfg.exp.train_seed)
    source_tokens, target_tokens = next(train_batch_iterator)
    state = init(key, source_tokens, target_tokens)
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    log.info(f"Number of parameters: {param_count // (10**6)} million")

    # ===== Training =====
    start_time = time.time()
    for step in range(cfg.exp.max_steps):
        source_tokens, target_tokens = next(train_batch_iterator)
        state, metrics = update(state, source_tokens, target_tokens)

        if step % cfg.exp.log_frequency == 0:
            metrics["SPS"] = cfg.exp.log_frequency / (time.time() - start_time)

            if step % cfg.exp.test_frequency == 0:
                test_batch_iterator = get_test_batch_iterator(
                    test_data, cfg.exp.batch_size, shuffle=True, rng=rng_test
                )
                metrics["test_loss"] = evaluate(state, test_batch_iterator)

            log.info("".join(f"{k}: {v:.6f} |" for k, v in metrics.items()))
            start_time = time.time()


if __name__ == "__main__":
    main()
