import logging
import os
import time
from pathlib import Path
from typing import NamedTuple, Generator, Optional

import haiku as hk
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
from chex import PRNGKey
from datasets import load_dataset
from omegaconf import DictConfig
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast
from utils import apply_label_smoothing
from dataset import encode_test_data, get_test_batch_iterator, get_train_batch_iterator
from model import Transformer, TransformerConfig

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"

# Reserves memory through deallocation but > 30% slower.
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


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
    pad_token = int(tokenizer.pad_token_id)

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
    test_data_iterator = load_dataset(path="wmt14", name="de-en", split="test")
    test_data = encode_test_data(test_data_iterator, tokenizer, cfg.model.sequence_len)

    # ===== Training set up =====
    transformer_cfg = TransformerConfig(
        key_size=cfg.model.key_size,
        ff_size=cfg.model.ff_size,
        model_size=cfg.model.model_size,
        vocab_size=cfg.model.vocab_size,
        num_heads=cfg.model.num_heads,
        num_layers=cfg.model.num_layers,
        dropout_rate=cfg.model.dropout_rate,
        pad_token=pad_token,
    )
    optimiser = optax.chain(
        optax.clip_by_global_norm(cfg.exp.grad_clip_value),
        # TODO(CK): test if adafactor can reserve memory
        # TODO(CK): add learning rate scheduler
        optax.adam(cfg.exp.learning_rate),
    )

    def forward(
        source_tokens: jnp.ndarray,
        target_tokens: jnp.ndarray,
        is_training: bool,
        config: TransformerConfig,
    ) -> jnp.ndarray:
        """Performs a forward pass through the model.

        The source and target masks that indicate padding are recomputed on the fly to 
        reserve memory.
        """
        model = Transformer(config=config)
        return model(source_tokens, target_tokens, is_training=is_training)
    
    @hk.transform
    def loss_fn(
        source_tokens: jnp.ndarray, 
        target_tokens: jnp.ndarray, 
        pad_token: int, 
        label_smoothing: Optional[float] = None,
        is_training: bool = True,
    ) -> jnp.ndarray:
        """Computes the loss of the language model with respect to its parameters."""
        # TODO(CK): update indexing to handle BOS end EOS tokens.
        logits = forward(
            source_tokens, target_tokens, is_training, transformer_cfg
        )  # [B, T, V]
        targets = jax.nn.one_hot(target_tokens, cfg.model.vocab_size)  # [B, T, V]

        if label_smoothing is not None:
            targets = apply_label_smoothing(targets, label_smoothing)

        mask = jnp.not_equal(target_tokens, pad_token)
        log_likelihood = jnp.sum(targets * jax.nn.log_softmax(logits), axis=-1)
        return -jnp.sum(log_likelihood * mask) / jnp.sum(mask)

    @jax.jit
    def update(
        state: TrainingState, 
        source_tokens: jnp.ndarray, 
        target_tokens: jnp.ndarray, 
        pad_token: int,
        label_smoothing: float
    ):
        """Performs an sgd step and returns training metrics."""
        key, sub_key = jax.random.split(state.key)
        loss_and_grad_fn = jax.value_and_grad(loss_fn.apply)
        loss, gradients = loss_and_grad_fn(
            state.params,
            sub_key,
            source_tokens,
            target_tokens,
            pad_token,
            label_smoothing
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
    def init(
        key: PRNGKey, 
        source_tokens: jnp.ndarray, 
        target_tokens: jnp.ndarray, 
        pad_token: int,
        label_smoothing: int
    ) -> TrainingState:
        key, sub_key = jax.random.split(key)
        initial_params = loss_fn.init(
            sub_key, source_tokens, target_tokens, pad_token, label_smoothing
        )
        initial_opt_state = optimiser.init(initial_params)
        return TrainingState(
            params=initial_params,
            opt_state=initial_opt_state,
            key=key,
            step=np.array(0),
        )

    # ===== Evaluation =====
    @jax.jit
    def eval_step(
        state: TrainingState, 
        source_tokens: jnp.ndarray, 
        target_tokens: jnp.ndarray, 
        pad_token: int,
    ) -> jnp.ndarray:
        _, sub_key = jax.random.split(state.key)
        loss_and_grad_fn = jax.value_and_grad(loss_fn.apply)
        loss, _ = loss_and_grad_fn(
            state.params,
            sub_key,
            source_tokens,
            target_tokens,
            pad_token,
            is_training=False
        )
        return loss

    def evaluate(
        state: TrainingState, 
        batch_iterator: Generator, 
        pad_token: int, 
    ) -> jnp.ndarray:
        losses = []
        for source_tokens, target_tokens in batch_iterator:
            losses.append(eval_step(state, source_tokens, target_tokens, pad_token))
        return jnp.mean(jnp.asarray(losses))

    # ===== Initialisation =====
    key = jax.random.PRNGKey(cfg.exp.train_seed)
    source_tokens, target_tokens = next(train_batch_iterator)
    state = init(key, source_tokens, target_tokens, pad_token, cfg.exp.label_smoothing)
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    log.info(f"Number of parameters: {param_count // (10**6)} million")

    # ===== Training =====
    start_time = time.time()
    for step in range(cfg.exp.max_steps):

        source_tokens, target_tokens = next(train_batch_iterator)
        state, metrics = update(
            state, source_tokens, target_tokens, pad_token, cfg.exp.label_smoothing
        )

        if step % cfg.exp.log_frequency == 0:
            metrics["SPS"] = cfg.exp.log_frequency / (time.time() - start_time)

            test_batch_iterator = get_test_batch_iterator(
                test_data, cfg.exp.batch_size, shuffle=True, rng=rng_test
            )
            metrics["test_loss"] = evaluate(state, test_batch_iterator, pad_token)

            log.info("".join(f"{k}: {v:.6f} |" for k, v in metrics.items()))
            start_time = time.time()


if __name__ == "__main__":
    main()
