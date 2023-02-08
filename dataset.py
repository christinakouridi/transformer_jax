import itertools
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np


class Batch(NamedTuple):
    source_tokens: jnp.ndarray  # Integer tokens, shape [B, T]
    target_tokens: jnp.ndarray  # Integer tokens, shape [B, T]


def encode(sample, tokenizer, sequence_len: int):
    def decorate(text):
        decorated = f"{tokenizer.bos_token} {text} {tokenizer.eos_token}"
        decorated = decorated.replace("\n", tokenizer.sep_token)
        return decorated

    # Here we don't return the mask to indicate the end of sequences / padding
    # to reserve memory; instead we compute these on the fly within the network.
    source_tokens = tokenizer(
        text=decorate(sample["en"]),
        truncation=True,
        max_length=sequence_len,
        padding="max_length",
    )["input_ids"]

    target_tokens = tokenizer(
        text=decorate(sample["de"]),
        truncation=True,
        max_length=sequence_len,
        padding="max_length",
    )["input_ids"]
    return {"source_tokens": source_tokens, "target_tokens": target_tokens}


def cycle_and_shuffle(data_iterator, tokenizer, sequence_len, buffer_size, rng):
    token_iterator = itertools.cycle(
        data_iterator.map(
            lambda x: encode(x["translation"], tokenizer, sequence_len),
            remove_columns=["translation"],
        )
    )
    buffer = [next(token_iterator) for _ in range(buffer_size)]
    rng.shuffle(buffer)
    for sample in token_iterator:
        idx = rng.integers(0, buffer_size - 1)
        result = buffer[idx]
        buffer[idx] = sample
        yield list(result.values())


def get_train_batch_iterator(
    data_iterator, tokenizer, batch_size, sequence_len, num_shuffle_batches, rng
):
    data_cycle = cycle_and_shuffle(
        data_iterator,
        tokenizer,
        sequence_len,
        buffer_size=batch_size * num_shuffle_batches,
        rng=rng,
    )
    while True:
        batch = np.stack([next(data_cycle) for _ in range(batch_size)])
        yield Batch(
            source_tokens=jnp.asarray(batch[:, 1, :]),
            target_tokens=jnp.asarray(batch[:, 0, :]),
        )


def encode_test_data(data_iterator, tokenizer, sequence_len):
    encode_vectorised = np.vectorize(
        lambda x: encode(x, tokenizer, sequence_len)
    )
    return encode_vectorised(data_iterator["translation"])


def get_test_batch_iterator(encoded_data, batch_size, shuffle, rng):
    if shuffle:
        rng.shuffle(encoded_data)

    def _get_batch(i):
        data = encoded_data[i * batch_size:i * batch_size + batch_size]
        batch = np.stack(
            [[x["target_tokens"], x["source_tokens"]] for x in data]
        )
        return batch

    num_batches = len(encoded_data) // batch_size
    for i in range(num_batches):
        batch = _get_batch(i)
        yield Batch(
            source_tokens=jnp.asarray(batch[:, 1, :]),
            target_tokens=jnp.asarray(batch[:, 0, :]),
        )

    # Get the remaining data that don't fit in a full batch.
    batch = _get_batch(i + 1)
    yield Batch(
        source_tokens=jnp.asarray(batch[:, 1, :]),
        target_tokens=jnp.asarray(batch[:, 0, :]),
    )
