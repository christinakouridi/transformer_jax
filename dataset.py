import itertools
from typing import Any, Dict, List, NamedTuple

import jax.numpy as jnp
import numpy as np
from datasets.dataset_dict import Dataset


class Batch(NamedTuple):
    source_tokens: jnp.ndarray  # Integer tokens, shape [B, T]
    target_tokens: jnp.ndarray  # Integer tokens, shape [B, T]


def encode(
    sample: Dict[str, str], tokenizer: Any, sequence_len: int
) -> Dict[str, List[int]]:
    """Encodes the source and target sentences in the string sample to tokens using the
    pre-trained Byte-Pair Encoder. 
    
    This involves a number of steps:
        - Decorating the string and beginning and end of statement tokens.
        - Truncating sequences to the maximum sequence length.
        - Padding sequences up to the maximum sequence length (with the pad token).
    
    Args:
        sample: A sample in dictionary format. It contains the source and target strings.
        tokenizer: Pre-trained tokenizer to be called for tokenizing strings.
        sequence_len: The maximum sequence length to allow in the dataset. Any sequences
            that are longer than this limit after tokenization are truncated; those that
            are shorter are padded.

    Returns:
        Dictionary of source and target tokens.
    """
    def decorate(text: str):
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


def cycle_and_shuffle(
    data_iterator: Dataset, 
    tokenizer: Any, 
    sequence_len: int, 
    buffer_size: int,
    rng: np.random.Generator
) -> List[List[int]]:
    """Iterator that returns a sample at a time for batch construction. 
    
    As the dataset is too large to be loaded all at one, it is provided as an input 
    data_iterator that returns one sample at a time. The dataset is effectively shuffled 
    before a sample is returned by maintaining a small buffer of samples."""
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
    data_iterator: Dataset, 
    tokenizer: Any, 
    batch_size: int,
    sequence_len: int, 
    num_shuffle_batches: int,
    rng: np.random.Generator
) -> Batch:
    """Return a memory-efficient iterator over training batches.

    The iterator continuously cycles through the data, so can be repeatedly called during
    training. The data is effectively shuffled before batched.
    """
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


def encode_test_data(
    data_iterator: Dataset, tokenizer: Any, sequence_len: int
) -> np.ndarray:
    """Encodes the test dataset similarly to the train set. 
    
    The tokenizer was pretrained on the train dataset only. As the test set is much
    smaller than the train set, it is loadded in memory all at once instead of one sample
    at a time when needed."""
    encode_vectorised = np.vectorize(lambda x: encode(x, tokenizer, sequence_len))
    return encode_vectorised(data_iterator["translation"])


def get_test_batch_iterator(
    encoded_data: np.ndarray, batch_size: int, shuffle: bool, rng: np.random.Generator
):
    """Return an iterator over test batches."""
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
