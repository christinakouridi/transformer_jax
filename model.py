from dataclasses import dataclass
from typing import Optional, Sequence

import haiku as hk
import jax
import jax.numpy as jnp


def layer_norm(x: jnp.ndarray, name: Optional[str] = None) -> jnp.ndarray:
    """Layer normalisation.

    It normalizes each sample in the batch independently across all features.
    This means that layer normalisation is not a simple re-parameterization
    of the network like batch and weight normalisation.
    """
    return hk.LayerNorm(
        axis=-1, create_scale=True, create_offset=True, name=name
    )(x)


@dataclass
class TransformerConfig:
    """Bundles together parameters of the transformer model."""

    key_size: int = 64
    ff_size: int = 2048
    model_size: int = 512
    vocab_size: int = 36992
    num_heads: int = 8
    num_layers: int = 6
    dropout_rate: float = 0.1
    pad_token: int = 0


class Embeddings(hk.Module):
    def __init__(self, model_size: int, vocab_size: int) -> jnp.ndarray:
        super().__init__()
        self.token_embedding_map = hk.Embed(
            vocab_size, model_size, w_init=hk.initializers.TruncatedNormal(
                stddev=0.02
            )
        )  # [V, D]
        self.model_size = model_size

    def get_token_embedding_map(self) -> jnp.ndarray:
        return self.token_embedding_map.embeddings

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # TODO(CK): Speed up indexing with vmap.
        # [B, T, D]
        return self.token_embedding_map(x) * jnp.sqrt(self.model_size)


class PositionalEmbeddings(hk.initializers.Initializer):
    """Fixed positional embeddings layer.

    Encodes the abolute position of tokens, giving the model useful information
    on their order in the sequence. Specifically a positional embedding is
    derived from a sinusoid function for dimensions with even indices, and a
    cosine function with odd indices:

        - PE(pos, 2i) = sin(pos/10000^(2i/model_size))
        - PE(pos, 2i+1) = cos(pos/10000^(2i/model_size))

    where pos is the position and i is the dimension index.
    """

    def __call__(self, shape=Sequence[int], dtype=jnp.float32) -> jnp.ndarray:
        sequence_len, model_size = shape

        pos_embeddings = jnp.zeros((sequence_len, model_size), dtype=dtype)
        pos = jnp.arange(sequence_len, dtype=dtype)[:, None]  # [T, 1]
        i = jnp.arange(0, model_size, 2, dtype=dtype)[None, :]  # [1, D/2]

        angle = pos / jnp.exp(i * jnp.log(10000) / model_size)  # [T, D/2]
        pos_embeddings = pos_embeddings.at[:, 0::2].set(jnp.sin(angle))
        pos_embeddings = pos_embeddings.at[:, 1::2].set(jnp.cos(angle))
        return jax.lax.stop_gradient(pos_embeddings)  # [B, T, D]


class FFNBlock(hk.Module):
    """Feed forward neural netwrk block, applied to each token independently
    unlike the attention layers.
    """

    def __init__(
        self,
        ff_size: int,
        model_size: int,
        w_init: Optional[hk.initializers.Initializer] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.layer_1 = hk.Linear(ff_size, w_init=w_init)
        self.layer_2 = hk.Linear(model_size, w_init=w_init)

    def __call__(self, x: jnp.ndarray):
        y = self.layer_1(x)
        y = jax.nn.gelu(y)  # Improvement over ReLU, also used in GPT-2.
        y = self.layer_2(y)
        return y


class MultiHeadAttention(hk.Module):
    """Multi-headed attention module.

    It attends over sequences of vectors to generate representations for them.

    Steps:
        1. Compute embeddings of keys (K), queries (Q), and values (V).
        2. Compute the attention weights W = softmax(QK^T / sqrt(key_size)).
        3. Output is another projection WV^T.

    Glossary of shapes:
        - T: Sequence length.
        - D: Embedding (model) size.
        - H: Number of attention heads.
    """

    def __init__(
        self,
        num_heads: int,
        key_size: int,
        w_init: hk.initializers.Initializer,
        value_size: Optional[int] = None,
        model_size: Optional[int] = None,
        neg_inf: Optional[float] = -1e30,
        name: Optional[str] = None,
    ):
        """Initialises the module.

        Args:
          num_heads: Number of independent attention heads (H).
          key_size: The size of keys (K) and queries (Q) used for attention.
          w_init: Initializer for weights in the linear projections.
          value_size: Optional size of the value projection (V). If None,
            defaults to the key size (K).
          model_size: Optional size of the output embedding (D). If None,
            defaults to the key size multiplied by the number of heads (K * H).
          name: Optional name for this module.
        """
        super().__init__(name=name)
        self.num_heads = num_heads
        self.key_size = key_size  # equals the query_size
        self.value_size = value_size or key_size
        self.model_size = model_size or key_size * num_heads
        self.w_init = w_init
        self.neg_inf = neg_inf or jnp.finfo(jnp.float32)

    @hk.transparent
    def _linear_projection(
        self,
        x: jnp.ndarray,
        head_size: int,
        name: Optional[str] = None,
    ) -> jnp.ndarray:
        # [B, T, K] -> [B, T, H * K]
        y = hk.Linear(
                self.num_heads * head_size, w_init=self.w_init, name=name
            )(x)
        *leading_dims, _ = x.shape
        return y.reshape((*leading_dims, self.num_heads, head_size))

    def __call__(
        self,
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Computes multi-headed attention with queries, keys & values.

        It broadcasts over the (leading) batch dimensions.

        Args:
          query: Embeddings sequence used to compute queries; shape [B, T, K].
          key: Embeddings sequence used to compute keys; shape [B, T, K].
          value: Embeddings sequence used to compute values; shape [B, T, K].
          mask: Optional mask applied to attention weights; shape [B, H=1, T, T]

        Returns:
          output: A new sequence of embeddings, consisting of a projection of
            the attention-weighted value projections; shape [B, T, D].
        """
        *leading_dims, sequence_len, _ = query.shape

        # TODO(CK): remove bias terms from linear layers / layer norm?
        query_projection = self._linear_projection(
            query, self.key_size, "query"
        )  # [T, H, K]
        key_projection = self._linear_projection(key, self.key_size, "key")
        value_projection = self._linear_projection(
            value, self.value_size, "value"
        )  # [T, H, K]

        # Compute the attention logits. Divide by factor such that when the
        # queries and keys have unit variance, the attention weights will have
        # unit variance, allowing softmax to stay diffuse and not saturate much.
        attn_logits = jnp.einsum(
            "bthk,bThk->bhtT", query_projection, key_projection
        ) / jnp.sqrt(
            self.key_size
        )  # [B, H, T, T]

        # Compute attention weights.
        if mask is not None:
            if mask.ndim != attn_logits.ndim:
                raise ValueError(
                    f"The mask and attention logits must have the same number"
                    f"of dimensions. The mask has {mask.ndim} dimensions and"
                    f"the attention logits {attn_logits}."
                )
            # The mask is the same for all heads.
            attn_logits = jnp.where(
                mask, attn_logits, self.neg_inf  # [B, H, T, T]
            )

        attn_weights = jax.nn.softmax(attn_logits)  # [B, H, T, T]

        # Weight the values by the attention and flatten the head vectors.
        value_weighted = jnp.einsum(
            "bhtT, bThk -> bthk", attn_weights, value_projection
        )
        value_weighted = jnp.reshape(
            value_weighted, (*leading_dims, sequence_len, -1)
        )  # [B, T, H * K]

        # Apply another projection to get the final embeddings.
        output = hk.Linear(self.model_size, w_init=self.w_init)(
            value_weighted
        )  # [B, T, D]
        return output


@dataclass
class Encoder(hk.Module):
    """Encoder module of the transformer model.

    It maps an input sequence of symbol representations (x1, ..., xn) to a
    sequence of continuous representations z = (z1, ..., zn). The encoder
    consists of a stack of identical layers applied in sequence, where each
    layer performs the function: Dropout(SubLayer(LayerNorm(x))) + x. The
    residual connections ensure that there's a strong gradient signal across the
    model, and that information about the original sequence is not lost.
    """

    config: TransformerConfig

    def __call__(
        self,
        source_embeddings: jnp.ndarray,
        source_mask: Optional[jnp.ndarray] = None,
        is_training: bool = True,
    ) -> jnp.ndarray:
        """Performs a forward pass through the encoder module.

        Args:
            source_embeddings: Input sequence embeddings [B, T, D]
        Returns:
            h: Output sequence embeddings from encoder [B, T, D]
        """

        # Kaiming / He initialization -- Truncated normal with scaled stddev.
        w_init = hk.initializers.VarianceScaling(2 / self.config.num_layers)
        dropout_rate = self.config.dropout_rate if is_training else 0.0

        h = source_embeddings
        for _ in range(self.config.num_layers):
            # 1. Multi-head self-attention, Add & Norm
            # Note that in the original paper normalisation was after attention,
            # but the paper "Pre-LN Transformer: On Layer Normalization in the
            # Transformer Architecture" shows that the scale of gradients will
            # be lower, and thus training stability higher, if the layer
            # normalization precedes attention.
            h_norm = layer_norm(h)  # [B, T, D]
            h_attn = MultiHeadAttention(
                self.config.num_heads,
                self.config.key_size,
                w_init,
                model_size=self.config.model_size,
            )(
                h_norm, h_norm, h_norm, source_mask
            )  # [B, T, D]

            h_attn = hk.dropout(
                rng=hk.next_rng_key(), rate=dropout_rate, x=h_attn
            )
            h = h + h_attn

            # 2. FFN block
            h_norm = layer_norm(h)
            h_dense = FFNBlock(
                self.config.ff_size, self.config.model_size, w_init=w_init
            )(x=h_norm)

            h_dense = hk.dropout(
                rng=hk.next_rng_key(), rate=dropout_rate, x=h_dense
            )
            h = h + h_dense
        return layer_norm(h)  # [B, T, D]


@dataclass
class Decoder(hk.Module):
    """Decoder module of the transformer model."""

    config: TransformerConfig

    def __call__(
        self,
        target_embeddings: jnp.ndarray,  # [B, T, D]
        encoder_output: jnp.ndarray,  # [B, T, D]
        target_mask: Optional[jnp.ndarray] = None,  # [B, 1, T, T],
        target_source_mask: Optional[jnp.ndarray] = None,  # [B, 1, T, T],
        is_training: bool = True,
    ):
        """Performs a forward pass thorugh the decoder module."""
        h = target_embeddings
        e = encoder_output  # already layer-normalised

        # Kaiming / He initialization -- Truncated normal with scaled stddev.
        w_init = hk.initializers.VarianceScaling(2 / self.config.num_layers)
        dropout_rate = self.config.dropout_rate if is_training else 0.0
        _, sequence_len, _ = h.shape

        # [B, T] -> [B, H, T, T] where [T, T] is a lower triangular matrix.
        # Mask is the same for all heads so no need to repeat it
        causal_mask = jnp.tril(jnp.ones((1, 1, sequence_len, sequence_len)))
        target_mask = target_mask * causal_mask  # [B, H=1, T, T]
        # TODO(CK): we don't need to add cusal mask to target_source_mask?

        for _ in range(self.config.num_layers):
            # 1. Multi-head self-attention, Add & Norm
            h_norm = layer_norm(h)  # [B, T, D]
            h_attn = MultiHeadAttention(
                self.config.num_heads,
                self.config.key_size,
                w_init,
                model_size=self.config.model_size,
            )(
                h_norm, h_norm, h_norm, target_mask
            )  # [B, T, D]

            h_attn = hk.dropout(
                rng=hk.next_rng_key(), rate=dropout_rate, x=h_attn
            )
            h = h + h_attn

            # 2. Multi-head cross-attention with encoder output, Add & Norm
            h_norm = layer_norm(h)  # [B, T, D]
            h_attn = MultiHeadAttention(
                self.config.num_heads,
                self.config.key_size,
                w_init,
                model_size=self.config.model_size,
            )(
                query=h_norm, key=e, value=e, mask=target_source_mask
            )  # [B, T, D]

            h_attn = hk.dropout(
                rng=hk.next_rng_key(), rate=dropout_rate, x=h_attn
            )
            h = h + h_attn

            # 3. FFN block
            h_norm = layer_norm(h)
            h_dense = FFNBlock(
                ff_size=self.config.ff_size,
                model_size=self.config.model_size,
                w_init=w_init,
            )(h_norm)

            h_dense = hk.dropout(
                rng=hk.next_rng_key(), rate=dropout_rate, x=h_dense
            )
            h = h + h_dense

        return layer_norm(h)  # [B, T, D]


@dataclass
class Transformer:
    config: TransformerConfig

    def __call__(
        self,
        source_tokens: jnp.ndarray,  # [B, T]
        target_tokens: jnp.ndarray,  # [B, T]
        source_mask: Optional[jnp.ndarray] = None,  # [B, T]
        target_mask: Optional[jnp.ndarray] = None,  # [B, T]
        is_training: bool = True,
    ) -> jnp.ndarray:
        sequence_len = source_tokens.shape[-1]
        dropout_rate = self.config.dropout_rate if is_training else 0.0
        embeddings = Embeddings(
            self.config.model_size, self.config.vocab_size
        )  # [V, D]
        positional_embeddings = hk.get_parameter(
            "positional_embeddings",
            [sequence_len, self.config.model_size],
            init=PositionalEmbeddings(),
        )

        # ===== Masking =====
        if source_mask is None:
            source_mask = source_tokens != self.config.pad_token
        if target_mask is None:
            target_mask = target_tokens != self.config.pad_token

        # Compute this before the source and target masks are transformed.
        target_source_mask = jnp.expand_dims(
            jnp.einsum("bt, bT -> btT", target_mask, source_mask), axis=1
        )  # [B, T] -> [B, T, T] -> [B, 1, T, T]
        source_mask = jnp.expand_dims(
            jnp.einsum("bt, bT -> btT", source_mask, source_mask), axis=1
        )
        target_mask = jnp.expand_dims(
            jnp.einsum("bt, bT -> btT", target_mask, target_mask), axis=1
        )

        # ===== Encoder =====
        # Prepare input embeddings.
        token_embeddings = embeddings(source_tokens)
        source_embeddings = hk.dropout(
            rng=hk.next_rng_key(),
            rate=dropout_rate,
            x=token_embeddings + positional_embeddings,
        )
        # Forward pass.
        encoder_output = Encoder(self.config)(
            source_embeddings, source_mask, is_training
        )

        # =====  Decoder =====
        target_embeddings = embeddings(target_tokens)
        target_embeddings = hk.dropout(
            rng=hk.next_rng_key(),
            rate=dropout_rate,
            x=target_embeddings + positional_embeddings,
        )

        # Forward pass.
        decoder_output = Decoder(self.config)(
            target_embeddings,
            encoder_output,
            target_mask,
            target_source_mask,
            is_training,
        )  # [B, T, D]

        # Convert the decoder output to predicted next-token weights.
        next_token_weights = jnp.dot(
            decoder_output, embeddings.get_token_embedding_map().T
        )
        return next_token_weights  # [B, T, V]
