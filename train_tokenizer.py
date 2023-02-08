import logging
from pathlib import Path

import hydra
from datasets import load_dataset
from omegaconf import DictConfig
from tokenizers import Regex, Tokenizer, normalizers
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

log = logging.getLogger(__name__)


class BPETokenizer:
    def __init__(self, vocab_size: int, batch_size: int):
        self.vocab_size = vocab_size
        self.batch_size = batch_size

        # Tokenize. Translate text data into numbers that the model can process.
        self.tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        self.tokenizer.normalizer = normalizers.Sequence(
            [
                normalizers.Replace("``", '"'),
                normalizers.Replace("''", '"'),
                normalizers.NFKD(),  # Normalization
                normalizers.StripAccents(),
                normalizers.Replace(Regex(" {2,}"), " "),  # Standardise spacing
            ]
        )
        self.tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

    def batch_iterator(self, data):
        """Shared source-target vocabulary."""
        for i in range(0, len(data), self.batch_size):
            batch = data[i: i + self.batch_size]["translation"]
            yield [sentence for sample in batch for sentence in sample.values()]

    def train(self, data):
        trainer = BpeTrainer(
            show_progress=True,
            vocab_size=self.vocab_size,
            special_tokens=[
                "<cls>",
                "<sep>",
                "<unk>",
                "<pad>",
                "<mask>",
                "<s>",
                "</s>",
            ],
        )
        self.tokenizer.train_from_iterator(self.batch_iterator(data), trainer)
        if self.tokenizer.get_vocab_size() != self.vocab_size:
            raise ValueError(
                f"The tokenizer has vocab_size: {self.vocab_size} instead of "
                f"{self.vocab_size}."
            )

    def save(self, path):
        if path.suffix != ".json":
            path = path / "tokenizer.json"

        path.parent.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(str(path))


@hydra.main(config_path="configs", config_name="transformer.yaml")
def main(cfg: DictConfig):
    bpe = BPETokenizer(
        vocab_size=cfg.tokenizer.vocab_size, batch_size=cfg.tokenizer.batch_size
    )
    pretokenized_sample = bpe.tokenizer.pre_tokenizer.pre_tokenize_str(
        'Training a transformer in JAX is fun'
    )
    log.info(
        f"Tokenisation before training of 'Training a transformer in JAX is "
        f"fun': {pretokenized_sample}."
    )

    data = load_dataset(path="wmt14", name="de-en", split="train")
    log.info(f"Data loaded! Number of rows: {data.num_rows}")

    bpe.train(data)
    path = Path.cwd() / cfg.tokenizer.save_path
    bpe.save(path)
    log.info(f"Trained finished. Stored trained tokenizer in: {path}")

    # Checks.
    output = bpe.tokenizer.encode("Training a transformer in JAX is fun")
    log.info(output.tokens)
    log.info(output.ids)
    log.info(bpe.tokenizer.decode(output.ids))

    max_sequence_len_source = max(
        list(
            map(
                lambda x: len(bpe.tokenizer.encode(x["en"]).tokens),
                data["translation"]
            )
        )
    )
    log.info(
        f"Maximum sequence length of tokenized data: {max_sequence_len_source}"
    )


if __name__ == "__main__":
    main()
