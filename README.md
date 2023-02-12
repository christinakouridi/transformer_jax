# Attention Is All You Need implementation in Jax
## Overview
This repository includes a Jax implementation of the Transformer architecture introduced in the paper ["Attention Is All You Need"](https://arxiv.org/pdf/1706.03762.pdf).
The aim is not to reproduce results exactly, but create a playground for testing large-scale training techniques and methods for improving memory and time complexity. 

All experiments apply the original encoder-decoder model to the WMT 2014 Enlgish-to-German translation task. 

## Installation

1. Create a new environment and install python 3.9, for example using conda:
```shell
$ conda create --name transformer_jax python=3.9
```
2. Within that environment, install JAX. Follow the instructions in the [official JAX repository](https://github.com/google/jax#installation) as the installation will differ depending on your hardware accelerator. The requirements that I have used can be found in `requiremenets_jax.txt`.

3. Install the remaining dependencies.
```shell
$ pip install -r requirements.txt
```

## Data
The transformer is trained on the `WMT 2014 Enlgish-to-German` dataset which constists of about 4.5 million sentence pairs. We start from raw strings, pre-process them, and train a tokenizer from scratch to encode them using byte-pair encoding. We utilise the `Hugging Face datasets` and `transformer` libraries for handling the tokenizer and data processing.


## Run
The default parameters in `src/configs/transformer.yaml` are the same as the original paper apart from `sequence_length` and `batch_size`. These base values are low enough to comfortably run on a `GeForce RTX 2070 Super` GPU (~70% memory utilisation).

Steps for running the transformer:

1. Train the tokenizer on the WMT 2014 Enlgish-to-German dataset. This is automatically cached for subsequent runs. The tokenizer needs a few minutes to be trained, but the script runs for longer to identify the maximum sequence length for the tokenized source and target sentences.

```shell
$ python src/train_tokenizer.py 
```

2. At the moment, by default the installation and parameters are configured for running on a GPU. You can reduce the architecture size by adjusting parameters in `src/configs/transformer.yaml`.

3. Train the transformer. 
```shell
$ python src/train.py
```

## Changes on the original architecture
To be included soon!