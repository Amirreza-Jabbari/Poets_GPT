## PoetsGPT

PoetsGPT is a character-level GPT language model implementation in PyTorch, designed for training and generating poetry or any text at the character level. This repository provides all the necessary scripts and configurations to train your own model, fine-tune on custom datasets, and generate creative text.

---

## Table of Contents

* [Features](#features)
* [Requirements](#requirements)
* [Installation](#installation)
* [Usage](#usage)

  * [Training](#training)
  * [Generation](#generation)
* [Configuration](#configuration)
* [Model Architecture](#model-architecture)
* [Contributing](#contributing)
* [License](#license)

---

## Features

* Character-level GPT model (Transformer-based) implemented in PyTorch
* Configurable model size, number of layers, heads, and embedding dimensions
* Training with mixed precision (automatic) when CUDA is available
* Checkpoint saving and loading for resuming training or inference
* Simple script interface for both training and generation

---

## Requirements

* Python 3.8 or higher
* PyTorch 1.11+ (with CUDA for GPU acceleration)
* tqdm

---

## Installation

1. Clone the repository:

```bash
git clone [https://github.com/Amirreza-Jabbari/Poets_GPT.git](https://github.com/Amirreza-Jabbari/Poets_GPT.git)
cd PoetsGPT
```

2. Prepare your text data file (e.g., `input.txt`) containing training text.

3. (Optional) Create a Python virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### Training

Launch training with default settings:

```bash
python PoetsGPT_LM.py \
  --input_file input.txt \
  --batch_size 128 \
  --block_size 256 \
  --max_iters 50000 \
  --eval_interval 500 \
  --learning_rate 3e-4 \
  --n_embd 384 \
  --n_head 6 \
  --n_layer 6 \
  --dropout 0.2
```

* Checkpoints are saved to `PoetsGPT.pth` at each eval interval.
* Training and validation loss are reported periodically.

### Generation

After training or loading an existing checkpoint, generate text using:

```bash
python PoetsGPT_LM.py --generate --generate_len 500
```

Generated output is printed to stdout and saved in `generated_output.txt`.

---

## Configuration

All script options can be viewed via:

```bash
python PoetsGPT_LM.py --help
```

Key arguments:

* `--input_file`: Path to the text file for training
* `--batch_size`: Number of sequences per batch
* `--block_size`: Context length (sequence length)
* `--max_iters`: Total training iterations
* `--eval_interval`: Iterations between evaluations and checkpoints
* `--learning_rate`: Initial learning rate for AdamW optimizer
* `--n_embd`: Embedding dimension size
* `--n_head`: Number of attention heads
* `--n_layer`: Number of transformer blocks
* `--dropout`: Dropout probability
* `--device`: `cpu` or `cuda`
* `--checkpoint`: Path to save/load model state
* `--generate`: Flag to run generation only
* `--generate_len`: Number of tokens to generate
* `--seed`: Random seed for reproducibility
* `--num_workers`: DataLoader worker count

---

## Model Architecture

* **Embedding**: Token and position embeddings combined to initialize input
* **Transformer Blocks**: Stacked `n_layer` blocks, each with:

  * Multi-head causal self-attention
  * Position-wise feed-forward network
  * Residual connections and layer normalization
* **Output**: Final layer normalization followed by a linear head projecting to vocabulary logits

See [PoetsGPT_LM.py](PoetsGPT_LM.py) for full implementation details.

---

## Contributing

Contributions are welcome! Please:

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/foo`)
3. Commit your changes (`git commit -m "feat: add foo"`)
4. Push to the branch (`git push origin feature/foo`)
5. Open a Pull Request

Ensure code is formatted and follows PEP8 style guidelines.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
