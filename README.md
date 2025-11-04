# Reproduce [*Attention is All You Need*](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) (PyTorch)

## Introduction

This repository is an unofficial reproduction of the experiments conducted in the seminal paper "Attention is All You Need" [[1](#reference)] using PyTorch. It implements the Transformer architecture and reproduces the results on two WMT14 datasets:
- Transformer-base model on WMT14 English-German translation task
- Transformer-big model on WMT14 English-French translation task

The training, validation, and test sets are from WMT14 [[2](#reference)]. This implementation achieves comparable results to the original paper.

## Model Architecture

This implementation uses PyTorch's native `nn.Transformer` module with the following configurations:

### Transformer-base
- **d_model**: 512
- **num_heads**: 8
- **num_layers**: 6 (both encoder and decoder)
- **dim_feedforward**: 2048
- **dropout**: 0.1

### Transformer-big
- **d_model**: 1024
- **num_heads**: 16
- **num_layers**: 6 (both encoder and decoder)
- **dim_feedforward**: 4096
- **dropout**: 0.3

Both models use BPE (Byte Pair Encoding) for English-German and WordPiece for English-French tokenization.

## Results

### Transformer (base) on WMT14 English-German

For the transformer (base) trained on the WMT14 English-German dataset, the reported BLEU score in the original paper is 27.3, shown by the red dashed line in the following figures.

For the experiment, the model is trained by four A40 GPUs. The reproduced results are from the average of 3 runs, and the error bands represent the interval of $\pm2$ standard deviations.

| ![Iterations vs. BLEU Score](./doc/transformer-base_en-de/step_vs_bleu.png) | ![Iterations vs. Val Loss](./doc/transformer-base_en-de/step_vs_valloss.png) |
|:--:|:--:|
| # of Iterations vs. BLEU Score | # of Iterations vs. Validation Loss |

| ![Training time vs. BLEU Score](./doc/transformer-base_en-de/time_vs_bleu.png) | ![Training time vs. Val Loss](./doc/transformer-base_en-de/time_vs_valloss.png) |
|:--:|:--:|
| Training time vs. BLEU Score | Training time vs. Validation Loss |

The table below reports the total number of iterations, the BLEU scores evaluated by the trained model at the last iteration, and the total training time.

|  Step  |        AMP         |   BLEU Score    | Training Time (hours) |
|:------:|:------------------:|:---------------:|:---------------------:|
| 112760 | :negative_squared_cross_mark: | 0.2760 ± 0.0013 |    7.7557 ± 0.0066    |
| 112760 | :white_check_mark: | 0.2758 ± 0.0022 |    5.2069 ± 0.0010    |

### Transformer (big) on WMT14 English-French

For the transformer (big) trained on the WMT14 English-French dataset, the reported BLEU score in the original paper is 41.0, shown by the red dashed line in the following figures.

For the experiment, the model is trained by eight A100 GPUs. The reproduced results are from a single run.

| ![Iterations vs. BLEU Score](./doc/transformer-big_en-fr/step_vs_bleu.png) | ![Iterations vs. Val Loss](./doc/transformer-big_en-fr/step_vs_valloss.png) |
|:--:|:--:|
| # of Iterations vs. BLEU Score | # of Iterations vs. Validation Loss |

| ![Training time vs. BLEU Score](./doc/transformer-big_en-fr/time_vs_bleu.png) | ![Training time vs. Val Loss](./doc/transformer-big_en-fr/time_vs_valloss.png) |
|:--:|:--:|
| Training time vs. BLEU Score | Training time vs. Validation Loss |

The table below reports the total number of iterations, the BLEU scores evaluated by the trained model at the last iteration, and the total training time.

|  Step  |    BLEU Score   | Training Time (hours) |
|:------:|:---------------:|:---------------------:|
| 112760 |     0.4228      |        12.8120        |


## Reproduce Experiments

### Python Environment

Here is an instruction to setup the environment for the experiments under a Linux system. The environment is managed by [uv](https://docs.astral.sh/uv/). Please check the uv's documentation for how to setup uv.

```bash
uv sync
```

> [!NOTE]
> A list of dependencies under test is available at [`requirements.txt`](./requirements.txt) for reference.

One has to login to wandb for uploading the metrics before running the experiments.
```
wandb login
```

### Prepare Data

The data preparation process involves downloading the WMT14 datasets, preprocessing the text, and tokenizing the data.

#### Download and Preprocess Raw Data

For WMT14 English-German dataset:
```
sh scripts/preprocess/prepare-wmt14en2de.sh
```

For WMT14 English-French dataset:
```
sh scripts/preprocess/prepare-wmt14en2fr.sh
```

These scripts will:
1. Download the Moses tokenizer scripts
2. Download the WMT14 parallel corpora for the respective language pairs
3. Preprocess the text data (normalize punctuation, remove non-printing characters, tokenize)
4. Split the data into training, validation, and test sets

#### Tokenize and Convert Data

After downloading and preprocessing, run the tokenization script:

For WMT14 English-German dataset:
```
uv run python -m src.preprocess --cfg-list config/data/wmt14_en_de.toml
```

For WMT14 English-French dataset:
```
uv run python -m src.preprocess --cfg-list config/data/wmt14_en_fr.toml
```

This process will:
1. Clean the data further by removing empty lines and fixing encoding issues
2. Train a tokenizer (BPE for En-De, WordPiece for En-Fr) on the training data
3. Convert all data splits to binary format for efficient loading during training
4. Save the processed data in the respective data directories

### Train

The experiments are conducted on a data center using Slurm as the scheduler. To run the training:

```
sbatch -A <PROJECT_ACCOUNT> scripts/train/4xA40.sh config/data/<DATA_CFG> config/train/<MODEL_CFG>
```

Where:
- `<PROJECT_ACCOUNT>` is the Slurm project account
- `<DATA_CFG>` could be:
  1. `wmt14_en_de.toml` for WMT14 English-German dataset
  2. `wmt14_en_fr.toml` for WMT14 English-French dataset
- `<MODEL_CFG>` could be:
  1. `transformer_base.toml` for transformer-base
  2. `transformer_big.toml` for transformer-big

The training scripts support different hardware configurations:
- [`scripts/train/4xA40.sh`](./scripts/train/4xA40.sh): For training on a single node with 4 A40 GPUs
- [`scripts/train/8xA40.sh`](./scripts/train/8xA40.sh): For training on a single node with 8 A40 GPUs
- [`scripts/train/8xA100.sh`](./scripts/train/8xA100.sh): For training on 2 nodes with 4 A100 GPUs each

For systems not based on Slurm, extract the `torchrun` command from the training scripts.

Training features:
- Automatic Mixed Precision (AMP) training for improved speed and memory efficiency
- Distributed training support for multi-GPU setups
- Checkpointing at regular intervals
- Logging with Weights & Biases integration
- Validation during training to monitor performance

### Evaluate

After training, the experiment directories will be under `./log/<SLURM_JOB_ID>/`. To evaluate the trained models by BLEU score:

1. Edit the evaluation configuration file [`config/eval/eval.toml`](./config/eval/eval.toml) to update `exp_dir` to the correct directory
2. Run the evaluation script:
```
uv run python src/eval.py --cfg-list config/eval/eval.toml
```

This will:
- Generate `test_log.csv` in `exp_dir` including BLEU scores for each checkpoint
- Save reference and hypothesis translations in the `inference` subdirectory
- Use beam search (default beam size: 4) with length penalty for generation

### Plot Training Curves

To generate training curves like those shown in [the result section](#results):

```
uv run python -m src.plot <EXP_DIR_OF_REPEAT_RUN_1> <EXP_DIR_OF_REPEAT_RUN_2> ...
```

Where each `<EXP_DIR>` is an experiment directory containing training logs. Multiple directories can be provided to generate error bands showing variance across runs.

This will generate plots in the `image/` directory:
- Training time vs. BLEU Score
- Number of iterations vs. BLEU Score
- Training time vs. Validation Loss
- Number of iterations vs. Validation Loss

## Configuration Files

This repository uses TOML configuration files to manage experiments:

### Data Configuration
- [`config/data/wmt14_en_de.toml`](./config/data/wmt14_en_de.toml): English-German dataset settings
- [`config/data/wmt14_en_fr.toml`](./config/data/wmt14_en_fr.toml): English-French dataset settings

### Model Training Configuration
- [`config/train/transformer_base.toml`](./config/train/transformer_base.toml): Transformer-base model settings
- [`config/train/transformer_big.toml`](./config/train/transformer_big.toml): Transformer-big model settings

### Evaluation Configuration
- [`config/eval/eval.toml`](./config/eval/eval.toml): Evaluation settings

## Customization

To customize the experiments:

1. Modify the existing configuration files or create new ones
2. Adjust model hyperparameters in the training config files
3. Change data preprocessing parameters in the data config files
4. Modify evaluation settings in the eval config file
5. Update training scripts for different hardware configurations

Key customizable parameters include:
- Model architecture (d_model, num_heads, num_layers, dim_feedforward, dropout)
- Training settings (max_epochs, label_smoothing, use_amp)
- Tokenization (vocab_size, tokenizer type)
- Evaluation (beam_size, length_penalty)

## Reference

[1] Vaswani, Ashish, et al. "Attention is all you need." *Advances in neural information processing systems* 30 (2017).

[2] Bojar, Ondřej, et al. "Findings of the 2014 workshop on statistical machine translation." *Proceedings of the ninth workshop on statistical machine translation*. 2014.
