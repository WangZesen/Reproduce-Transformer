# Reproduce [*Attention is All You Need*](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) (PyTorch)

![PyTorch](https://img.shields.io/badge/PyTorch-2.9.0-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Flash Attention](https://img.shields.io/badge/Flash_Attention_2-2.8.3-blue?style=flat&logo=lightning)
![AMP](https://img.shields.io/badge/Precision-AMP-EE4C2C?style=flat&logo=pytorch&logoColor=white)

## Introduction

This repo is an unofficial reproduction of the experiments conducted in the paper "Attention is All You Need" [[1](#reference)] based on PyTorch 2.9.1. The training, validation and test sets are from WMT14 [[2](#reference)]. The idea is to provide a code base for algorithm verification and easy adaptation for this classic and well-studied machine learning task.

The repo adopts the [automatic precision training (AMP)](https://docs.pytorch.org/docs/stable/amp.html) with `bfloat16` by default (*better performance with Ampere GPUs*), and it makes use of [`flash_attn_varlen_func`](https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#20-complete-rewrite-2x-faster) for efficient computation with batches with varying lengths. Compared with the original paper, the model architecture has slight differences: 1. it uses pre-layernorm style instead of post-LN, and 2. except for the embedding layers, the initialization follows the PyTorch's defaults. 

## Reproduced Results

## How to Reproduce

### Environment

The environment is managed by [uv](https://docs.astral.sh/uv/). Please check the uv's documentation for how to setup uv. To reproduce the environment, simply run
```bash
uv sync
```

> Just in case that the wheel for Flash Attention doesn't fit your system, please find available wheels [here](https://flashattn.dev/) and replace the `flash-attn` with the wheel you found.

### Prepare Data

For WMT14 English-German dataset,
```
sh scripts/data/prepare-wmt14en2de.sh
uv run -m src.preprocess --cfg-list configs/data/wmt14_en_de.toml
```

For WMT14 English-French dataset,
```
sh scripts/data/prepare-wmt14en2fr.sh
uv run -m src.preprocess --cfg-list configs/data/wmt14_en_fr.toml
```

### Train

The experiments are conducted on a cluster using Slurm as the scheduler. To run the training with four A40 GPUs, 

```
sbatch -A <PROJECT_ACCOUNT> scripts/train/4xA40.sh config/data/<DATA_CFG> config/model/<MODEL_CFG>
```
where `<PROJECT_ACCOUNT>` is the slurm project account, and `<DATA_CFG>` could be (1) `wmt14_en_de.toml` for WMT14 English-German dataset (2) `wmt14_en_fr.toml` for WMT14 English-French dataset, and `<MODEL_CFG>` could be (1) `base.toml` for transformer-base (~63M parameters) and (2) `big.toml` for transformer-big (~213M parameters).

One can extract the commands in [`scripts/train/4xA40.sh`](./scripts/train/4xA40.sh) to run seperately if the system is not based on slurm.

### Evaluate

After the training, the experiment directories should be under `./log/<SLURM_JOB_ID>/`. To evaluate the trained models by BLEU score, firstly edit the evaluation configuration file [`eval.toml`](./config/eval/eval.toml) to update `exp_dir` to the correct directory. Then run
```
uv run python src/eval.py --cfg-list config/eval/eval.toml
```
It will generate `test_log.csv` in `exp_dir` including the BLEU scores.

One can also plot the training curves like the figures shown in [the result section](#results) by
```
uv run python -m src.plot <EXP_DIR_OF_REPEAT_RUN_1> <...>
```
where `<EXP_DIR_OF_REPEAT_RUN_1>` means the `exp_dir` used in the last step, and one can have multiple runs to generate the error bands.

## Reference

[1] Vaswani, Ashish, et al. "Attention is all you need." *Advances in neural information processing systems* 30 (2017).

[2] Bojar, Ondřej, et al. "Findings of the 2014 workshop on statistical machine translation." *Proceedings of the ninth workshop on statistical machine translation*. 2014.