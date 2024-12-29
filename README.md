# Reproduce [*Attention is All You Need*](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) (PyTorch)

## Introduction

This repo is an unofficial reproduction of the experiments conducted in the paper "Attention is All You Need" [[1](#reference)] based on PyTorch 2.3.0 (latest stable version by the time of setting up this repo). The training, validation and test sets are from WMT14 [[2](#reference)].

## Results

### Transformer (base) on WMT14 English-German

For the transformer (base) trained in WMT14 English-German dataset, the reported BLEU score in the original paper is 27.3 which is shown by the red dashed line in the following figures.

For the experiment, the model is trained by four A40 GPUs. The reproduced results are from the average of 3 runs and the error bands represnet the interval of $\pm2$ standard deviations.

| ![space-1.jpg](./doc/transformer-base_en-de/step_vs_bleu.png) | ![space-1.jpg](./doc/transformer-base_en-de/step_vs_valloss.png)
|:--:| :--: |
| # of Iterations vs. BLEU Score | # of Iterations vs. Val Loss |

| ![space-1.jpg](./doc/transformer-base_en-de/time_vs_bleu.png) | ![space-1.jpg](./doc/transformer-base_en-de/time_vs_valloss.png)
|:--:| :--: |
| Training time vs. BLEU Score | Training time vs. Val Loss |

The table below reports the total number of iterations, the BLEU scores evaluated by the trained model at the last iteration, and the total training time.

|  Step  |        AMP         |   BLEU Score    | Training Time (hours) |
|:------:|:------------------:|:---------------:|:---------------------:|
| 112760 | :negative_squared_cross_mark: | 0.2760 ± 0.0013 |    7.7557 ± 0.0066    |
| 112760 | :white_check_mark: | 0.2758 ± 0.0022 |    5.2069 ± 0.0010    |


### Transformer (big) on WMT14 English-French
For the transformer (base) trained in WMT14 English-German dataset, the reported BLEU score in the original paper is 41.0 which is shown by the red dashed line in the following figures.

For the experiment, the model is trained by eight A100 GPUs. The reproduced results are from a single run.

| ![space-1.jpg](./doc/transformer-big_en-fr/step_vs_bleu.png) | ![space-1.jpg](./doc/transformer-big_en-fr/step_vs_valloss.png)
|:--:| :--: |
| # of Iterations vs. BLEU Score | # of Iterations vs. Val Loss |

| ![space-1.jpg](./doc/transformer-big_en-fr/time_vs_bleu.png) | ![space-1.jpg](./doc/transformer-big_en-fr/time_vs_valloss.png)
|:--:| :--: |
| Training time vs. BLEU Score | Training time vs. Val Loss |

The table below reports the total number of iterations, the BLEU scores evaluated by the trained model at the last iteration, and the total training time.

|  Step  |    BLEU Score   | Training Time (hours) |
|:------:|:---------------:|:---------------------:|
| 112760 |     0.4228      |        12.8120        |


## Reproduce Experiments

### Python Environment

Here is an instruction to setup a conda environment for the experiments.

```bash
conda create -n reproduce python=3.12
conda activate reproduce
# install the latest version of PyTorch
pip3 install torch torchvision torchaudio
# install other dependencies
pip install wandb seaborn evaluate tokenizers loguru scipy tqdm tomli-w pydantic
```

> [!NOTE]
> A list of dependencies under test is available at [`requirements.txt`](./requirements.txt) for reference.

One has to login to wandb for uploading the metrics before runing the experiments.
```
wandb login
```

### Prepare Data

For WMT14 English-German dataset,
```
sh script/preprocess/prepare-wmt14en2de.sh
python -m src.preprocess --data-cfg config/data/wmt14_en_de.toml
```

For WMT14 English-French dataset,
```
sh script/preprocess/prepare-wmt14en2fr.sh
python -m src.preprocess --data-cfg config/data/wmt14_en_fr.toml
```

### Train

The experiments are conducted on a data center using Slurm as the scheduler. To run the training with four A40 GPUs, 

```
sbatch -A <PROJECT_ACCOUNT> script/train/4xA40.sh $(which torchrun) config/data/<DATA_CFG> config/train/<MODEL_CFG>
```
where `<PROJECT_ACCOUNT>` is the slurm project account, and `<DATA_CFG>` could be (1) `wmt14_en_de.toml` for WMT14 English-German dataset (2) `wmt14_en_fr.toml` for WMT14 English-French dataset, and `<MODEL_CFG>` could be (1) `transformer_base.toml` for transformer-base and (2) `transformer_big.toml` for transformer-big.

One can extract the command in [`script/train/4xA40.sh`](./script/train/4xA40.sh) to run seperately if the system is not based on slurm.

### Evaluate

After the training, the experiment directories should be under `./log/<SLURM_JOB_ID>/`. To evaluate the trained models by BLEU score, firstly edit the evaluation configuration file [`eval.toml`](./config/eval/eval.toml) to update `exp_dir` to the correct directory. Then run
```
python src/eval.py --eval-cfg config/eval/eval.toml
```
It will generate `test_log.csv` in `exp_dir` including the BLEU scores.

One can also plot the training curves like the figures shown in [the result section](#results) by
```
python -m src.plot <EXP_DIR_OF_REPEAT_RUN_1> <...>
```
where `<EXP_DIR_OF_REPEAT_RUN_1>` means the `exp_dir` used in the last step, and one can have multiple runs to generate the error bands.

## Reference

[1] Vaswani, Ashish, et al. "Attention is all you need." *Advances in neural information processing systems* 30 (2017).

[2] Bojar, Ondřej, et al. "Findings of the 2014 workshop on statistical machine translation." *Proceedings of the ninth workshop on statistical machine translation*. 2014.