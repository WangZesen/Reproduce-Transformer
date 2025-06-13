import os
import argparse
import tomllib
import hashlib
from typing import Optional, Tuple
from pydantic import BaseModel, Field, computed_field, ConfigDict
from loguru import logger
from functools import cached_property

PROJECT_DIR = os.path.relpath(os.path.join(os.path.dirname(__file__), '..'), '.')
class SPECIAL_TOKENS:
    PAD = '[PAD]'
    UNK = '[UNK]'
    SOS = '[SOS]'
    EOS = '[EOS]'
    ALL = [PAD, UNK, SOS, EOS]


class _BaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Tokenizer(_BaseModel):
    model: str = Field(default='bpe')
    vocab_size: int = Field(default=37120)
    min_freq: int = Field(default=2)


class Data(_BaseModel):
    data_dir: str = Field(default=os.path.join(PROJECT_DIR, 'data', 'wmt14_en_de'))
    src_lang: str = Field(default='en')
    tgt_lang: str = Field(default='de')
    truncate: int = Field(default=156)
    tokenizer: Tokenizer = Field(default_factory=Tokenizer)
    batch_efficiency: float = Field(default=0.35)

    @computed_field(repr=False)
    @property
    def tag(self) -> str:
        return hashlib.md5(str(self.__repr__()).encode()).hexdigest()[:10]

    @computed_field(repr=False)
    @property
    def output_dir(self) -> str:
        return os.path.join(self.data_dir, self.tag)


class Reproduce(_BaseModel):
    seed: int = Field(default=810975)


class Network(_BaseModel):
    @computed_field(repr=False)
    @property
    def world_size(self) -> int:
        return int(os.environ.get('WORLD_SIZE', '1'))

    @computed_field(repr=False)
    @property
    def rank(self) -> int:
        return int(os.environ.get('RANK', '0'))
    
    @computed_field(repr=False)
    @property
    def local_rank(self) -> int:
        return int(os.environ.get('LOCAL_RANK', '0'))
    
    @computed_field(repr=False)
    @property
    def local_world_size(self) -> int:
        return int(os.environ.get('LOCAL_WORLD_SIZE', '1'))


class Model(_BaseModel):
    arch: str = Field(default='transformer')
    d_model: int = Field(default=512)
    num_heads: int = Field(default=8)
    num_layers: int = Field(default=6)
    dim_feedforward: int = Field(default=2048)
    dropout: float = Field(default=0.1)


class Optimizer(_BaseModel):
    name: str = Field(default='adam')
    lr: float = Field(default=0.0007)
    betas: Tuple[float, float] = Field(default=(0.9, 0.98))
    eps: float = Field(default=1e-9)


class LRScheduler(_BaseModel):
    type: str = Field(default='inverse_sqrt')
    warmup_steps: int = Field(default=4000)
    warmup_decay: float = Field(default=0.01)


class Log(_BaseModel):
    log_freq: int = Field(default=250)
    wandb_on: bool = Field(default=False)
    wandb_project: str = Field(default='Reproduce-Transformer')
    checkpoint_freq: int = Field(default=2)

    @computed_field
    @cached_property
    def job_id(self) -> str:
        return os.environ.get('JOB_ID', '0')

    @computed_field
    @property
    def log_dir(self) -> str:
        return os.path.join(PROJECT_DIR, 'log', self.job_id)


class Train(_BaseModel):
    max_tokens_per_batch: int = Field(default=25000)
    label_smoothing: float = Field(default=0.1)
    checkpoint_dir: Optional[str] = None
    max_epochs: int = Field(default=20)
    use_amp: bool = Field(default=False)
    reproduce: Reproduce = Field(default_factory=Reproduce)
    network: Network = Field(default_factory=Network)
    model: Model = Field(default_factory=Model)
    optim: Optimizer = Field(default_factory=Optimizer)
    lr_scheduler: LRScheduler = Field(default_factory=LRScheduler)
    log: Log = Field(default_factory=Log)

    @computed_field(repr=False)
    @property
    def max_tokens_per_local_batch(self) -> int:
        return self.max_tokens_per_batch // self.network.world_size

    @max_tokens_per_local_batch.setter
    def max_tokens_per_local_batch(self, value: int):
        self.max_tokens_per_batch = value * self.network.world_size


class Eval(_BaseModel):
    exp_dir: Optional[str] = Field(default=None)
    beam_size: int = Field(default=4)
    length_penalty: float = Field(default=0.6)
    tolerance: int = Field(default=50)


class Config(_BaseModel):
    data: Data = Field(default_factory=Data)
    train: Train = Field(default_factory=Train)
    eval: Eval = Field(default_factory=Eval)


def _load_toml(config_dir):
    with open(config_dir, "rb") as f:
        config = tomllib.load(f)
    return config


def _merge(a: dict, b: dict, path=[]):
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                _merge(a[key], b[key], path + [str(key)])
            elif a[key] != b[key]:
                a[key] = b[key]
                logger.warning(
                    f"Overriding {'.'.join(path + [str(key)])} with {b[key]}"
                )
        else:
            a[key] = b[key]
    return a


def parse_config() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg-list",
        nargs="+",
        required=False,
        help="List of configurations to override",
    )
    args = parser.parse_args()

    if args.cfg_list is None:
        return Config()
    raw_cfgs = [_load_toml(cfg_dir) for cfg_dir in args.cfg_list]
    raw_cfg = {}
    for cfg in raw_cfgs:
        raw_cfg = _merge(raw_cfg, cfg)

    return Config.model_validate(raw_cfg)
