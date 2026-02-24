import os
import glob
import argparse
import tomllib
import hashlib
from typing import Literal, Optional, Tuple, Union
from pydantic import BaseModel, Field, computed_field, ConfigDict
from loguru import logger
from functools import cached_property
import tomli_w
from enum import Enum

PROJECT_DIR = os.path.relpath(os.path.join(os.path.dirname(__file__), ".."), ".")


class SPECIAL_TOKENS:
    PAD = "[PAD]"
    UNK = "[UNK]"
    SOS = "[SOS]"
    EOS = "[EOS]"
    ALL = [PAD, UNK, SOS, EOS]


class Topology(str, Enum):
    ONE_PEER_RING = "1p-ring"
    ONE_PEER_EXP = "1p-exp"
    COMPLETE = "complete"


class _BaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Tokenizer(_BaseModel):
    model: str = Field(default="bpe")
    vocab_size: int = Field(default=37120)
    min_freq: int = Field(default=2)


class Data(_BaseModel):
    data_dir: str = Field(default=os.path.join(PROJECT_DIR, "data", "wmt14_en_de"))
    src_lang: str = Field(default="en")
    tgt_lang: str = Field(default="de")
    truncate: int = Field(default=156)
    tokenizer: Tokenizer = Field(default_factory=Tokenizer)

    @computed_field(repr=False)
    @property
    def tag(self) -> str:
        return hashlib.md5(str(self.__repr__()).encode()).hexdigest()[:10]

    @computed_field(repr=False)
    @property
    def output_dir(self) -> str:
        return os.path.join(self.data_dir, self.tag)


class Network(_BaseModel):
    @computed_field(repr=False)
    @property
    def world_size(self) -> int:
        return int(os.environ.get("WORLD_SIZE", "1"))

    @computed_field(repr=False)
    @property
    def rank(self) -> int:
        return int(os.environ.get("RANK", "0"))

    @computed_field(repr=False)
    @property
    def local_rank(self) -> int:
        return int(os.environ.get("LOCAL_RANK", "0"))

    @computed_field(repr=False)
    @property
    def local_world_size(self) -> int:
        return int(os.environ.get("LOCAL_WORLD_SIZE", "1"))


class Model(_BaseModel):
    d_model: int = Field(default=512)
    num_heads: int = Field(default=8)
    num_layers: int = Field(default=6)
    dim_feedforward: int = Field(default=2048)
    dropout: float = Field(default=0.1)


class AdamConfig(_BaseModel):
    name: Literal["adam"] = Field(default="adam")
    lr: float = Field(default=0.0007)
    betas: Tuple[float, float] = Field(default=(0.9, 0.98))
    eps: float = Field(default=1e-9)


OPTIMIZERS = Union[AdamConfig]


class LRScheduler(_BaseModel):
    type: str = Field(default="inverse_sqrt")
    warmup_steps: int = Field(default=4000)
    warmup_decay: float = Field(default=0.01)


class Log(_BaseModel):
    log_freq: int = Field(default=250)
    wandb_on: bool = Field(default=True)
    wandb_project: str = Field(default="reproduce-transformer")
    checkpoint_freq: int = Field(default=1)
    with_states: bool = Field(
        default=False, description="Whether to save optimizer and lr scheduler states in checkpoints."
    )

    @computed_field
    @cached_property
    def job_id(self) -> str:
        if "JOB_ID" in os.environ:
            return os.environ["JOB_ID"]
        elif "SLURM_JOB_ID" in os.environ:
            return os.environ["SLURM_JOB_ID"]
        else:
            return "local_job"

    @computed_field
    @property
    def log_dir(self) -> str:
        return os.path.join(PROJECT_DIR, "log", self.job_id)


class PyTorchDDPBackend(_BaseModel):
    name: Literal["pytorch_ddp"] = Field(default="pytorch_ddp")


class DecentDPBackend(_BaseModel):
    name: Literal["decent_dp"] = Field(default="decent_dp")
    topology: Topology = Field(default=Topology.ONE_PEER_RING)


BACKENDS = Union[PyTorchDDPBackend, DecentDPBackend]


class Train(_BaseModel):
    max_tokens_per_batch: int = Field(default=25000)
    label_smoothing: float = Field(default=0.1)
    checkpoint_dir: Optional[str] = None
    max_epochs: int = Field(default=20)
    seed: int = Field(default=42)
    network: Network = Field(default_factory=Network)
    model: Model = Field(default_factory=Model)
    optim: OPTIMIZERS = Field(default_factory=AdamConfig, discriminator="name")
    lr_scheduler: LRScheduler = Field(default_factory=LRScheduler)
    log: Log = Field(default_factory=Log)
    backend: BACKENDS = Field(default_factory=PyTorchDDPBackend, discriminator="name")

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
                logger.warning(f"Overriding {'.'.join(path + [str(key)])} with {b[key]}")
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


def parse_eval_config() -> Config:
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

    cfg = Config.model_validate(raw_cfg)

    if cfg.eval.exp_dir is None:
        raise ValueError("Please specify the experiment directory in the configuration file.")

    other_cfgs = sorted(
        glob.glob(os.path.join(cfg.eval.exp_dir, "config", "*.toml")), key=lambda x: int(x.split("/")[-1].split("_")[0])
    )
    all_cfgs = other_cfgs + args.cfg_list

    raw_cfgs = [_load_toml(cfg_dir) for cfg_dir in all_cfgs]
    raw_cfg = {}
    for cfg in raw_cfgs:
        raw_cfg = _merge(raw_cfg, cfg)

    return Config.model_validate(raw_cfg)


def dump_config(cfg: Config, output_dir: str) -> None:
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    with open(output_dir, "wb") as f:
        tomli_w.dump(cfg.model_dump(exclude_none=True), f)
