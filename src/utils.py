import os
from statistics import mean
import torch.distributed as dist
from collections import deque
from loguru import logger
import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler
from typing import List, Tuple, Callable
from src.conf import Config


class SmoothedValue:
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{avg:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.window_size = window_size
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append((value, n))
        self.count += n
        self.total += value * n

    @property
    def avg(self) -> float:
        count = 0.0
        total = 0.0
        for i in reversed(range(len(self.deque))):
            if count + self.deque[i][1] >= self.window_size:
                total += self.deque[i][0] * (self.window_size - count)
                count = self.window_size
                break
            count += self.deque[i][1]
            total += self.deque[i][0] * self.deque[i][1]
        if count == 0:
            return 0
        return total / count

    @property
    def global_avg(self):
        return self.total / self.count

    def __str__(self):
        return self.fmt.format(avg=self.avg, global_avg=self.global_avg)


def initialize_dist() -> None:
    if ("WORLD_SIZE" in os.environ) and dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        if dist.get_rank() == 0:
            logger.info(f"Initialized the process group with {dist.get_world_size()} processes.")
        dist.barrier()


def get_optim_fn(cfg: Config) -> Callable[[List[Tuple[str, torch.Tensor]]], Optimizer]:
    match cfg.train.optim.name.lower():
        case "adam":

            def optim_fn(params: List[Tuple[str, torch.Tensor]]) -> Optimizer:
                return torch.optim.Adam(
                    [param for _, param in params],
                    lr=cfg.train.optim.lr,
                    betas=cfg.train.optim.betas,
                    eps=cfg.train.optim.eps,
                )

            return optim_fn
        case _:
            raise ValueError(f"Unsupported optimizer: {cfg.train.optim.name}")


def get_optim(cfg: Config, model: Module) -> Optimizer:
    optim_fn = get_optim_fn(cfg)
    params = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    return optim_fn(params)  # type: ignore


def get_lr_scheduler_fn(cfg: Config) -> Callable[[Optimizer], LRScheduler]:
    match cfg.train.lr_scheduler.type.lower():
        case "inverse_sqrt":

            def lr_scheduler_fn(optimizer: Optimizer) -> LRScheduler:
                def lr_lambda(step: int) -> float:
                    if step < cfg.train.lr_scheduler.warmup_steps:
                        return (1 - cfg.train.lr_scheduler.warmup_decay) * (
                            step + 1
                        ) / cfg.train.lr_scheduler.warmup_steps + cfg.train.lr_scheduler.warmup_decay
                    else:
                        return (cfg.train.lr_scheduler.warmup_steps**0.5) * ((step + 1) ** (-0.5))

                return LambdaLR(optimizer, lr_lambda=lr_lambda)

            return lr_scheduler_fn
        case _:
            raise ValueError(f"Unsupported lr_scheduler: {cfg.train.lr_scheduler.type}")


def get_lr_scheduler(cfg: "Config", optimizer: Optimizer) -> LRScheduler:
    lr_scheduler_fn = get_lr_scheduler_fn(cfg)
    return lr_scheduler_fn(optimizer)


def gather_statistics(
    train_loss: float, train_perplexity: float, val_loss: float, val_perplexity: float
) -> List[float]:
    log = [train_loss, train_perplexity, val_loss, val_perplexity]
    if dist.is_available() and dist.is_initialized():
        object_list = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(object_list, log)
        for i in range(len(log)):
            log[i] = mean([x[i] for x in object_list])  # type: ignore
    return log


def get_group_name(cfg: Config) -> str:
    """Generate a group name based on the configuration."""
    name = ""
    if cfg.train.model.d_model == 512:
        name += "base - "
    else:
        name += "big - "

    if cfg.train.backend.name == "decent_dp":
        if cfg.train.optim.name == "adam":
            name += f"d-{cfg.train.optim.name.lower()}"
        elif cfg.train.optim.name == "accumadam":
            name += f"d-{cfg.train.optim.name.lower()}@{cfg.train.optim.accum_iters}"
        else:
            raise ValueError(f"Unsupported optimizer for decent_dp: {cfg.train.optim.name}")
        if cfg.train.backend.adaptive_consensus_omega > 0:
            if cfg.train.backend.omega != 1.0:
                name += f" - ac {cfg.train.backend.adaptive_consensus_omega}@{cfg.train.backend.ac_start_step}@{cfg.train.backend.adaptive_consensus_p}@o{cfg.train.backend.omega:.2f} - {cfg.train.backend.topology}"
            else:
                name += f" - ac {cfg.train.backend.adaptive_consensus_omega}@{cfg.train.backend.ac_start_step}@{cfg.train.backend.adaptive_consensus_p} - {cfg.train.backend.topology}"
    else:
        name += f"{cfg.train.optim.name.lower()}"

    return name


def get_run_name(cfg: Config) -> str:
    """Generate a unique run name based on the configuration."""
    name = get_group_name(cfg)
    name += f" - {cfg.train.log.job_id} - {cfg.train.seed}"
    return name
