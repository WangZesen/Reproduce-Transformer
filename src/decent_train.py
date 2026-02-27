import os
import time
import torch
from torch.nn import Module
import torch.distributed as dist
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.profiler import schedule, profile, ProfilerActivity
from src.conf import parse_config, Config, dump_config, Topology
from src.data.dataloader import get_dataloaders, DataLoader
from src.data.dataset import get_datasets
from src.model import TransformerModule
from src.utils import (
    get_optim,
    get_lr_scheduler,
    initialize,
    SmoothedValue,
    gather_statistics,
    get_group_name,
    get_run_name,
)
import wandb
import pandas as pd
from loguru import logger
from typing import Tuple
from dataclasses import dataclass


@dataclass
class CommGroup:
    ranks: list[int]
    weight: float
    group: dist.ProcessGroup


class DecentDP(torch.nn.Module):
    def __init__(
        self,
        base_module: torch.nn.Module,
        topology: Topology,
        comm_block_size_mb: float = 50.0,
    ):
        super().__init__()

        self._module = base_module.cuda()
        self._topology = topology
        self._comm_block_size_mb = comm_block_size_mb

        # acquire distributed info from env variables
        self._world_size = dist.get_world_size()
        self._rank = dist.get_rank()
        self._local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
        self._local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        # sync parameters at start
        self._sync_params()

        # remap parameters into buckets
        self._create_buckets()

        # create communication groups based on topology
        self._create_comm_groups()

        # training step counter
        self._step = 0

        # comm op handles (one per block)
        self._comm_ops: list[dist.Work] = []

    @torch.no_grad()
    def _sync_params(self):
        for param in self._module.parameters():
            dist.broadcast(param.data, src=0)

    @torch.no_grad()
    def _create_buckets(self):
        self._bucket_total_size = sum([self._align(param.numel()) for param in self._module.parameters()])
        self._param_bucket = torch.zeros((self._bucket_total_size), dtype=torch.float32, device="cuda")
        self._comm_bucket = torch.zeros((self._bucket_total_size), dtype=torch.float32, device="cuda")

        # Split buckets into fixed-size blocks (in number of float32 elements)
        block_elems = max(1, int(self._comm_block_size_mb * 1024 * 1024 / 4))
        self._param_blocks: list[torch.Tensor] = []
        self._comm_blocks: list[torch.Tensor] = []
        for start in range(0, self._bucket_total_size, block_elems):
            end = min(start + block_elems, self._bucket_total_size)
            self._param_blocks.append(self._param_bucket[start:end])
            self._comm_blocks.append(self._comm_bucket[start:end])

        offset = 0
        for param in self._module.parameters():
            size = param.numel()
            aligned_size = self._align(size)

            assert param.is_contiguous(), "Parameters must be contiguous"

            # Copy parameter data into the bucket
            chunk = self._param_bucket[offset : offset + size]
            chunk.copy_(param.data.view(-1))

            # Store a view of the chunk back to the parameter for easy access
            param.data = chunk.view_as(param)

            offset += aligned_size

    @torch.no_grad()
    def _create_comm_groups(self):
        match self._topology:
            case Topology.ONE_PEER_RING:
                self._comm_groups = []
                for i in range(0, self._world_size, 2):
                    ranks = sorted([i, (i + 1) % self._world_size])
                    group = dist.new_group(ranks=ranks)
                    if self._rank in ranks:
                        assert group is not None
                        self._comm_groups.append(CommGroup(ranks=ranks, weight=0.5, group=group))
                for i in range(1, self._world_size, 2):
                    ranks = sorted([i, (i + 1) % self._world_size])
                    group = dist.new_group(ranks=ranks)
                    if self._rank in ranks:
                        assert group is not None
                        self._comm_groups.append(CommGroup(ranks=ranks, weight=0.5, group=group))
            case Topology.COMPLETE:
                ranks = list(range(self._world_size))
                group = dist.new_group(ranks=ranks)
                weight = 1.0 / self._world_size
                assert group is not None
                self._comm_groups = [CommGroup(ranks=ranks, weight=weight, group=group)]
            case Topology.ONE_PEER_EXP:
                self._comm_groups = []
                exp = 1
                while exp < self._world_size:
                    for i in range(self._world_size):
                        j = i ^ exp
                        if i < j and j < self._world_size:
                            ranks = sorted([i, j])
                            group = dist.new_group(ranks=ranks)
                            if self._rank in ranks:
                                assert group is not None
                                self._comm_groups.append(CommGroup(ranks=ranks, weight=0.5, group=group))
                    exp <<= 1
            case _:
                raise ValueError(f"Unsupported topology: {self._topology}")

    @torch.no_grad()
    def mix(self, gamma: float = 1.0):
        for op, param_block, comm_block in zip(self._comm_ops, self._param_blocks, self._comm_blocks):
            op.wait()
            param_block.lerp_(comm_block, gamma)
        self._comm_ops = []

    @torch.no_grad()
    def start_comm(self):
        comm_group = self._comm_groups[self._step % len(self._comm_groups)]
        self._comm_ops = []
        for param_block, comm_block in zip(self._param_blocks, self._comm_blocks):
            comm_block.copy_(param_block).mul_(comm_group.weight)
            op = dist.all_reduce(
                comm_block,
                op=dist.ReduceOp.SUM,
                group=comm_group.group,
                async_op=True,
            )
            self._comm_ops.append(op)  # type: ignore
        self._step += 1

    @torch.no_grad()
    def global_avg(self) -> float:
        self._backup = self._param_bucket.clone()
        dist.all_reduce(self._param_bucket, op=dist.ReduceOp.AVG)
        d2c = torch.norm(self._param_bucket - self._backup)
        dist.all_reduce(d2c, op=dist.ReduceOp.AVG)
        return d2c.item()

    @torch.no_grad()
    def restore(self):
        self._param_bucket.copy_(self._backup)
        del self._backup

    @torch.no_grad()
    def sync_buffers(self):
        for buffer in self._module.buffers():
            if buffer.dtype in [torch.float16, torch.float32, torch.float64]:
                dist.all_reduce(buffer, op=dist.ReduceOp.AVG)
            else:
                dist.broadcast(buffer, src=0)

    @torch.no_grad()
    def extract_momentum(self, optim: torch.optim.Optimizer) -> torch.Tensor:
        if isinstance(optim, torch.optim.SGD):
            momentum_buffer = []
            for param in self._module.parameters():
                if param in optim.state and "momentum_buffer" in optim.state[param]:
                    momentum_buffer.append(optim.state[param]["momentum_buffer"].view(-1))
                    if self._align(param.numel()) > param.numel():
                        padding = self._align(param.numel()) - param.numel()
                        momentum_buffer.append(torch.zeros(padding, device=param.device))
                else:
                    raise ValueError(
                        "Momentum buffer not found for a parameter. Make sure to call this after the first optimization step."
                    )
            return torch.cat(momentum_buffer)
        elif isinstance(optim, torch.optim.Adam) or isinstance(optim, torch.optim.AdamW):
            momentum_buffer = []
            for param in self._module.parameters():
                if param in optim.state and "exp_avg" in optim.state[param]:
                    momentum_buffer.append(optim.state[param]["exp_avg"].view(-1))
                    if self._align(param.numel()) > param.numel():
                        padding = self._align(param.numel()) - param.numel()
                        momentum_buffer.append(torch.zeros(padding, device=param.device))
                else:
                    raise ValueError(
                        "Exp_avg buffer not found for a parameter. Make sure to call this after the first optimization step."
                    )
            return torch.cat(momentum_buffer)
        else:
            raise ValueError("Unsupported optimizer type for momentum extraction")

    def _align(self, size: int):
        return ((size + 31) // 32) * 32

    def forward(self, *args, **kwargs):
        return self._module(*args, **kwargs)

    def parameters(self, recurse: bool = True):
        yield from self._module.parameters(recurse=recurse)

    def named_parameters(self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True):
        yield from self._module.named_parameters(prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate)

    def train(self, mode: bool = True):
        self._module.train(mode)
        return self

    def eval(self):
        self._module.eval()
        return self


def load_checkpoint(
    model: Module, optimizer: Optimizer, lr_scheduler: LRScheduler, checkpoint_dir: str
) -> Tuple[int, int, float]:
    raise NotImplementedError("Checkpoint loading is not implemented yet")


def train_epoch(
    cfg: Config,
    epoch: int,
    step: int,
    model: DecentDP,
    train_ds: DataLoader,
    optimizer: Optimizer,
    lr_scheduler: LRScheduler,
    criterion: Module,
    profiler,
):
    if dist.is_initialized():
        dist.barrier()
    torch.cuda.synchronize()

    start_time = time.time()
    last_log_time = start_time
    last_log_idx = 0
    model.train()
    total_step = len(train_ds)
    loss_metric = SmoothedValue(cfg.train.log.log_freq)
    tpb_metric = SmoothedValue(cfg.train.log.log_freq)

    if cfg.train.network.rank == 0:
        logger.info(f"[Train Epoch {epoch + 1}] {total_step} steps")

    for batch_idx, batch in enumerate(train_ds):
        src = batch["src"].cuda(non_blocking=True)
        tgt = batch["tgt"].cuda(non_blocking=True)
        src_pos_ids = batch["src_pos_ids"].cuda(non_blocking=True)
        tgt_pos_ids = batch["tgt_pos_ids"].cuda(non_blocking=True)
        cu_src_lens = batch["cu_src_lens"].cuda(non_blocking=True)
        cu_tgt_lens = batch["cu_tgt_lens"].cuda(non_blocking=True)
        max_src_len = batch["max_src_len"]
        max_tgt_len = batch["max_tgt_len"]
        label = batch["label"].cuda(non_blocking=True)
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logit = model(src, tgt, src_pos_ids, tgt_pos_ids, cu_src_lens, cu_tgt_lens, max_src_len, max_tgt_len)
            loss = criterion(logit, label)
        loss.backward()
        model.mix()
        optimizer.step()
        lr_scheduler.step()
        model.start_comm()

        # update metrics
        loss_metric.update(loss.item())
        tpb_metric.update((src.size(0) + tgt.size(0) + batch["batch_size"]) * cfg.train.network.world_size)
        lr = lr_scheduler.get_last_lr()[0]
        step += 1

        if (cfg.train.network.rank == 0) and (step % cfg.train.log.log_freq == 0):
            current_log_time = time.time()
            elapsed_time = current_log_time - last_log_time
            logger.info(
                f"step: {step} ({elapsed_time / (batch_idx - last_log_idx + 1):.3f} s/it), "
                f"loss: {loss_metric.avg:.6f}, lr: {lr:.6f}, tpb: {tpb_metric.avg:.1f}",
                f"mem: {torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024:.2f} GB",
            )
            last_log_time = current_log_time
            last_log_idx = batch_idx
            if cfg.train.log.wandb_on:
                wandb.log(
                    {
                        "train_loss": loss_metric.avg,
                        "learning_rate": lr,
                        "tokens_per_batch": tpb_metric.avg,
                    },
                    step=step,
                )
        profiler.step()

    torch.cuda.synchronize()
    return loss_metric.global_avg, step, time.time() - start_time


@torch.no_grad()
def val_epoch(
    cfg: Config,
    model: DecentDP,
    epoch: int,
    val_ds: DataLoader,
    criterion: Module,
):
    model.eval()
    total_loss = 0.0
    if cfg.train.network.rank == 0:
        logger.info(f"[Valid Epoch {epoch + 1}] {len(val_ds)} steps")

    for batch in val_ds:
        src = batch["src"].cuda(non_blocking=True)
        tgt = batch["tgt"].cuda(non_blocking=True)
        src_pos_ids = batch["src_pos_ids"].cuda(non_blocking=True)
        tgt_pos_ids = batch["tgt_pos_ids"].cuda(non_blocking=True)
        cu_src_lens = batch["cu_src_lens"].cuda(non_blocking=True)
        cu_tgt_lens = batch["cu_tgt_lens"].cuda(non_blocking=True)
        max_src_len = batch["max_src_len"]
        max_tgt_len = batch["max_tgt_len"]
        label = batch["label"].cuda(non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logit = model(src, tgt, src_pos_ids, tgt_pos_ids, cu_src_lens, cu_tgt_lens, max_src_len, max_tgt_len)
            loss = criterion(logit, label)

        total_loss += loss.item()
    return total_loss / len(val_ds)


def main():
    cfg = parse_config()
    assert cfg.train is not None, "Training configuration must be provided"
    assert cfg.train.backend.name == "decent_dp", "DecentDP backend must be used for this script"

    # Set up distributed training and random seeds
    initialize()
    torch.manual_seed(cfg.train.seed + cfg.train.network.rank)

    # Load datasets and dataloaders
    tokenizer_dir = cfg.data.output_dir
    train_dataset, val_dataset = get_datasets(cfg, tokenizer_dir)
    train_ds, val_ds = get_dataloaders(cfg, train_dataset, val_dataset)

    # Initialize model, optimizer, and learning rate scheduler
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=cfg.train.label_smoothing)
    model = TransformerModule(
        cfg.data.tokenizer.vocab_size,
        cfg.train.model.d_model,
        cfg.train.model.num_heads,
        cfg.train.model.num_layers,
        cfg.train.model.dim_feedforward,
        cfg.train.model.dropout,
    )
    model = model.cuda()
    model.forward = torch.compile(model.forward)
    if cfg.train.network.rank == 0:
        if cfg.train.log.wandb_on:
            wandb.init(
                project=cfg.train.log.wandb_project,
                name=get_run_name(cfg),
                group=get_group_name(cfg),
                config=cfg.model_dump(),
                save_code=True,
                dir=os.environ.get("TMPDIR", "/tmp"),
            )
        logger.info(cfg.model_dump_json(indent=4))
        logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters())}")
        dump_config(cfg, os.path.join(cfg.train.log.log_dir, "config.toml"))

    optimizer = get_optim(cfg, model)
    lr_scheduler = get_lr_scheduler(cfg, optimizer)

    # Decentralized Data Parallel
    assert cfg.train.network.world_size > 1
    model = DecentDP(model, topology=cfg.train.backend.topology, comm_block_size_mb=cfg.train.backend.comm_block_size_mb)

    # Training loop
    global_step = 0
    start_epoch = 0
    total_train_time = 0.0

    if cfg.train.checkpoint_dir:
        global_step, start_epoch, total_train_time = load_checkpoint(
            model, optimizer, lr_scheduler, cfg.train.checkpoint_dir
        )

    train_log = pd.DataFrame(
        columns=[
            "epoch",
            "step",
            "train_loss",
            "val_loss",
            "d2c",
            "time",
            "checkpoint_dir",
        ]
    )

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=128, warmup=2, active=8, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            os.path.join(cfg.train.log.log_dir, "tb_trace"), worker_name=f"worker_{cfg.train.network.rank:02d}"
        ),
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
    ) as p:
        for epoch in range(start_epoch, cfg.train.max_epochs):
            train_ds.batch_sampler.set_epoch(epoch)  # type: ignore
            train_loss, global_step, train_time = train_epoch(
                cfg,
                epoch,
                global_step,
                model,
                train_ds,
                optimizer,
                lr_scheduler,
                criterion,
                p,
            )
            total_train_time += train_time
            d2c = model.global_avg()

            val_loss = val_epoch(cfg, model, epoch, val_ds, criterion)
            train_loss, val_loss, d2c = gather_statistics(train_loss, val_loss, d2c)

            if cfg.train.network.rank == 0:
                logger.info(
                    f"[Epoch {epoch + 1}] train loss: {train_loss:.6f}, val loss: {val_loss:.6f}, d2c: {d2c:.6f}, "
                    f"epoch time: {train_time:.2f} s, total train time: {total_train_time:.2f} s"
                )
                checkpoint_dir = ""
                if ((epoch + 1) % cfg.train.log.checkpoint_freq == 0) or (epoch == cfg.train.max_epochs - 1):
                    checkpoint_dir = os.path.join(cfg.train.log.log_dir, "checkpoint", f"epoch_{epoch + 1}.pt")
                    os.makedirs(os.path.dirname(checkpoint_dir), exist_ok=True)
                    # TODO: save optimizer and lr scheduler states for resuming training
                    torch.save(
                        {
                            "model_state_dict": model._module.state_dict(),
                            "global_step": global_step,
                            "epoch": epoch + 1,
                            "total_train_time": total_train_time,
                        },
                        checkpoint_dir,
                    )
                    logger.info(f"Saved checkpoint to {checkpoint_dir}")

                train_log.loc[epoch - start_epoch] = [
                    epoch + 1,
                    global_step,
                    train_loss,
                    val_loss,
                    d2c,
                    total_train_time,
                    checkpoint_dir,
                ]
                train_log.to_csv(os.path.join(cfg.train.log.log_dir, "train_log.csv"), index=False)

                if cfg.train.log.wandb_on:
                    wandb.log(
                        {
                            "epoch_train_loss": train_loss,
                            "val_loss": val_loss,
                            "d2c": d2c,
                            "epoch_time": train_time,
                            "total_train_time": total_train_time,
                            "epoch": epoch + 1,
                        },
                        step=global_step,
                    )

            model.restore()

            if dist.is_initialized():
                dist.barrier()
    dist.destroy_process_group()
    if (cfg.train.log.wandb_on) and (cfg.train.network.rank == 0):
        wandb.finish()


if __name__ == "__main__":
    main()
