import os
import time
import torch
from torch.nn import Module
import torch.distributed as dist
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import schedule, profile, ProfilerActivity
from src.conf import parse_config, Config, dump_config
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


def load_checkpoint(
    model: Module, optimizer: Optimizer, lr_scheduler: LRScheduler, checkpoint_dir: str
) -> Tuple[int, int, float]:
    state = torch.load(checkpoint_dir)
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optim_state_dict"])
    lr_scheduler.load_state_dict(state["scheduler_state_dict"])
    logger.info(f"Loaded checkpoint from {checkpoint_dir}")
    return state["global_step"], state["epoch"], state["total_train_time"]


def train_epoch(
    cfg: Config,
    epoch: int,
    step: int,
    model: Module,
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
        labels = batch["label"].cuda(non_blocking=True)
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(src, tgt, src_pos_ids, tgt_pos_ids, cu_src_lens, cu_tgt_lens, max_src_len, max_tgt_len)
            loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()
        lr_scheduler.step()

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
    model: Module,
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
        labels = batch["label"].cuda(non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(src, tgt, src_pos_ids, tgt_pos_ids, cu_src_lens, cu_tgt_lens, max_src_len, max_tgt_len)
            loss = criterion(logits, labels)

        total_loss += loss.item()
    return total_loss / len(val_ds)


def main():
    cfg = parse_config()
    assert cfg.train is not None, "Training configuration must be provided"
    assert cfg.train.backend.name == "pytorch_ddp", "PyTorch DDP backend must be used for this script"

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
    model.forward = torch.compile(model.forward, dynamic=True)  
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

    # DDP
    if cfg.train.network.world_size > 1:
        model = DDP(model, broadcast_buffers=False, gradient_as_bucket_view=True)

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

            val_loss = val_epoch(cfg, model, epoch, val_ds, criterion)
            train_loss, val_loss = gather_statistics(train_loss, val_loss)

            if cfg.train.network.rank == 0:
                logger.info(
                    f"[Epoch {epoch + 1}] train loss: {train_loss:.6f}, val loss: {val_loss:.6f}, "
                    f"epoch time: {train_time:.2f} s, total train time: {total_train_time:.2f} s"
                )
                checkpoint_dir = ""
                if ((epoch + 1) % cfg.train.log.checkpoint_freq == 0) or (epoch == cfg.train.max_epochs - 1):
                    checkpoint_dir = os.path.join(cfg.train.log.log_dir, "checkpoint", f"epoch_{epoch + 1}.pt")
                    os.makedirs(os.path.dirname(checkpoint_dir), exist_ok=True)
                    if cfg.train.log.with_states or (epoch == cfg.train.max_epochs - 1):
                        torch.save(
                            {
                                "model_state_dict": model.module.state_dict()
                                if isinstance(model, DDP)
                                else model.state_dict(),
                                "optim_state_dict": optimizer.state_dict(),
                                "scheduler_state_dict": lr_scheduler.state_dict(),
                                "global_step": global_step,
                                "epoch": epoch + 1,
                                "total_train_time": total_train_time,
                            },
                            checkpoint_dir,
                        )
                    else:
                        torch.save(
                            {
                                "model_state_dict": model.module.state_dict()
                                if isinstance(model, DDP)
                                else model.state_dict(),
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
                    total_train_time,
                    checkpoint_dir,
                ]
                train_log.to_csv(os.path.join(cfg.train.log.log_dir, "train_log.csv"), index=False)

                if cfg.train.log.wandb_on:
                    wandb.log(
                        {
                            "epoch_train_loss": train_loss,
                            "val_loss": val_loss,
                            "epoch_time": train_time,
                            "total_train_time": total_train_time,
                            "epoch": epoch + 1,
                        },
                        step=global_step,
                    )

            if dist.is_initialized():
                dist.barrier()


if __name__ == "__main__":
    main()
