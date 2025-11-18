'''
    Set the environment variables for distributed training before importing any packages.
    Note: It's assumed that each node has a single GPU, and the choice of RDMA interface
          specified by NCCL_IB_HCA is subject to the actual network configuration under test.
'''

import os
import sys
import subprocess
from loguru import logger
import math
import time
import wandb
import torch
import tomli_w
import pandas as pd
from src.utils import SmoothedValue, initialize_dist, get_optim_fn, get_lr_scheduler_fn, gather_statistics, get_run_name, get_group_name
from src.conf import Config, parse_config, SPECIAL_TOKENS
from src.data import get_datasets, get_dataloaders
from src.model import get_model
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn import Module
from src.decent_dp.ddp import DecentralizedDataParallel as DecentDP
from torch import GradScaler
from torch.profiler import schedule, profile, ProfilerActivity

logger.remove()
logger.add(sys.stdout, level='DEBUG')

rank = int(os.environ.get('RANK', 0))
local_rank = int(os.environ.get('LOCAL_RANK', 0))
world_size = int(os.environ.get('WORLD_SIZE', 1))
local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', 1))
if local_world_size > 1:
    devices = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    assert len(devices) == local_world_size, 'Each process must have a single GPU.'
    os.environ['CUDA_VISIBLE_DEVICES'] = devices[local_rank]
slurm_id = os.environ.get('SLURM_JOB_ID', '0')
models = subprocess.check_output('nvidia-smi -L', shell=True).decode('utf-8')
if 'Tesla T4' in models:
    os.environ['NCCL_IB_HCA'] = 'mlx5'


'''
    Helper functions
'''
def get_mixing_factor(cfg: Config, step: int, steps_per_epoch: int) -> float:
    """
    Calculate the mixing factor for adaptive consensus based on the current step.
    """
    assert cfg.train.backend.name == 'decent_dp', 'Only decent_dp backend is supported for this function.'
    if cfg.train.backend.adaptive_consensus_omega == 0.0:
        return 1.0
    else:
        if (step <= cfg.train.lr_scheduler.warmup_steps) or (step <= cfg.train.backend.ac_start_step):
            return 1.0
        else:
            assert cfg.train.lr_scheduler.type == 'inverse_sqrt'
            min_lr = 0.0 # (cfg.train.lr_scheduler.warmup_steps ** 0.5) * ((cfg.train.max_epochs * steps_per_epoch + 1) ** (-0.5))
            max_lr = (cfg.train.lr_scheduler.warmup_steps ** 0.5) * (cfg.train.backend.ac_start_step ** (-0.5))
            lr = (cfg.train.lr_scheduler.warmup_steps ** 0.5) * ((step + 1) ** (-0.5))
            factor = ((lr - min_lr) / (max_lr - min_lr)) ** cfg.train.backend.adaptive_consensus_p
            factor = factor * cfg.train.backend.adaptive_consensus_omega + (1.0 - cfg.train.backend.adaptive_consensus_omega)
            return factor


def train_epoch(cfg: Config,
                epoch: int,
                step: int,
                model: DecentDP,
                train_ds: DataLoader,
                criterion: Module,
                scaler: GradScaler,
                profiler):
    assert cfg.train is not None
    start_time = time.time()
    model.train()
    total_step = len(train_ds)
    loss_metric = SmoothedValue(cfg.train.log.log_freq)
    perplexity_metric = SmoothedValue(cfg.train.log.log_freq)
    tpb_metric = SmoothedValue(cfg.train.log.log_freq)
    if rank == 0:
        logger.info(f'[Train Epoch {epoch+1}] {total_step} batches')

    for batch_idx, (src, tgt, n_tokens) in enumerate(train_ds):
        # Move data to GPU
        src = src.to('cuda', non_blocking=True)
        tgt = tgt.to('cuda', non_blocking=True)

        # Forward pass
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=cfg.train.use_amp):
            pred = model(src, tgt[:, :-1])
            loss = criterion(pred.view(-1, cfg.data.tokenizer.vocab_size), tgt[:, 1:].reshape(-1))

        # Backward pass
        scaler.scale(loss).backward()

        # Update metrics
        _loss = loss.detach().item()
        loss_metric.update(_loss)
        perplexity_metric.update(math.exp(_loss))
        tpb_metric.update(n_tokens * cfg.train.network.world_size)
        lr = model._optims[0].param_groups[0]['lr'] if len(model._optims) > 0 else 0.0
        step += 1

        if rank == 0:
            if cfg.train.log.wandb_on and (step % cfg.train.log.log_freq == 0):
                wandb.log({'loss': loss_metric.avg, 'lr': lr, 'tpb': tpb_metric.avg, 'perplexity': perplexity_metric.avg, 'mixing_factor': model._mixing_factor}, step=step)

            if step % cfg.train.log.log_freq == 0:
                epoch_step = batch_idx + 1
                logger.info(f'step: {step} ({(time.time() - start_time) / epoch_step:5.2f} s/it), loss: {loss_metric.avg:.6f}, perplexity: {perplexity_metric.avg:.6f}' + \
                        f', lr: {lr:.6f}, tpb: {tpb_metric.avg:.1f}, mem: {torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024:.2f} GB'
                        f', mixing_factor: {model._mixing_factor:.6f}')
        
        # set mixing factor for adaptive consensus
        model.set_mixing_factor(get_mixing_factor(cfg, step, total_step))

        profiler.step()
        del src, tgt, pred, loss
    return loss_metric.global_avg, perplexity_metric.global_avg, step, time.time() - start_time


@torch.no_grad()
def valid(cfg: Config,
          epoch: int,
          model: Module,
          val_ds: DataLoader,
          criterion: Module):

    model.eval()
    total_loss = 0
    total_perplexity = 0
    if rank == 0:
        logger.info(f'[Validate Epoch {epoch+1}]')

    for src, tgt, _ in val_ds:
        src = src.to('cuda', non_blocking=True)
        tgt = tgt.to('cuda', non_blocking=True)
        pred = model(src, tgt[:, :-1])
        loss = criterion(pred.view(-1, cfg.data.tokenizer.vocab_size), tgt[:, 1:].reshape(-1))
        total_loss += loss.item()
        total_perplexity += math.exp(loss.item())
    return total_loss / len(val_ds), total_perplexity / len(val_ds)


def main():
    cfg = parse_config()
    assert cfg.train is not None
    assert cfg.train.backend.name == 'decent_dp', 'Only decent_dp backend is supported for this script.'
    assert cfg.train.network.world_size > 1, 'This script is for distributed training only.'

    '''
        Initialize the distributed process group and set random seed
    '''
    assert torch.cuda.is_available(), 'CUDA is not available' # always use CUDA
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    initialize_dist()
    torch.manual_seed(cfg.train.reproduce.seed)

    '''
        Intialize the datasets and the dataloaders.
    '''
    tokenizer_dir = cfg.data.output_dir
    tokenizer = Tokenizer.from_file(os.path.join(tokenizer_dir, 'tokenizer'))
    train_dataset, val_dataset, test_dataset = get_datasets(cfg, tokenizer_dir)
    train_ds, val_ds = get_dataloaders(cfg,
                                       train_dataset,
                                       val_dataset,
                                       test_dataset,
                                       tokenizer,
                                       SPECIAL_TOKENS.PAD)
    
    '''
        Initialize model, optimizer, and loss.
    '''
    model = get_model(cfg, tokenizer, SPECIAL_TOKENS.PAD)
    model.to('cuda')
    if rank == 0:
        trainable_params = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values())
        logger.info(f'[INFO] Model trainable parameters: {trainable_params}')
    
    optimizer_fn = get_optim_fn(cfg)
    lr_scheduler_fn = get_lr_scheduler_fn(cfg)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id(SPECIAL_TOKENS.PAD),
                                          label_smoothing=cfg.train.label_smoothing)
    scaler = torch.GradScaler(enabled=cfg.train.use_amp, device='cuda')

    
    model = DecentDP(model,
                     optim_fn=optimizer_fn,
                     lr_scheduler_fn=lr_scheduler_fn,
                     topology=cfg.train.backend.topology,
                     scaler=scaler)

    if rank == 0:
        logger.info(cfg.model_dump_json())
        if cfg.train.log.wandb_on:
            wandb.init(project=cfg.train.log.wandb_project,
                       config=cfg.model_dump(),
                       name=get_run_name(cfg),
                       group=get_group_name(cfg),
                       dir=os.environ['TMPDIR'])
        logger.info(model)
        with open(os.path.join(cfg.train.log.log_dir, 'data_cfg.dump.toml'), 'wb') as f:
            tomli_w.dump(cfg.data.model_dump(exclude_none=True), f)
        with open(os.path.join(cfg.train.log.log_dir, 'train_cfg.dump.toml'), 'wb') as f:
            tomli_w.dump(cfg.train.model_dump(exclude_none=True), f)
    
    '''
        Load the model from checkpoint if specified.
    '''

    global_step = 0
    start_epoch = 0
    total_train_time = 0.

    if cfg.train.checkpoint_dir:
        raise NotImplementedError('Checkpoint loading is not implemented yet.')
    
    if dist.is_initialized():
        dist.barrier()
    
    '''
        Training loop.
    '''
    if rank == 0:
        train_log = pd.DataFrame(columns=['epoch', 'step', 'train_loss', 'train_perplexity', 'val_loss', 'val_perplexity', 'time', 'checkpoint_dir'])

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(
            wait=64,
            warmup=2,
            active=8,
            repeat=1
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            os.path.join(cfg.train.log.log_dir, 'tb_trace'),
            worker_name=f'worker_{rank:02d}'
        ),
        record_shapes=True,
        with_stack=True,
        profile_memory=True
    ) as p:
        for epoch in range(start_epoch, cfg.train.max_epochs):
            train_ds.batch_sampler.set_epoch(epoch) # type: ignore
            train_loss, train_perplexity, global_step, epoch_train_time = train_epoch(cfg,
                                                                                      epoch,
                                                                                      global_step,
                                                                                      model,
                                                                                      train_ds,
                                                                                      criterion,
                                                                                      scaler,
                                                                                      p)
            total_train_time += epoch_train_time
            model.global_avg(may_revert=True)
            val_loss, val_perplexity = valid(cfg, epoch, model, val_ds, criterion)
            stats = gather_statistics(train_loss, train_perplexity, val_loss, val_perplexity)

            if rank == 0:
                logger.info(f'[epoch {epoch+1}] train loss: {stats[0]:.6f}, train perplexity: {stats[1]:.6f}, ' \
                            + f'val loss: {stats[2]:.6f}, val perplexity: {stats[3]:.6f}, ' \
                            + f'epoch time: {total_train_time:.3f} s')
                checkpoint_dir = ""
                if ((epoch + 1) % cfg.train.log.checkpoint_freq == 0) or (epoch == cfg.train.max_epochs - 1):
                    checkpoint_dir = os.path.join(cfg.train.log.log_dir, 'checkpoints', f'checkpoint_{epoch+1}.pt')
                    os.makedirs(os.path.dirname(checkpoint_dir), exist_ok=True)
                    torch.save({
                        'epoch': epoch + 1,
                        'global_step': global_step,
                        'total_train_time': total_train_time,
                        'model_state_dict': model._model.state_dict(),
                        'val_loss': stats[2],
                        'val_perplexity': stats[3]
                    }, checkpoint_dir)
                if cfg.train.log.wandb_on:
                    wandb.log({'epoch_train_loss': stats[0],
                               'epoch_train_perplexity': stats[1],
                               'val_loss': stats[2],
                               'val_perplexity': stats[3],
                               'time': total_train_time,
                               'epoch': epoch + 1,
                               }, step=global_step)
                train_log.loc[epoch - start_epoch] = [epoch + 1, # type: ignore
                                                      global_step,
                                                      stats[0],
                                                      stats[1],
                                                      stats[2],
                                                      stats[3],
                                                      total_train_time,
                                                      os.path.abspath(checkpoint_dir) if checkpoint_dir else '']
                train_log.to_csv(os.path.join(cfg.train.log.log_dir, 'train_log.csv'), index=False) # type: ignore

            model.revert_global_avg()

            dist.barrier()

    logger.info(f'Training finished in {total_train_time:.3f} s')
    if rank == 0:
        if cfg.train.log.wandb_on:
            wandb.finish()


if __name__ == '__main__':
    main()
