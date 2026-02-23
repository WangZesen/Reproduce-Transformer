import torch
from torch.utils.data import DataLoader
from typing import List, Tuple
from src.data.dataset import WMTDataset
from src.conf import Config
from src.data.sampler import DistributedTokenBatchSampler


def unbatched_collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int, torch.Tensor, int]:
    src_batch, tgt_batch = zip(*batch)
    src_packed = torch.cat(src_batch)
    # remove the last token (EOS) from target sequences for teacher forcing
    tgt_packed = torch.cat([tgt[:-1] for tgt in tgt_batch])
    # also prepare the labels by removing the first token (BOS) from target sequences
    label_packed = torch.cat([tgt[1:] for tgt in tgt_batch])
    cum_src_lengths = torch.cumsum(torch.tensor([0] + [src.size(0) for src in src_batch]), dim=0, dtype=torch.int32)
    cum_tgt_lengths = torch.cumsum(torch.tensor([0] + [tgt.size(0) - 1 for tgt in tgt_batch]), dim=0, dtype=torch.int32)
    max_src_len = max(src.size(0) for src in src_batch)
    max_tgt_len = max(tgt.size(0) - 1 for tgt in tgt_batch)
    return src_packed, tgt_packed, cum_src_lengths, cum_tgt_lengths, max_src_len, max_tgt_len, label_packed, len(batch)


def get_dataloaders(
    cfg: Config, train_dataset: WMTDataset, val_dataset: WMTDataset, train_shuffle: bool = True
) -> Tuple[DataLoader, DataLoader]:
    assert cfg.train is not None

    train_sampler = DistributedTokenBatchSampler(
        dataset=train_dataset,
        seed=cfg.train.seed,
        max_tokens=cfg.train.max_tokens_per_local_batch,
        shuffle=train_shuffle,
        total_epochs=cfg.train.max_epochs,
        drop_last=True,
    )
    val_sampler = DistributedTokenBatchSampler(
        dataset=val_dataset,
        seed=cfg.train.seed,
        max_tokens=cfg.train.max_tokens_per_local_batch,
        shuffle=False,
        total_epochs=cfg.train.max_epochs,
        drop_last=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=unbatched_collate_fn,
        pin_memory=True,
        num_workers=1,
        prefetch_factor=1,
        persistent_workers=True,
    )
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, collate_fn=unbatched_collate_fn, pin_memory=True)
    return train_loader, val_loader


def get_dataloader(cfg: Config, dataset: WMTDataset, shuffle: bool = False, drop_last: bool = False) -> DataLoader:
    assert cfg.train is not None

    sampler = DistributedTokenBatchSampler(
        dataset=dataset,
        seed=cfg.train.seed,
        max_tokens=cfg.train.max_tokens_per_local_batch,
        shuffle=shuffle,
        total_epochs=cfg.train.max_epochs,
        drop_last=drop_last,
    )
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=unbatched_collate_fn, pin_memory=True)
    return loader
