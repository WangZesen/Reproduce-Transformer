import random
import torch.distributed as dist
from torch.utils.data import Sampler
from typing import Iterator, List
from src.data.dataset import WMTDataset
from loguru import logger
import time

class DistributedTokenBatchSampler(Sampler[List[int]]):
    def __init__(self,
                 dataset: "WMTDataset",
                 seed: int,
                 max_tokens: int,
                 shuffle: bool = False):
        self._dataset = dataset
        self._seed = seed
        self._max_tokens = max_tokens
        assert max_tokens > 0, "max_tokens must be a positive integer"
        self._shuffle = shuffle

        if dist.is_available() and dist.is_initialized():
            self._num_replicas = dist.get_world_size()
            self._rank = dist.get_rank()
        else:
            self._num_replicas = 1
            self._rank = 0

        self._epoch = 0

    def _create_batches(self, seed: int = 42, epoch: int = 0) -> List[List[int]]:
        start_time = time.time()
        src_token_stats, tgt_token_stats = self._dataset.get_token_stats()
        data = list(zip(src_token_stats, tgt_token_stats, list(range(len(src_token_stats)))))
        random.seed(seed + epoch * 1007)
        random.shuffle(data)

        batches = []
        batch = []
        src_num_tokens = 0
        tgt_num_tokens = 0

        for i in range(len(data)):
            src_num_tokens += data[i][0]
            tgt_num_tokens += data[i][1]
            batch.append(data[i][2])
            
            if (src_num_tokens + tgt_num_tokens > self._max_tokens * 2) or (i == len(data) - 1):
                batches.append(batch)
                batch = []
                src_num_tokens = 0
                tgt_num_tokens = 0
        logger.trace(f"Created {len(batches)} batches in {time.time() - start_time:.5f} seconds")
        return batches

    def __iter__(self) -> Iterator[List[int]]:
        if self._shuffle:
            batches = self._create_batches(seed=self._seed, epoch=self._epoch)
        else:
            batches = self._create_batches()

        num_batches = len(batches)
        num_batches_per_replica = num_batches // self._num_replicas
        for i in range(num_batches_per_replica):
            index = i * self._num_replicas + self._rank
            yield batches[index]

    def __len__(self) -> int:
        if self._shuffle:
            batches = self._create_batches(seed=self._seed, epoch=self._epoch)
        else:
            batches = self._create_batches()
        num_batches = len(batches)
        return num_batches // self._num_replicas

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch
