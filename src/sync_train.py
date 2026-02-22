import os
import torch
from src.conf import SPECIAL_TOKENS, parse_config
from src.data.dataloader import get_dataloaders
from src.data.dataset import get_datasets
from src.model import TransformerModule
from src.utils import get_optim, get_lr_scheduler
from tokenizers import Tokenizer
from loguru import logger

logger.level("TRACE")

cfg = parse_config()
tokenizer_dir = cfg.data.output_dir
print(tokenizer_dir)
print(cfg.train.max_tokens_per_local_batch)
tokenizer = Tokenizer.from_file(os.path.join(tokenizer_dir, "tokenizer"))
train_dataset, val_dataset = get_datasets(cfg, tokenizer_dir)
train_ds, val_ds = get_dataloaders(cfg, train_dataset, val_dataset)
criterion = torch.nn.CrossEntropyLoss(
    ignore_index=tokenizer.token_to_id(SPECIAL_TOKENS.PAD), label_smoothing=cfg.train.label_smoothing
)

model = TransformerModule(
    cfg.data.tokenizer.vocab_size,
    cfg.train.model.d_model,
    cfg.train.model.num_heads,
    cfg.train.model.num_layers,
    cfg.train.model.dim_feedforward,
    cfg.train.model.dropout,
)
model = model.cuda()


optim = get_optim(cfg, model)
lr_scheduler = get_lr_scheduler(cfg, optim)

import time

train_ds.batch_sampler.set_epoch(123)  # type: ignore
print(len(train_ds))

for idx, batch in enumerate(train_ds):
    src = batch[0].to("cuda", non_blocking=True)
    tgt = batch[1].to("cuda", non_blocking=True)
    cu_src_lens = batch[2].to("cuda", non_blocking=True)
    cu_tgt_lens = batch[3].to("cuda", non_blocking=True)
    labels = batch[6].to("cuda", non_blocking=True)

    start = time.time()
    optim.zero_grad()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(src, tgt, cu_src_lens, cu_tgt_lens, batch[4], batch[5])
        loss = criterion(logits, labels)
    loss.backward()
    optim.step()
    lr_scheduler.step()
    print(f"Loss: {loss}, Time: {time.time() - start}, Idx: {idx}")
