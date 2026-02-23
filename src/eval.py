import os
import pandas as pd
from tokenizers import Tokenizer
import torch
import evaluate
from src.conf import SPECIAL_TOKENS, parse_eval_config
from src.data.dataloader import get_dataloader
from src.data.dataset import get_dataset
from src.model import TransformerModule, beam_search
from loguru import logger
from tqdm import tqdm

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def read_train_log(log_file_dir: str):
    data = pd.read_csv(log_file_dir, sep=",")
    logger.info(f"Read training log from {log_file_dir}")
    logger.info(f"\tNumber of epochs: {len(data)} (from {data['epoch'].min()} to {data['epoch'].max()})")
    logger.info(f"\tNumber of steps: {data['step'].max()}")
    return data


def read_test_log(log_file_dir: str):
    data = pd.read_csv(log_file_dir, sep=",")
    logger.info(f"Read test log from {log_file_dir}")
    logger.info(f"\tNumber of epochs: {len(data)} (from {data['epoch'].min()} to {data['epoch'].max()})")
    logger.info(f"\tNumber of steps: {data['step'].max()}")
    return data


def main():
    cfg = parse_eval_config()
    cfg.train.max_tokens_per_batch = 512  # disable batching for evaluation
    assert cfg.eval.exp_dir is not None, "Please specify the experiment directory in the configuration file."

    train_log_dir = os.path.join(cfg.eval.exp_dir, "train_log.csv")
    assert os.path.exists(train_log_dir), f"Train log file {train_log_dir} does not exist."
    train_log = read_train_log(train_log_dir)

    test_log_dir = os.path.join(cfg.eval.exp_dir, "test_log.csv")
    if os.path.exists(test_log_dir):
        test_log = read_test_log(test_log_dir)
    else:
        columns = train_log.columns.to_list()
        columns.append("BLEU")
        columns.append("METEOR")
        test_log = pd.DataFrame(columns=columns)
    out_log = pd.DataFrame(columns=test_log.columns)
    cnt = 0

    tokenizer_dir = cfg.data.output_dir
    tokenizer = Tokenizer.from_file(os.path.join(tokenizer_dir, "tokenizer"))
    token_sos_id = tokenizer.token_to_id(SPECIAL_TOKENS.SOS)
    token_eos_id = tokenizer.token_to_id(SPECIAL_TOKENS.EOS)

    test_dataset = get_dataset(cfg, tokenizer_dir, split="test")
    test_ds = get_dataloader(cfg, test_dataset)

    model = TransformerModule(
        cfg.data.tokenizer.vocab_size,
        cfg.train.model.d_model,
        cfg.train.model.num_heads,
        cfg.train.model.num_layers,
        cfg.train.model.dim_feedforward,
        cfg.train.model.dropout,
    )
    model = model.cuda()

    torch.use_deterministic_algorithms(True)

    bleu = evaluate.load("bleu")
    meteor = evaluate.load("meteor")

    for i in range(len(train_log)):
        if train_log.iloc[i]["epoch"] in test_log["epoch"].values:
            logger.info(f"Skip epoch {train_log.iloc[i]['epoch']} as it has been evaluated.")
            out_log.loc[cnt] = test_log[test_log["epoch"] == train_log.iloc[i]["epoch"]].values.tolist()[0]
            cnt += 1
            continue

        checkpoint_dir = train_log.iloc[i]["checkpoint_dir"]
        if (
            (not isinstance(checkpoint_dir, str))
            or (len(checkpoint_dir) == 0)
            or (not os.path.exists(checkpoint_dir))
            or (not os.path.isfile(checkpoint_dir))
        ):
            logger.info(f"Skip epoch {train_log.iloc[i]['epoch']} as the checkpoint does not exist.")
            out_log.loc[cnt] = train_log.iloc[i].values.tolist() + [pd.NA, pd.NA]
            cnt += 1
            continue

        state_dict = torch.load(checkpoint_dir, map_location="cuda")["model_state_dict"]
        model.load_state_dict(state_dict)
        model.eval()

        logger.info(f"Evaluating checkpoint from epoch {train_log.iloc[i]['epoch']} at {checkpoint_dir}...")

        predictions = []
        references = []

        with torch.no_grad():
            for batch in tqdm(test_ds, desc=f"Evaluating epoch {train_log.iloc[i]['epoch']}"):
                src = batch[0].to("cuda", non_blocking=True)
                cu_src_lens = batch[2].to("cuda", non_blocking=True)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    preds = beam_search(
                        model,
                        src,
                        cu_src_lens,
                        batch[4],
                        cfg.eval.beam_size,
                        cfg.eval.tolerance,
                        token_sos_id,
                        token_eos_id,
                    )

                batch_size = batch[7]
                tgt = batch[1].numpy(force=True)
                cu_tgt_lens = batch[3].numpy(force=True)
                for j in range(batch_size):
                    start_idx = cu_tgt_lens[j]
                    end_idx = cu_tgt_lens[j + 1]
                    references.append(tokenizer.decode(tgt[start_idx:end_idx]))
                    predictions.append(tokenizer.decode(preds[j]))

        bleu_score = bleu.compute(predictions=predictions, references=references)['bleu']  # type: ignore
        meteor_score = meteor.compute(predictions=predictions, references=references)['meteor']  # type: ignore
        logger.info(f"[Epoch {train_log.iloc[i]['epoch']}] BLEU Score: {bleu_score:.4f}, METEOR Score: {meteor_score:.4f}")
        out_log.loc[cnt] = train_log.iloc[i].values.tolist() + [bleu_score, meteor_score]
        cnt += 1
    out_log.to_csv(test_log_dir, index=False)
    logger.info(f"Saved evaluation log to {test_log_dir}")


if __name__ == "__main__":
    main()
