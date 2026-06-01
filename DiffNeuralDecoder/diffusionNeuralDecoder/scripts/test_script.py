
# Libraries
import logging
import csv
import torch
import numpy as np
import argparse
import os
import sys
from collections import OrderedDict
from dotenv import load_dotenv
from copy import deepcopy
from tqdm import tqdm
from time import time
from torch.utils.data import DataLoader, random_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))        # .../diffusionNeuralDecoder/scripts
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)                      # .../diffusionNeuralDecoder
REPO_DIR = os.path.dirname(PROJECT_DIR)                        # .../DiffNeuralDecoder
LOG_DIR = os.path.join(PROJECT_DIR, "logs")

if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

load_dotenv(os.path.join(PROJECT_DIR, ".env"))

# Modules
from diffusion_model import PhonemeDiT
from diffusion import create_diffusion
from diffusionNeuralDecoder.datasets.speechDataset import BrainToTextDataset, PhonemeDataset
from scripts.pretrain import (
    _get_env,
    _resolve_path,
    requires_grad,
    root_logger,
    save_checkpoint,
    training_step,
)
from scripts.brain_finetune import _batch_loss

def _configure_finetune_logger() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    root_logger.setLevel(logging.INFO)
    finetune_log = os.path.abspath(os.path.join(LOG_DIR, "test_results.log"))

    # Add exactly one finetune file handler even if this script is imported/run repeatedly.
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler) and os.path.abspath(handler.baseFilename) == finetune_log:
            return

    fh = logging.FileHandler(finetune_log)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    root_logger.addHandler(fh)

BASE_DIR = _get_env("BASE_DIR", default=PROJECT_DIR)
COMPETITION_DATA_DIR = _resolve_path(BASE_DIR, _get_env("COMPETITION_DATA_DIR", default="../../../competition_data"))
PREPROCESSED_DATA_DIR = _get_env("PREPROCESSED_DATA_DIR", default="/net/scratch/afan2025/preprocessed_data")
CHECKPOINT_DIR = _resolve_path(BASE_DIR, _get_env("CHECKPOINT_DIR", default="./checkpoints"))

Z_BRAIN_DIM = _get_env("Z_BRAIN_DIM", int)
D_MODEL = _get_env("D_MODEL", int)
MAX_TEXT_LEN = _get_env("MAX_TEXT_LEN", int)
VOCAB_SIZE = _get_env("VOCAB_SIZE", int)
MODEL_DEPTH = _get_env("MODEL_DEPTH", int)
NUM_HEADS = _get_env("NUM_HEADS", int)
MLP_RATIO = _get_env("MLP_RATIO", float)
DECODER_METHOD = _get_env("DECODER_METHOD", default="nn")
DIFFUSION_NOISE_SCHEDULE = _get_env("DIFFUSION_NOISE_SCHEDULE", default="cosine")

def main(args):
    _configure_finetune_logger()
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    assert torch.cuda.is_available(), "Using a GPU"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info(f"initializing dataset from partition {args.partition}")
    dataset = BrainToTextDataset(data_path=PREPROCESSED_DATA_DIR, partition = args.partition)

    data_loader  = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    logging.info("Dataset loaded from %s and partition %s", PREPROCESSED_DATA_DIR, args.partition)
    logging.info("Samples: total=%d %s=%d", len(dataset), args.partition, len(dataset))

    model = PhonemeDiT(
        d_model=D_MODEL,
        vocab_size=dataset.vocab_size,
        depth=MODEL_DEPTH,
        max_len=MAX_TEXT_LEN,
        num_heads=NUM_HEADS,
        mlp_ratio=MLP_RATIO,
        use_cross_attention=args.conditional == "conditional",  
        z_brain_dim=Z_BRAIN_DIM,
        use_final_layer=False,
        ).to(device)
    
    if args.conditional == "conditional":
        checkpoint_name = "finetune_step2_best.pt"
    else:
        checkpoint_name = "best.pt"
    ckpt_path = _resolve_path(BASE_DIR, os.path.join(CHECKPOINT_DIR, checkpoint_name))
    model_checkpoint = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(model_checkpoint["model"], strict=False)
    logging.info("Loaded model from %s", ckpt_path)
    logging.info("Missing keys (expected new finetune modules): %s", missing)
    logging.info("Unexpected keys: %s", unexpected)

    # diffusion scheduler but for inference time
    diffusion_scheduler = create_diffusion(
        timestep_respacing="", #maybe should be different for test inference?
        noise_schedule=DIFFUSION_NOISE_SCHEDULE,
        learn_sigma=False,
        sigma_small=True,
        predict_xstart=False,
    )
    logging.info("Diffusion scheduler created with noise schedule: %s", DIFFUSION_NOISE_SCHEDULE)

    model.eval()

    if args.conditional == "conditional":
        with torch.no_grad():
            for batch in data_loader:
                loss = _batch_loss(model, batch, device, diffusion_scheduler)
    else:
        with torch.no_grad():
            for _ in range(100): #just testing to see if it produces proper phoneme groupings

                # just a standard forward pass
                noise = torch.randn((1, 30, D_MODEL), device = device) #doing 1 inference per batch (B, D_MODEL, S)
                # fake_ids = torch.randint(0, 75, (1, 30), device=device)
                noise_mask = torch.ones(1, 30, dtype=torch.bool, device=device)
                t = torch.randint(0, diffusion_scheduler.num_timesteps, (noise.shape[0],), device=device)

                # run forward pass
                pred = model(noise, noise_mask, t)

                phoneme_seq = model.decode_tok(pred)

                # output to a file or print to main if using srun over sbatch



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--conditional", type=str, choices=["conditional","unconditional"], default="conditional")
    parser.add_argument("--partition", type=str, choices=["test", "competitionHoldOut"], default="test")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()
    main(args)