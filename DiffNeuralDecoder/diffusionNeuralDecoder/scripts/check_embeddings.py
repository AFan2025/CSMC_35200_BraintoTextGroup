
# Libraries
import logging
import argparse
import os
import sys
import editdistance

import numpy as np
import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader

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
from diffusionNeuralDecoder.datasets.speechDataset import BrainToTextDataset, ID_TO_PHONE
from scripts.pretrain import (
    _get_env,
    _resolve_path,
)
from scripts.brain_finetune import _batch_loss

def _configure_stdout_logger() -> logging.Logger:
    logger = logging.getLogger("test_script")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(sh)
    return logger

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

@torch.no_grad()
def enumerate_embeddings(model):
    model.eval()
    E = model.x_embedder

    E = E.detach().cpu().numpy()
    V, d = E.shape()
    rank = min(V,d)

    U, S, V = np.linalg.svd(E, full_matrices=False)
    p_raw = S / (S.sum() + 1e-12)
    p_raw_nz = p_raw[p_raw > 1e-12]
    effective_rank_raw = float(np.exp(-np.sum(p_raw_nz * np.log(p_raw_nz))))

    eigs = S ** 2
    participation_ratio = float((eigs.sum() ** 2) / (np.sum(eigs ** 2) + 1e-12))


def main(args):
    logger = _configure_stdout_logger()
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    assert torch.cuda.is_available(), "Using a GPU"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = BrainToTextDataset(data_path=PREPROCESSED_DATA_DIR, partition = args.partition)

    model = PhonemeDiT(
        d_model=D_MODEL,
        vocab_size=dataset.vocab_size,
        depth=MODEL_DEPTH,
        max_len=MAX_TEXT_LEN,
        num_heads=NUM_HEADS,
        mlp_ratio=MLP_RATIO,
        use_cross_attention= False,  
        z_brain_dim=Z_BRAIN_DIM,
        use_final_layer=False,
        ).to(device)
    
    if args.checkpoint is not None:
        checkpoint_name = args.checkpoint
    elif args.conditional == "conditional":
        checkpoint_name = "finetune_step2_best.pt"
    else:
        checkpoint_name = "best.pt"
    ckpt_path = _resolve_path(BASE_DIR, os.path.join(CHECKPOINT_DIR, checkpoint_name))
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model_checkpoint = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(model_checkpoint["model"], strict=False)
    logger.info("Loaded model from %s", ckpt_path)
    logger.info("Missing keys after load: %s", missing)
    logger.info("Unexpected keys after load: %s", unexpected)