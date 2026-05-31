
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

if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

load_dotenv(os.path.join(PROJECT_DIR, ".env"))

# Modules
from diffusion_model import PhonemeDiT
from diffusion import create_diffusion
from diffusionNeuralDecoder.datasets.speechDataset import PhonemeDataset
from scripts.pretrain import _get_env, _resolve_path, update_ema, save_checkpoint, load_checkpoint, requires_grad, training_step

# claude recced using a warmup run
from torch.optim.lr_scheduler import LambdaLR

logging.basicConfig(
    filename='app.log', 
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

BASE_DIR = _get_env('BASE_DIR', default=PROJECT_DIR)
GEN_PHONEME_DIR = _get_env('GEN_PHONEME_DIR')
COMPETITION_DATA_DIR = _resolve_path(BASE_DIR, _get_env('COMPETITION_DATA_DIR', default='../../../competition_data'))
CHECKPOINT_DIR = _resolve_path(BASE_DIR, _get_env('CHECKPOINT_DIR', default='./checkpoints'))
LOG_DIR = os.path.join(PROJECT_DIR, "logs")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

Z_BRAIN_DIM = _get_env('Z_BRAIN_DIM', int)
D_MODEL = _get_env('D_MODEL', int)
MAX_TEXT_LEN = _get_env('MAX_TEXT_LEN', int)
VOCAB_SIZE = _get_env('VOCAB_SIZE', int)
MODEL_DEPTH = _get_env('MODEL_DEPTH', int)
NUM_HEADS = _get_env('NUM_HEADS', int)
MLP_RATIO = _get_env('MLP_RATIO', float)
DECODER_METHOD = _get_env('DECODER_METHOD', default='nn')
DIFFUSION_NOISE_SCHEDULE = _get_env('DIFFUSION_NOISE_SCHEDULE', default='cosine')

def step_1(model, opt):
    pass

def step_2(model, opt):
    pass


def main(args):
    """
    Primary Training Script
    """

    assert torch.cuda._is_available(), "Using a GPU"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # LOAD DATASET

    # Load Model + Initialize new parameters
    model = PhonemeDiT(
        d_model=D_MODEL,
        vocab_size=VOCAB_SIZE,
        depth=MODEL_DEPTH,
        max_len=MAX_TEXT_LEN,
        num_heads=NUM_HEADS,
        mlp_ratio=MLP_RATIO,
        use_cross_attention=True,  # now True
        z_brain_dim=Z_BRAIN_DIM,
        use_final_layer=True,).to(device)

    pretrained = torch.load(os.path.join(CHECKPOINT_DIR,"best.pt"), map_location='cpu')
    logging.info("loaded models from ")

    missing, unexpected = model.load_state_dict(pretrained['model'], strict=False)
    logging.info(f"Missing keys (expected — these are new modules): {missing}")
    logging.ino(f"Unexpected keys (should be empty): {unexpected}")

    # EMA?
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    update_ema(ema, model, decay=0)  # sync

    # STEP 1: Brain Encoder Warm Start, Freeze Base Model
    for name, param in model.named_parameters():
        if 'brain_encoder' in name or 'cross_attn' in name or 'cross_attn_gate' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4
    )

    # run first step training
    step_1(model, opt)

    # STEP 2: Later Layer Unfreeze join finetuning
    # unfreeze
    unfreeze_top_n = 2
    total_blocks = len(model.blocks)
    for i, block in enumerate(model.blocks):
        if i >= total_blocks - unfreeze_top_n:
            for param in block.parameters():
                param.requires_grad = True

    # reinitialize opt
    opt = torch.optim.AdamW([
        {'params': [p for n, p in model.named_parameters() 
                    if ('brain_encoder' in n or 'cross_attn' in n) and p.requires_grad],
        'lr': 1e-4},
        {'params': [p for n, p in model.named_parameters() 
                    if 'blocks' in n and 'cross_attn' not in n and p.requires_grad],
        'lr': 1e-5},
    ])

    step_2(model, opt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    # parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    # parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    # parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--train-split", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=5)
    args = parser.parse_args()
    main(args)