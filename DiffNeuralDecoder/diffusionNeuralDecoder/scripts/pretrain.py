
# Libraries
import logging
import csv
import torch
import torch.nn.functional as F
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
import math

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
from diffusionNeuralDecoder.datasets.speechDataset import PhonemeDataset, PHONEMES

# claude recced using a warmup run
from torch.optim.lr_scheduler import LambdaLR
warmup_steps = 1000
def warmup_schedule(step):
    if step < warmup_steps:
        return step / warmup_steps
    return 1.0

# load .env variables
def _get_env(name, cast=None, default=None):
    raw = os.getenv(name)
    if raw is None:
        if default is not None:
            return default
        raise ValueError(f"Missing required environment variable: {name}")
    return cast(raw) if cast is not None else raw


def _resolve_path(base_dir, path_value):
    return path_value if os.path.isabs(path_value) else os.path.normpath(os.path.join(base_dir, path_value))



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

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# File handler
fh = logging.FileHandler(os.path.join(LOG_DIR, 'pretrain.log'))
fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
root_logger.addHandler(fh)

# Also keep console output so you can see it live
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
root_logger.addHandler(sh)

@torch.no_grad()
def anisotropy_torch(E):
    V = E.shape[0]
    U = F.normalize(E, dim=1)
    return ((U.sum(0).pow(2).sum() - V) / (V * (V - 1))).item()

@torch.no_grad()
def participation_ratio_diagnose(E):
    Ec = E.detach().cpu().numpy()
    Ec = Ec - Ec.mean(axis=0, keepdims=True)
    S = np.linalg.svd(Ec, compute_uv=False)
    eigs = S ** 2
    return float((eigs.sum() ** 2) / (np.sum(eigs ** 2) + 1e-12))

@torch.no_grad()
def effective_rank(E):
    Ec = E.detach().cpu().numpy()
    Ec = Ec - Ec.mean(axis=0, keepdims=True)
    S = np.linalg.svd(Ec, compute_uv=False)
    p = S / (S.sum() + 1e-12)
    p_nz = p[p > 1e-12]
    entropy = -np.sum(p_nz * np.log(p_nz))
    return float(np.exp(entropy))

def decode_latent_to_tokens(model, z_hat):
    """Decode the model's direct latent predictions back to token ids via the embedding table."""
    logits = z_hat @ model.x_embedder.weight.T
    return logits.argmax(dim=-1)


def training_step(model, x_clean, x_mask, t, scheduler, token_ids=None, brain_data=None, brain_mask=None):
    noise = torch.randn_like(x_clean)
    x_noisy = scheduler.q_sample(x_clean, t, noise = noise)
    
    # directly predicting z_hat
    z_hat = model(x_noisy, x_mask, t, brain_data, brain_mask)

    # noise pred paradigm
    # noise_pred = model(x_noisy, x_mask, t, brain_data, brain_mask)
    # print(f"noise stats: mean={noise.mean().item():.4f}, std={noise.std().item():.4f}")
    # print(f"noise_pred stats: mean={noise_pred.mean().item():.4f}, std={noise_pred.std().item():.4f}")
    # print(f"x_clean stats: mean={x_clean.mean().item():.4f}, std={x_clean.std().item():.6f}")
    # print(f"x_noisy stats: mean={x_noisy.mean().item():.4f}, std={x_noisy.std().item():.6f}")

    # converting back to cleaned token predictions
    sqrt_alpha = torch.tensor(scheduler.sqrt_alphas_cumprod, device=x_clean.device, dtype=x_clean.dtype)[t].view(-1, 1, 1)
    sqrt_one_minus = torch.tensor(scheduler.sqrt_one_minus_alphas_cumprod, device=x_clean.device, dtype=x_clean.dtype)[t].view(-1, 1, 1)
    # z_hat = (x_noisy - sqrt_one_minus * noise_pred) / sqrt_alpha

    per_pos = ((z_hat - x_clean) ** 2).mean(dim=-1)  # (B, S)
    # loss = (per_pos * x_mask.float()).sum() / x_mask.float().sum() #THIS IS FOR MASKING OUT THE PADDED POSITIONS WHICH WE AREN'T DOING ANYMORE

    loss = per_pos.mean()

    logits_anchor = z_hat @ model.x_embedder.weight.T
    anchor_loss = F.cross_entropy(
        logits_anchor.view(-1, model.x_embedder.weight.shape[0]),
        token_ids.view(-1),
        # ignore_index=PHONEMES.index('<pad>'),  # <pad> is index 0
    )

    loss += anchor_loss
    # DEPRECATED: old loss was doing MSE on noise themselves, as with DiT, switching to anchor loss
    # per_pos = ((noise_pred - noise) ** 2).mean(dim=-1)  # (B, S)
    # # print(f"per_pos stats: mean={per_pos.mean().item():.6f}, max={per_pos.max().item():.6f}")

    # loss = (per_pos * x_mask.float()).sum() / x_mask.float().sum()
    # # print(f"final loss: {loss.item():.6f}")
    return loss, anchor_loss

# Additional Methods

# used for EMA
@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def save_checkpoint(model, ema, optimizer, epoch, step, val_loss, path):
    torch.save({
        'model': model.state_dict(),
        'ema': ema.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'step': step,
        'val_loss': val_loss,
    }, path)

def load_checkpoint(path, model, ema, optimizer):
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    ema.load_state_dict(ckpt['ema'])
    optimizer.load_state_dict(ckpt['optimizer'])
    return ckpt['epoch'], ckpt['val_loss']


def init_metrics_file(log_dir):
    run_id = int(time())
    metrics_path = os.path.join(log_dir, f"loss_metrics_{run_id}.csv")
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "phase",
            "epoch",
            "step",
            "loss",
            "steps_per_sec",
            "lr",
            "anisotropy",
            "effective_rank",
            "participation_ratio",
        ])
    return metrics_path


def append_metric(
    metrics_path,
    phase,
    epoch,
    step,
    loss,
    steps_per_sec="",
    lr="",
    anisotropy="",
    effective_rank="",
    participation_ratio="",
):
    with open(metrics_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            phase,
            epoch,
            step,
            loss,
            steps_per_sec,
            lr,
            anisotropy,
            effective_rank,
            participation_ratio,
        ])

def main(args):
    """
    Primary Training Script
    """

    # Set up the cuda infrastructure (this is where any Slurm thigns are needed)
    assert torch.cuda.is_available(), "Using a GPU"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataaset laoding
    phoneme_data_path = _resolve_path(BASE_DIR, GEN_PHONEME_DIR)
    dataset = PhonemeDataset(phoneme_data_path)
    if not 0.0 < args.train_split < 1.0:
        raise ValueError(f"TRAIN_SPLIT must be between 0 and 1, got {args.train_split}")

    train_size = int(len(dataset) * args.train_split)
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(args.global_seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle = True,
        num_workers = args.num_workers,
        pin_memory = True,
        drop_last = True,
        persistent_workers = True,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
    )

    logging.info(f"Dataset contains {len(dataset):,} samples ({phoneme_data_path})")
    logging.info(f"Train split: {len(train_dataset):,}, Val split: {len(val_dataset):,}")
    logging.info(f"Vocab size of dataset is {dataset.vocab_size}, provided vocab size is {VOCAB_SIZE}")

    # initialize model
    model = PhonemeDiT(
        d_model= D_MODEL,
        vocab_size= dataset.vocab_size,
        depth = MODEL_DEPTH,
        max_len = MAX_TEXT_LEN,
        num_heads = NUM_HEADS,
        mlp_ratio = MLP_RATIO,
        use_cross_attention = False, #pretraining is unconditional
        frequency_embedding_size = 256, #can change but honestly don't
        z_brain_dim = Z_BRAIN_DIM,
        decoder_approach = DECODER_METHOD).to(device)
    logging.info(f"Model initiated using device {device}")

    # ema (used in original meta DiT paper)
    logging.info(f"creating EMA")
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    update_ema(ema, model, decay=0)

    # Creating diffusion scheduler
    diffusion_scheduler = create_diffusion(timestep_respacing="",
                                        noise_schedule = DIFFUSION_NOISE_SCHEDULE,
                                        learn_sigma = False,
                                        sigma_small = True,
                                        predict_xstart = False) #default training steps, not for inference
    logging.info(f"Diffusion Scheduler created, with {DIFFUSION_NOISE_SCHEDULE} noise schedule")

    # initialize training objects
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    # claude suggested warmup scheduler
    scheduler = LambdaLR(opt, lr_lambda=warmup_schedule)

    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()
    best_val_loss = np.inf
    metrics_path = init_metrics_file(LOG_DIR)
    logging.info(f"Streaming train/val metrics to {metrics_path}")

    latest_path = os.path.join(CHECKPOINT_DIR, "latest.pt")
    if not args.train_from_scratch and os.path.exists(latest_path):
        start_epoch, best_val_loss = load_checkpoint(latest_path, model, ema, opt)
        start_epoch += 1  # resume from next epoch
        logging.info(f"Resumed from epoch {start_epoch}")
    else:
        start_epoch = 0
        if args.train_from_scratch:
            logging.info("--train-from-scratch set: ignoring existing checkpoints")

    logging.info(f"Training for {args.epochs} epochs")
    for epoch in tqdm(range(start_epoch, args.epochs)):
        logging.info(f"Beginning epoch {epoch}")
        model.train()
        for batch in train_loader:
            token_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            # logging.info(f"x min: {x.min()}, x max: {x.max()}, vocab_size: {model.x_embedder.num_embeddings}")
            x = model.embed_tok(token_ids)

            t = torch.randint(0, diffusion_scheduler.num_timesteps, (x.shape[0],), device=device)
            # logging.info(f"t shape: {t.shape}, t min: {t.min()}, t max: {t.max()}, t device: {t.device}")
            # logging.info(f"num_timesteps: {diffusion_scheduler.num_timesteps}")
            # loss_dict = diffusion_scheduler.training_losses(model, x, t) #DiT codebase has "model_kwargs" but idk what that is
            # loss = loss_dict["loss"].mean()
            loss, anchor_l = training_step(model, x, mask, t, diffusion_scheduler, token_ids=token_ids)
            opt.zero_grad()
            loss.backward()

            # gradient clipping  
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            scheduler.step()
            update_ema(ema, model)

            # Logging loss values
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_loss = avg_loss.item()
                current_lr = scheduler.get_last_lr()[0]
                ani = anisotropy_torch(model.x_embedder.weight)
                effective_r = effective_rank(model.x_embedder.weight)
                participation_r = participation_ratio_diagnose(model.x_embedder.weight)
                logging.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.6f}, Train Steps/Sec: {steps_per_sec:.2f}, Anisotropy: {ani:.4f}, Effective Rank: {effective_r:.4f}, Participation Ratio: {participation_r:.4f}, anchor loss {anchor_l:.4f}")
                append_metric(
                    metrics_path,
                    "train",
                    epoch,
                    train_steps,
                    avg_loss,
                    steps_per_sec,
                    current_lr,
                    ani,
                    effective_r,
                    participation_r,
                )
                # Reset monitoring variables: 
                running_loss = 0
                log_steps = 0
                start_time = time()

        # validation
        if epoch % args.ckpt_every == 0:
            val_losses = []
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    token_ids = batch["input_ids"].to(device)
                    mask = batch["attention_mask"].to(device)
                    x = model.embed_tok(token_ids)
                    t = torch.randint(0, diffusion_scheduler.num_timesteps, (x.shape[0],), device=device)
                    val_loss, _ = training_step(model, x, mask, t, diffusion_scheduler, token_ids=token_ids)
                    val_losses.append(val_loss.item())

                avg_val_loss = np.mean(val_losses)
                logging.info(f"(epoch={epoch:04d}) Val Loss: {avg_val_loss:.6f}")
                print(f"(epoch={epoch:04d}) Val Loss: {avg_val_loss:.6f}")
                current_lr = scheduler.get_last_lr()[0]
                ani = anisotropy_torch(model.x_embedder.weight)
                effective_r = effective_rank(model.x_embedder.weight)
                participation_r = participation_ratio_diagnose(model.x_embedder.weight)
                append_metric(
                    metrics_path,
                    "val",
                    epoch,
                    train_steps,
                    float(avg_val_loss),
                    "",
                    current_lr,
                    ani,
                    effective_r,
                    participation_r,
                )
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    save_checkpoint(model, ema, opt, epoch, train_steps, best_val_loss, os.path.join(CHECKPOINT_DIR, "best.pt"))
                
                save_checkpoint(model, ema, opt, epoch, train_steps, best_val_loss,
                        os.path.join(CHECKPOINT_DIR, "latest.pt"))

        model.train()

    model.eval()
    with torch.no_grad():
        sample_sequences = []
        for batch in val_loader:
            token_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            x = model.embed_tok(token_ids)
            t = torch.randint(0, diffusion_scheduler.num_timesteps, (x.shape[0],), device=device)
            z_hat = model(x, mask, t, None, None)
            decoded = decode_latent_to_tokens(model, z_hat)
            for i in range(decoded.shape[0]):
                valid_len = int(mask[i].sum().item())
                sample_sequences.append(decoded[i, :valid_len].detach().cpu().tolist())
                if len(sample_sequences) >= 10:
                    break
            if len(sample_sequences) >= 10:
                break

        logging.info("Sample decoded sequences from the current model:")
        for idx, seq in enumerate(sample_sequences[:10]):
            logging.info(f"sample(out) {idx}: {seq} \n sample(in) {idx}: {token_ids}")

        logging.info("Checking embedding gradients")
        E = model.x_embedder.weight.detach().cpu()
        logging.info("mean:", E.mean().item(), "std:", E.std().item())
        logging.info("per-row norm std:", E.norm(dim=1).std().item())

        E_fresh = torch.empty_like(E)
        torch.nn.init.normal_(E_fresh, std=1.0)  # match your actual init call

        logging.info("KS-ish check — row norm distributions:")
        logging.info("trained :", E.norm(dim=1)[:10])
        logging.info("fresh   :", E_fresh.norm(dim=1)[:10])

    logging.info("Done!")


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
    parser.add_argument("--train-from-scratch", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
