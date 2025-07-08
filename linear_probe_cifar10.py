"""Linear evaluation on CIFAR‑10 for a MAE‑pretrained DeiT‑Small encoder.

Example:
    uv run python linear_probe_cifar10.py \
        --ckpt ./outputs_mae_cifar10/mae_ep600.pt \
        --batch_size 512 --epochs 100
"""
import argparse
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from models_mae import mae_deit_small_patch16, MaskedAutoencoderViT  # type: ignore

from torch.utils.tensorboard import SummaryWriter

# -----------------------------------------------------------------------------
# Data loader
# -----------------------------------------------------------------------------

def build_loader(path: str, batch: int, workers: int) -> Tuple[DataLoader, DataLoader]:
    mean = [0.4914, 0.4822, 0.4465]
    std  = [0.2470, 0.2435, 0.2616]
    tfm = transforms.Compose([
        transforms.Resize(224, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train = datasets.CIFAR10(path, True,  download=True, transform=tfm)
    test  = datasets.CIFAR10(path, False, download=True, transform=tfm)
    return (
        DataLoader(train, batch, True,  num_workers=workers, pin_memory=True),
        DataLoader(test,  batch, False, num_workers=workers, pin_memory=True),
    )

# -----------------------------------------------------------------------------
# Feature extraction (no‑mask CLS token)
# -----------------------------------------------------------------------------

def extract_cls(model: MaskedAutoencoderViT, x: torch.Tensor):
    with torch.no_grad(), torch.amp.autocast(device_type="cuda"):
        latent, _, _ = model.forward_encoder(x, mask_ratio=0.0)
    return latent[:, 0]  # (B, D)

# -----------------------------------------------------------------------------
# Train / Eval helpers
# -----------------------------------------------------------------------------

def train_epoch(enc, head, loader, opt, device):
    enc.eval(); head.train(); loss_fn = nn.CrossEntropyLoss()
    tot, correct, running = 0, 0, 0.0
    for x, y in tqdm(loader, leave=False):
        x, y = x.to(device), y.to(device)
        feats = extract_cls(enc, x)
        preds = head(feats)
        loss = loss_fn(preds, y)
        opt.zero_grad(); loss.backward(); opt.step()
        running += loss.item() * x.size(0)
        correct += (preds.argmax(1) == y).sum().item()
        tot += x.size(0)
    return running / tot, correct / tot


def evaluate(enc, head, loader, device):
    enc.eval(); head.eval(); correct, tot = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            feats = extract_cls(enc, x)
            preds = head(feats)
            correct += (preds.argmax(1) == y).sum().item()
            tot += x.size(0)
    return correct / tot

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = build_loader(args.data_path, args.batch_size, args.num_workers)

    # ---- encoder (frozen) ----
    enc: MaskedAutoencoderViT = mae_deit_small_patch16(img_size=224, norm_pix_loss=True)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    enc.load_state_dict(ckpt.get("model", ckpt), strict=False)
    enc.to(device)
    for p in enc.parameters():
        p.requires_grad = False

    feat_dim = enc.pos_embed.shape[-1]  # 384 for DeiT‑Small
    head = nn.Linear(feat_dim, 10).to(device)

    opt   = optim.SGD(head.parameters(), lr=args.lr, momentum=0.9, weight_decay=0)
    sched = CosineAnnealingLR(opt, T_max=args.epochs)

    writer = SummaryWriter(log_dir=args.out_dir)
    for ep in range(args.epochs):
        tr_loss, tr_acc = train_epoch(enc, head, train_loader, opt, device)
        te_acc = evaluate(enc, head, test_loader, device)
        sched.step()
        print(f"Ep {ep+1:03}/{args.epochs}  loss={tr_loss:.4f}  train={tr_acc*100:.2f}%  test={te_acc*100:.2f}%")
        
        writer.add_scalar("LR", opt.param_groups[0]['lr'], ep+1)
        writer.add_scalar("Loss/train", tr_loss, ep+1)
        writer.add_scalar("Acc/train", tr_acc, ep+1)
        writer.add_scalar("Acc/test", te_acc, ep+1)
    writer.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser("Linear probe CIFAR‑10 with MAE encoder")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--data_path", type=str, default="./data")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--out_dir", type=str, default="./outputs_linear_cifar10")
    args = ap.parse_args()
    main(args)
