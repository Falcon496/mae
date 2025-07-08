"""Fine‑tune a MAE‑pretrained DeiT‑Small encoder on **CIFAR‑10** with stronger
regularisation to avoid over‑fitting.

Key improvements (v2):
• **Mixed augmentation**  – RandAug + Mixup + CutMix + RandomErasing.
• **Label smoothing**     – CE(label_smoothing=0.1).
• **LR policy**           – 5‑epoch warm‑up → cosine decay, base_lr=2e‑4.
• **Layer‑wise LR decay** – deeper layers learn faster; implemented via
  per‑parameter lr_scale.
• **Early‑stopping**      – stop if `patience` epochs w/o test‑acc improve.
• Optional TensorBoard via --tb.
"""
import argparse
import datetime
import os
from functools import partial
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torchvision.transforms.autoaugment import RandAugment
from tqdm import tqdm

from models_mae import mae_deit_small_patch16, MaskedAutoencoderViT  # type: ignore

# -----------------------------------------------------------------------------
# CIFAR‑10 DATALOADER
# -----------------------------------------------------------------------------

MEAN = [0.4914, 0.4822, 0.4465]
STD  = [0.2470, 0.2435, 0.2616]


def build_loader(path: str, batch: int, workers: int) -> Tuple[DataLoader, DataLoader]:
    train_tf = transforms.Compose([
        transforms.Resize(224, antialias=True),
        RandAugment(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
    ])
    test_tf = transforms.Compose([
        transforms.Resize(224, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    train_set = datasets.CIFAR10(path, True,  download=True, transform=train_tf)
    test_set  = datasets.CIFAR10(path, False, download=True, transform=test_tf)
    return (
        DataLoader(train_set, batch, True,  num_workers=workers, pin_memory=True, drop_last=True),
        DataLoader(test_set,  batch, False, num_workers=workers, pin_memory=True),
    )

# -----------------------------------------------------------------------------
# MODEL  =  Encoder  +  Linear CLS Head
# -----------------------------------------------------------------------------

class CLSHead(nn.Module):
    def __init__(self, encoder: MaskedAutoencoderViT, n_cls: int = 10):
        super().__init__()
        self.encoder = encoder
        self.fc = nn.Linear(encoder.pos_embed.shape[-1], n_cls)

    def forward(self, x):
        with torch.amp.autocast(device_type="cuda"):
            latent, _, _ = self.encoder.forward_encoder(x, mask_ratio=0.0)
        return self.fc(latent[:, 0])

# -----------------------------------------------------------------------------
# TRAIN / EVAL LOOPS
# -----------------------------------------------------------------------------

def run_epoch(model, loader, opt, scaler, device, mixup_fn=None, train=True):
    """One epoch. Chooses criterion depending on whether soft labels are fed."""
    if train:
        model.train()
    else:
        model.eval()
    # criterion selection
    if train and mixup_fn is not None:
        criterion = SoftTargetCrossEntropy()
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    total, correct, loss_sum = 0, 0, 0.0
    context = torch.no_grad() if not train else torch.enable_grad()
    with context, torch.amp.autocast(device_type="cuda"):
        for x, y in tqdm(loader, leave=False):
            x, y = x.to(device), y.to(device)
            if train and mixup_fn is not None:
                x, y = mixup_fn(x, y)
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            loss_sum += loss.item() * x.size(0)
            pred = logits.argmax(1)
            correct += (pred == y.argmax(1) if y.ndim==2 else pred == y).sum().item()
            total += x.size(0)
    return loss_sum / total, correct / total

# -----------------------------------------------------------------------------
# LR DECAY PER LAYER (simple power‑law)
# -----------------------------------------------------------------------------

def param_groups_with_decay(model: nn.Module, base_lr: float, decay_rate: float = 0.9):
    groups = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "encoder" not in name:
            groups.append({"params": p, "lr": base_lr})
            continue
        layer_idx = int(name.split("blocks.")[-1].split(".")[0]) if "blocks." in name else 0
        lr_scale = decay_rate ** (24 - layer_idx)  # deeper layers (> idx) higher lr
        groups.append({"params": p, "lr": base_lr * lr_scale})
    return groups

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = build_loader(args.data_path, args.batch_size, args.num_workers)

    # ---- MODEL ----
    enc: MaskedAutoencoderViT = mae_deit_small_patch16(img_size=224, norm_pix_loss=True)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    enc.load_state_dict(ckpt.get("model", ckpt), strict=False)
    model = CLSHead(enc).to(device)

    # ---- MIXUP / CUTMIX ----
    mixup_fn = Mixup(mixup_alpha=0.2, cutmix_alpha=1.0, prob=1.0, num_classes=10)  # CIFAR‑10 has 10 classes

    # ---- OPTIM & LR ----
    base_lr = 2e-4
    param_groups = param_groups_with_decay(model, base_lr, decay_rate=0.75)
    opt = optim.AdamW(param_groups, betas=(0.9, 0.999), weight_decay=0.05)
    warm  = LinearLR(opt, start_factor=0.1, total_iters=5)
    cosine = CosineAnnealingLR(opt, T_max=args.epochs - 5, eta_min=base_lr * 1e-2)
    sched = SequentialLR(opt, [warm, cosine], milestones=[5])
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    writer = SummaryWriter(os.path.join(args.out_dir, "runs")) if args.tb else None
    best, patience_cnt = 0.0, 0
    os.makedirs(args.out_dir, exist_ok=True)

    for ep in range(args.epochs):
        tr_loss, tr_acc = run_epoch(model, train_loader, opt, scaler, device, mixup_fn, train=True)
        te_loss, te_acc = run_epoch(model, test_loader, opt, scaler, device, train=False)
        sched.step()
        lr_now = max(pg['lr'] for pg in opt.param_groups)  # report highest LR in groups
        print(f"Ep {ep+1:03}/{args.epochs}  train {tr_acc*100:.2f}%  test {te_acc*100:.2f}%  loss {tr_loss:.4f}/{te_loss:.4f}  lr {lr_now:.6f}")

        if writer:
            writer.add_scalar("Acc/train", tr_acc, ep+1)
            writer.add_scalar("Acc/test",  te_acc, ep+1)
            writer.add_scalar("Loss/train", tr_loss, ep+1)
            writer.add_scalar("Loss/test",  te_loss, ep+1)
            writer.add_scalar("LR", lr_now, ep+1)

        if te_acc > best:
            best, patience_cnt = te_acc, 0
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best_finetune.pt"))
        else:
            patience_cnt += 1
        if patience_cnt >= args.patience:
            print(f"Early stopping at epoch {ep+1}")
            break

    if writer:
        writer.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser("Fine‑tune MAE encoder on CIFAR‑10")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--data_path", type=str, default="./data")
    ap.add_argument("--out_dir", type=str, default="./out/output_finetune_cifar10")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--tb", action="store_true", help="log to TensorBoard")
    ap.add_argument("--patience", type=int, default=20, help="early stop patience")
    args = ap.parse_args()

    print("Start:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    main(args)
    print("End  :", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
