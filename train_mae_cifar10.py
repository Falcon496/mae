import argparse
import datetime
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# --- MAE + DeiT-Small encoder ---
from models_mae import mae_deit_small_patch16


def build_dataloader(data_dir: str, batch_size: int, num_workers: int):
    """CIFAR‑10 loaders (32→224)"""
    mean = [0.4914, 0.4822, 0.4465]
    std  = [0.2470, 0.2435, 0.2616]

    common = [transforms.Resize(224, antialias=True),
              transforms.ToTensor(),
              transforms.Normalize(mean, std)]

    train_tf = transforms.Compose([transforms.RandomHorizontalFlip(), *common])
    test_tf  = transforms.Compose(common)

    train_set = datasets.CIFAR10(data_dir, True,  download=True, transform=train_tf)
    test_set  = datasets.CIFAR10(data_dir, False, download=True, transform=test_tf)

    train_loader = DataLoader(train_set, batch_size, True,  num_workers=num_workers,
                              pin_memory=True, drop_last=True)
    test_loader  = DataLoader(test_set,  batch_size, False, num_workers=num_workers,
                              pin_memory=True)
    return train_loader, test_loader


def train_one_epoch(model, loader, optimizer, scaler, device, mask_ratio):
    model.train()
    running = 0.0
    for imgs, _ in tqdm(loader, leave=False):
        imgs = imgs.to(device)
        with torch.amp.autocast(device_type="cuda"):
            loss, _, _ = model(imgs, mask_ratio)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running += loss.item() * imgs.size(0)
    return running / len(loader.dataset)


def evaluate(model, loader, device, mask_ratio):
    model.eval(); tot = 0.0
    with torch.no_grad(), torch.amp.autocast(device_type="cuda"):
        for imgs, _ in loader:
            imgs = imgs.to(device)
            loss, _, _ = model(imgs, mask_ratio)
            tot += loss.item() * imgs.size(0)
    return tot / len(loader.dataset)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = build_dataloader(args.data_path, args.batch_size, args.num_workers)

    model = mae_deit_small_patch16(img_size=224, norm_pix_loss=True).to(device)
    total_params = sum(p.numel() for p in model.parameters())/1e6
    print(f"Model: DeiT‑Small MAE  | Params: {total_params:.1f}M  | Batch {args.batch_size}  | Epochs {args.epochs}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9,0.95), weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler(enabled=device.type=="cuda")

    os.makedirs(args.out_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(args.out_dir, "runs"))

    for ep in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, args.mask_ratio)
        val_loss   = evaluate(model, test_loader, device, args.mask_ratio) if (ep+1)%args.eval_interval==0 else None
        scheduler.step()

        msg = f"Ep {ep+1:03}/{args.epochs}  train={train_loss:.4f}"
        if val_loss is not None:
            msg += f"  val={val_loss:.4f}"
        msg += f"  lr={optimizer.param_groups[0]['lr']:.6f}"
        print(msg)

        writer.add_scalar("Loss/train", train_loss, ep+1)
        writer.add_scalar("LR", optimizer.param_groups[0]['lr'], ep+1)
        if val_loss is not None:
            writer.add_scalar("Loss/val", val_loss, ep+1)

        if (ep+1)%args.ckpt_interval==0 or ep+1==args.epochs:
            torch.save({"epoch":ep+1,
                        "model":model.state_dict(),
                        "opt":optimizer.state_dict(),
                        "scaler":scaler.state_dict()},
                       os.path.join(args.out_dir, f"mae_ep{ep+1}.pt"))
    writer.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser("MAE pre-train on CIFAR‑10")
    ap.add_argument("--data_path", type=str, default="./data")
    ap.add_argument("--out_dir",   type=str, default="./outputs_mae_cifar10")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--epochs",     type=int, default=300)
    ap.add_argument("--lr",         type=float, default=7.5e-5)
    ap.add_argument("--mask_ratio", type=float, default=0.75)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--eval_interval", type=int, default=25)
    ap.add_argument("--ckpt_interval", type=int, default=50)
    args = ap.parse_args()

    print("Start:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    main(args)
    print("End  :", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
