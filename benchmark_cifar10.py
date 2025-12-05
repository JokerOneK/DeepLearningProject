import os
import csv
import math
import copy
import time
import argparse
import threading
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

try:
    from ptflops import get_model_complexity_info
except ImportError:
    get_model_complexity_info = None



class GPUEnergyMeter:
    def __init__(self, device_index: int = 0, interval_s: float = 0.2):
        self.device_index = device_index
        self.interval_s = interval_s
        self._running = False
        self._thread = None
        self.energy_j = 0.0
        self._last_t = None
        self._last_p = None

        try:
            import pynvml
            self.pynvml = pynvml
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            self.available = True
        except Exception:
            self.available = False

    def _read_power_w(self) -> float:
        mw = self.pynvml.nvmlDeviceGetPowerUsage(self.handle)  # milliwatts
        return float(mw) / 1000.0

    def start(self):
        if not self.available:
            self.energy_j = float("nan")
            return
        self.energy_j = 0.0
        self._running = True
        self._last_t = time.time()
        self._last_p = self._read_power_w()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            time.sleep(self.interval_s)
            t = time.time()
            p = self._read_power_w()
            dt = t - self._last_t
            self.energy_j += 0.5 * (self._last_p + p) * dt
            self._last_t, self._last_p = t, p

    def stop(self) -> float:
        if not self.available:
            return float("nan")
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        return self.energy_j


#  Utils

def seed_everything(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)

def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_flops_and_params_ptflops(model: nn.Module, input_res=(3, 32, 32)) -> Tuple[str, str]:
    if get_model_complexity_info is None:
        return "N/A (install ptflops)", "N/A"
    device_backup = next(model.parameters()).device
    model_cpu = model.to("cpu")
    with torch.no_grad():
        flops_str, params_str = get_model_complexity_info(
            model_cpu,
            input_res,
            as_strings=True,
            print_per_layer_stat=False,
            verbose=False,
        )
    model.to(device_backup)
    return flops_str, params_str

def append_rows_to_csv(path: str, rows: List[Dict]):
    if not rows:
        return
    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if (not file_exists) or (f.tell() == 0):
            writer.writeheader()
        writer.writerows(rows)


#  Our models (CNN / ViT / Hybrid)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10, base_channels: int = 64):
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        self.features = nn.Sequential(
            nn.Conv2d(3, c1, 3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c1, 3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32 -> 16

            nn.Conv2d(c1, c2, 3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c2, 3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16 -> 8

            nn.Conv2d(c2, c3, 3, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3, c3, 3, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(c3, num_classes)

    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.head(x)


class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=128):
        super().__init__()
        assert img_size % patch_size == 0
        self.grid = img_size // patch_size
        self.num_patches = self.grid * self.grid
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)                 # (B, C, 8, 8)
        x = x.flatten(2).transpose(1, 2) # (B, 64, C)
        return x


class ConvStemTokenizerSafe(nn.Module):
    def __init__(
        self,
        img_size: int = 32,
        in_chans: int = 3,
        embed_dim: int = 128,
        stem_depth: int = 2,
        stem_channels: Tuple[int, int, int] = (64, 96, 128),
    ):
        super().__init__()
        stem_depth = max(1, min(stem_depth, 3))
        self.stem_depth = stem_depth

        layers = []
        cur = in_chans
        for i in range(stem_depth):
            out = stem_channels[i]
            layers += [
                nn.Conv2d(cur, out, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out),
                nn.ReLU(inplace=True),
            ]
            cur = out
        self.stem = nn.Sequential(*layers)

        self.patchify = nn.Conv2d(cur, embed_dim, kernel_size=4, stride=4, padding=0)

        self.num_patches = 8 * 8

    def forward(self, x):
        x = self.stem(x)          # (B, C, 32, 32)
        x = self.patchify(x)      # (B, D, 8, 8)
        x = x.flatten(2).transpose(1, 2)  # (B, 64, D)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 2.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)

        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        r = x
        x = self.norm1(x)
        a, _ = self.attn(x, x, x)
        x = r + self.drop(a)

        r = x
        x = self.norm2(x)
        x = r + self.drop(self.mlp(x))
        return x


class ViTClassifier(nn.Module):
    def __init__(self, embed_dim=128, depth=3, num_heads=4, mlp_ratio=2.0, num_patches=64, num_classes=10):
        super().__init__()
        self.embed_dim = embed_dim
        self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos = nn.Parameter(torch.zeros(1, 1 + num_patches, embed_dim))

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.normal_(self.cls, std=0.02)
        nn.init.normal_(self.pos, std=0.02)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, tokens):
        B, N, C = tokens.shape
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, tokens], dim=1)
        x = x + self.pos[:, : N + 1, :]

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return self.head(x[:, 0])


class PureViTClassifier(nn.Module):
    def __init__(self, embed_dim=128, depth=3, num_heads=4, mlp_ratio=2.0, patch_size=4, num_classes=10):
        super().__init__()
        self.patch = PatchEmbed(img_size=32, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        self.vit = ViTClassifier(embed_dim, depth, num_heads, mlp_ratio, num_patches=self.patch.num_patches, num_classes=num_classes)

    def forward(self, x):
        t = self.patch(x)
        return self.vit(t)


class HybridConvViTClassifier(nn.Module):
    def __init__(self, embed_dim=128, depth=3, num_heads=4, mlp_ratio=2.0, stem_depth=2, num_classes=10):
        super().__init__()
        self.tok = ConvStemTokenizerSafe(img_size=32, in_chans=3, embed_dim=embed_dim, stem_depth=stem_depth)
        self.vit = ViTClassifier(embed_dim, depth, num_heads, mlp_ratio, num_patches=self.tok.num_patches, num_classes=num_classes)

    def forward(self, x):
        t = self.tok(x)
        return self.vit(t)


#  Compact Transformer baseline (CCT-style)

class SequencePooling(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, x):
        # x: (B, N, D)
        w = self.attn(x).squeeze(-1)      # (B, N)
        w = torch.softmax(w, dim=1)       # (B, N)
        pooled = (x * w.unsqueeze(-1)).sum(dim=1)  # (B, D)
        return pooled


class CCTCompactClassifier(nn.Module):
    def __init__(self, embed_dim=128, depth=3, num_heads=4, mlp_ratio=2.0, stem_depth=2, num_classes=10):
        super().__init__()
        self.tok = ConvStemTokenizerSafe(img_size=32, in_chans=3, embed_dim=embed_dim, stem_depth=stem_depth)
        self.pos = nn.Parameter(torch.zeros(1, self.tok.num_patches, embed_dim))

        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.pool = SequencePooling(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.normal_(self.pos, std=0.02)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        t = self.tok(x)                   # (B, 64, D)
        t = t + self.pos[:, : t.size(1), :]
        for blk in self.blocks:
            t = blk(t)
        t = self.norm(t)
        pooled = self.pool(t)
        return self.head(pooled)


#  CIFAR-ResNet baseline (strong CNN)

class CIFARBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = None
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return out


class CIFARResNet(nn.Module):
    def __init__(self, depth: int = 56, num_classes: int = 10):
        super().__init__()
        assert (depth - 2) % 6 == 0, "CIFAR ResNet depth should be 6n+2 (e.g., 44, 56, 110)"
        n = (depth - 2) // 6

        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(16, n, stride=1)
        self.layer2 = self._make_layer(32, n, stride=2)
        self.layer3 = self._make_layer(64, n, stride=2)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, planes, blocks, stride):
        layers = [CIFARBasicBlock(self.in_planes, planes, stride=stride)]
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(CIFARBasicBlock(self.in_planes, planes, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


#  Training / eval

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_accum_steps: int = 1,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    optimizer.zero_grad(set_to_none=True)

    for i, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets) / grad_accum_steps
        loss.backward()

        if (i + 1) % grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        bs = images.size(0)
        total_loss += loss.item() * bs * grad_accum_steps
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_samples += bs

    # если датасет не кратен accum_steps
    if (len(loader) % grad_accum_steps) != 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)
        bs = images.size(0)
        total_loss += loss.item() * bs
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_samples += bs
    return total_loss / total_samples, total_correct / total_samples


#  Experiment config

@dataclass
class ModelConfig:
    name: str
    family: str   # ours_cnn / ours_vit / ours_hybrid / sota_resnet / sota_cct
    scale: str    # small / medium / large
    params: Dict  # builder args


def build_model(cfg: ModelConfig) -> nn.Module:
    if cfg.family == "ours_cnn":
        return SimpleCNN(num_classes=10, base_channels=cfg.params["base_channels"])
    if cfg.family == "ours_vit":
        return PureViTClassifier(
            embed_dim=cfg.params["embed_dim"],
            depth=cfg.params["depth"],
            num_heads=cfg.params["num_heads"],
            mlp_ratio=cfg.params["mlp_ratio"],
            patch_size=4,
            num_classes=10,
        )
    if cfg.family == "ours_hybrid":
        return HybridConvViTClassifier(
            embed_dim=cfg.params["embed_dim"],
            depth=cfg.params["depth"],
            num_heads=cfg.params["num_heads"],
            mlp_ratio=cfg.params["mlp_ratio"],
            stem_depth=cfg.params["stem_depth"],
            num_classes=10,
        )
    if cfg.family == "sota_resnet":
        return CIFARResNet(depth=cfg.params["depth"], num_classes=10)
    if cfg.family == "sota_cct":
        return CCTCompactClassifier(
            embed_dim=cfg.params["embed_dim"],
            depth=cfg.params["depth"],
            num_heads=cfg.params["num_heads"],
            mlp_ratio=cfg.params["mlp_ratio"],
            stem_depth=cfg.params["stem_depth"],
            num_classes=10,
        )
    raise ValueError(f"Unknown family: {cfg.family}")


def make_optimizer_and_scheduler(
    cfg: ModelConfig,
    model: nn.Module,
    epochs: int,
    batch_size: int,
):
    if cfg.family in ("ours_cnn", "sota_resnet"):
        # типичный рецепт для CIFAR: SGD momentum
        base_lr_128 = 0.1
        lr = base_lr_128 * (batch_size / 128.0)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        return optimizer, scheduler

    # transformers
    lr = 3e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=5e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    return optimizer, scheduler


#  Main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device_index", type=int, default=0)
    parser.add_argument("--out_prefix", type=str, default="cifar10_benchmark")
    parser.add_argument("--filter", type=str, default="")              # "_small" or "sota_" or "ours_vit"
    parser.add_argument("--no_ptflops", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # Dataset
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    ])

    root = "./data"
    full_train = datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
    test_ds = datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)

    val_size = 5000
    train_size = len(full_train) - val_size
    train_ds, val_ds = random_split(
        full_train,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    # Model configs
    configs: List[ModelConfig] = [
        # OURS
        ModelConfig("ours_cnn_small",   "ours_cnn",   "small",  {"base_channels": 48}),
        ModelConfig("ours_vit_small",   "ours_vit",   "small",  {"embed_dim": 160, "depth": 3, "num_heads": 4, "mlp_ratio": 2.0}),
        ModelConfig("ours_hyb_small",   "ours_hybrid","small",  {"embed_dim": 160, "depth": 3, "num_heads": 4, "mlp_ratio": 2.0, "stem_depth": 1}),

        ModelConfig("ours_cnn_medium",  "ours_cnn",   "medium", {"base_channels": 64}),
        ModelConfig("ours_vit_medium",  "ours_vit",   "medium", {"embed_dim": 192, "depth": 3, "num_heads": 4, "mlp_ratio": 3.0}),
        ModelConfig("ours_hyb_medium",  "ours_hybrid","medium", {"embed_dim": 160, "depth": 4, "num_heads": 4, "mlp_ratio": 3.0, "stem_depth": 2}),

        ModelConfig("ours_cnn_large",   "ours_cnn",   "large",  {"base_channels": 80}),
        ModelConfig("ours_vit_large",   "ours_vit",   "large",  {"embed_dim": 192, "depth": 4, "num_heads": 4, "mlp_ratio": 4.0}),
        ModelConfig("ours_hyb_large",   "ours_hybrid","large",  {"embed_dim": 192, "depth": 4, "num_heads": 4, "mlp_ratio": 3.0, "stem_depth": 3}),

        # SOTA/strong baselines
        ModelConfig("sota_resnet44",    "sota_resnet","small",  {"depth": 44}),
        ModelConfig("sota_resnet56",    "sota_resnet","medium", {"depth": 56}),
        ModelConfig("sota_resnet110",   "sota_resnet","large",  {"depth": 110}),

        ModelConfig("sota_cct_small",   "sota_cct",   "small",  {"embed_dim": 160, "depth": 3, "num_heads": 4, "mlp_ratio": 2.0, "stem_depth": 2}),
        ModelConfig("sota_cct_medium",  "sota_cct",   "medium", {"embed_dim": 192, "depth": 4, "num_heads": 4, "mlp_ratio": 2.0, "stem_depth": 2}),
        ModelConfig("sota_cct_large",   "sota_cct",   "large",  {"embed_dim": 256, "depth": 4, "num_heads": 8, "mlp_ratio": 2.0, "stem_depth": 2}),
    ]

    if args.filter:
        configs = [c for c in configs if args.filter in c.name]
        print(f"Filter='{args.filter}' -> running {len(configs)} models")

    summary_path = f"{args.out_prefix}_summary.csv"
    curves_path = f"{args.out_prefix}_curves.csv"

    if os.path.exists(summary_path):
        os.remove(summary_path)
    if os.path.exists(curves_path):
        os.remove(curves_path)

    criterion = nn.CrossEntropyLoss()

    summary_rows: List[Dict] = []

    for cfg in configs:
        print("\n" + "=" * 100)
        print(f"Model: {cfg.name} | family={cfg.family} | scale={cfg.scale} | params={cfg.params}")

        model = build_model(cfg)

        # params + flops
        trainable_params = count_trainable_params(model)
        if args.no_ptflops:
            flops_str, ptflops_params = "N/A", "N/A"
        else:
            flops_str, ptflops_params = get_flops_and_params_ptflops(model, input_res=(3, 32, 32))

        print(f"Trainable params: {trainable_params:,}")
        print(f"ptflops params:   {ptflops_params}")
        print(f"ptflops FLOPs:    {flops_str}")

        model.to(device)

        optimizer, scheduler = make_optimizer_and_scheduler(cfg, model, args.epochs, args.batch_size)

        best_val_acc = -1.0
        best_epoch = -1
        best_test_acc = -1.0
        best_test_loss = float("inf")
        best_state_cpu: Optional[Dict[str, torch.Tensor]] = None

        meter = GPUEnergyMeter(device_index=args.device_index, interval_s=0.2)

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()

            meter.start()
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device, grad_accum_steps=args.grad_accum
            )
            train_energy_j = meter.stop()

            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            scheduler.step()
            epoch_time_s = time.time() - t0
            lr_now = optimizer.param_groups[0]["lr"]

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                best_state_cpu = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            curves_row = {
                "model": cfg.name,
                "family": cfg.family,
                "scale": cfg.scale,
                "epoch": epoch,
                "lr": lr_now,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "train_err": 1.0 - train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_err": 1.0 - val_acc,
                "epoch_time_s": epoch_time_s,
                "train_energy_j": train_energy_j,
            }
            append_rows_to_csv(curves_path, [curves_row])

            if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
                print(
                    f"[{cfg.name}] epoch {epoch:03d}/{args.epochs} | "
                    f"lr {lr_now:.5g} | "
                    f"train acc {train_acc:.4f} (err {1-train_acc:.4f}) | "
                    f"val acc {val_acc:.4f} (err {1-val_acc:.4f}) | "
                    f"time {epoch_time_s:.1f}s | energy {train_energy_j:.1f}J"
                )

        assert best_state_cpu is not None
        model.load_state_dict(best_state_cpu)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        best_test_acc = test_acc
        best_test_loss = test_loss

        print(f"\nBest epoch by val acc: {best_epoch} | best val acc={best_val_acc:.4f}")
        print(f"Test at best epoch: test acc={best_test_acc:.4f}, test loss={best_test_loss:.4f}")

        summary_row = {
            "model": cfg.name,
            "family": cfg.family,
            "scale": cfg.scale,
            "trainable_params": trainable_params,
            "ptflops_params": ptflops_params,
            "ptflops_flops": flops_str,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "best_epoch": best_epoch,
            "best_val_acc": best_val_acc,
            "test_acc_at_best": best_test_acc,
            "test_loss_at_best": best_test_loss,
        }
        summary_rows.append(summary_row)
        append_rows_to_csv(summary_path, [summary_row])

        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\n" + "#" * 100)
    print(f"Done. Summary saved to: {summary_path}")
    print(f"Curves saved to:  {curves_path}")


if __name__ == "__main__":
    main()
