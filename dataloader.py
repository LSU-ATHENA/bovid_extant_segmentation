import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from functools import partial
from PIL import Image
import torch.nn.functional as F
from preprocess import PreProObj


class SegmentationPairDataset(Dataset):
    def __init__(self, raw_root, bw_root, binarize_mask = True, exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"), prepro_obj = None, args = None):
        self.raw_root = raw_root
        self.bw_root = bw_root
        self.binarize_mask = binarize_mask
        self.exts = {e.lower() for e in exts}
        self.prepro_obj = prepro_obj
        self.args = args

        raw_paths = [p for p in raw_root.rglob("*") if p.is_file() and p.suffix.lower() in self.exts]
        bw_by_name = {p.name: p for p in bw_root.rglob("*") if p.is_file() and p.suffix.lower() in self.exts}

        self.pairs = []
        for rp in raw_paths:
            bp = bw_by_name.get(rp.name)
            if bp is not None:
                self.pairs.append((rp, bp))

        self.pairs.sort(key = lambda t: t[0].name)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        x_path, y_path = self.pairs[idx]

        # I just changed a bunch of stuff here since it wasnt being used
        with Image.open(x_path) as img:
            x = np.array(img.convert("RGB"), dtype = np.uint8)
            
            x = self.prepro_obj.process(x)
            x = x.astype(np.float32)
            x /= 255.0
            x = torch.from_numpy(x).permute(2, 0, 1)  # [3, H, W]
            
            if self.args.downsample_factor > 1:
                x = F.interpolate(x.unsqueeze(0), scale_factor = 1 / self.args.downsample_factor, mode = 'bilinear', align_corners = False).squeeze(0)

        with Image.open(y_path) as mask:
            y = np.array(mask.convert("L"), dtype=np.uint8)
            y = (y == 0).astype(np.float32)  # background=255 -> 0, foreground=0 -> 1
            y = torch.from_numpy(y)
            y = y.unsqueeze(0) # [H, W] -> [1, H, W]
            
            if self.args.downsample_factor > 1:
                y = F.interpolate(y.unsqueeze(0), scale_factor = 1 / self.args.downsample_factor, mode = 'nearest').squeeze(0)

        return {"x": x, "y": y, "name": x_path.name}

def pad_collate_fn(batch, ignore_index=255, size_divisor=None):
    max_h = max(item["x"].shape[1] for item in batch)
    max_w = max(item["x"].shape[2] for item in batch)

    if size_divisor is not None and size_divisor > 1:
        max_h = ((max_h + size_divisor - 1) // size_divisor) * size_divisor
        max_w = ((max_w + size_divisor - 1) // size_divisor) * size_divisor

    xs, ys, names = [], [], []
    for item in batch:
        x, y = item["x"], item["y"]
        _, h, w = x.shape
        pad_h, pad_w = max_h - h, max_w - w

        x = F.pad(x, (0, pad_w, 0, pad_h), value=0.0)
        y = F.pad(y, (0, pad_w, 0, pad_h), value=0)

        xs.append(x)
        ys.append(y)
        names.append(item["name"])

    return {
        "x": torch.stack(xs, dim=0),  # [B, 3, Hmax, Wmax]
        "y": torch.stack(ys, dim=0),  # [B, Hmax, Wmax]
        "name": names
    }


def make_train_test_loaders(
    seg_dataset,
    train_ratio=0.8,
    seed=42,
    batch_size=8,
    num_workers=0,
    ignore_index=255,
    size_divisor=32,
):
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be between 0 and 1 (exclusive).")

    n_total = len(seg_dataset)
    n_train = int(n_total * train_ratio)
    n_test = n_total - n_train

    split_gen = torch.Generator().manual_seed(seed)
    train_dataset, test_dataset = torch.utils.data.random_split(
        seg_dataset, [n_train, n_test], generator=split_gen
    )

    collate = partial(pad_collate_fn, ignore_index=ignore_index, size_divisor=size_divisor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
    )

    return train_loader, test_loader


if __name__ == "__main__":
    from pathlib import Path

    raw_root = Path("filt_res_data/raw")
    bw_root = Path("filt_res_data/bw")

    dataset = SegmentationPairDataset(raw_root, bw_root, binarize_mask=True)
    print(f"Total pairs found: {len(dataset)}")

    train_loader, test_loader = make_train_test_loaders(dataset)

    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    print("\n--- Train loader ---")
    for i, batch in enumerate(train_loader):
        x, y, names = batch["x"], batch["y"], batch["name"]
        print(f"  Batch {i}: x={tuple(x.shape)} dtype={x.dtype}, y={tuple(y.shape)} dtype={y.dtype}, names={names}")

    print("\n--- Test loader ---")
    for i, batch in enumerate(test_loader):
        x, y, names = batch["x"], batch["y"], batch["name"]
        print(f"  Batch {i}: x={tuple(x.shape)} dtype={x.dtype}, y={tuple(y.shape)} dtype={y.dtype}, names={names}")