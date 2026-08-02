import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from functools import partial
from PIL import Image
import torch.nn.functional as F
import csv
from pathlib import Path


class ClassificationDataset(Dataset):
    def __init__(
        self,
        raw_root,
        taxonomy_csv_path="bovid_taxonomy.csv",
        exts=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"),
        prepro_obj=None,
        args=None,
    ):
        self.raw_root = Path(raw_root)
        self.taxonomy_csv_path = Path(taxonomy_csv_path)
        self.exts = {e.lower() for e in exts}
        self.prepro_obj = prepro_obj
        self.args = args

        if not self.raw_root.exists() or not self.raw_root.is_dir():
            raise FileNotFoundError(
                f"Raw data directory not found: {self.raw_root}. "
                "Check --raw_data and avoid unresolved placeholders like '{raw_data}'."
            )

        if not self.taxonomy_csv_path.exists():
            raise FileNotFoundError(f"Taxonomy CSV not found: {self.taxonomy_csv_path}")

        self.samples = []
        self.class_to_idx = {}
        self.idx_to_class = []

        filename_to_class = {}
        conflicts = []
        with self.taxonomy_csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = (row.get("filename") or "").strip()
                class_name = (row.get("class") or "").strip()

                if not filename or not class_name:
                    continue
                if Path(filename).suffix.lower() not in self.exts:
                    continue

                key = filename.lower()
                prev = filename_to_class.get(key)
                if prev is None:
                    filename_to_class[key] = class_name
                elif prev != class_name:
                    conflicts.append((filename, prev, class_name))

        if conflicts:
            ex = conflicts[0]
            raise ValueError(
                "Conflicting class assignments in taxonomy CSV for filename "
                f"'{ex[0]}': '{ex[1]}' vs '{ex[2]}'."
            )

        all_files = [p for p in self.raw_root.rglob("*") if p.is_file() and p.suffix.lower() in self.exts]
        all_files.sort(key=lambda p: p.name.lower())

        unlabeled_files = []
        for image_path in all_files:
            filename = image_path.name
            class_name = filename_to_class.get(filename.lower())
            if class_name is None:
                unlabeled_files.append(filename)
                continue

            if class_name not in self.class_to_idx:
                self.class_to_idx[class_name] = len(self.idx_to_class)
                self.idx_to_class.append(class_name)
            class_idx = self.class_to_idx[class_name]

            self.samples.append((image_path, class_idx, class_name, filename))

        self.num_unlabeled_files = len(unlabeled_files)
        self.num_csv_labels = len(filename_to_class)
        self.num_files_scanned = len(all_files)

        if not self.samples:
            raise ValueError(
                "No classification samples were loaded. "
                f"Scanned {self.num_files_scanned} files under '{self.raw_root}' and found 0 filename matches in taxonomy CSV."
            )

        self.samples.sort(key=lambda t: t[3].lower())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x_path, class_idx, class_name, filename = self.samples[idx]

        with Image.open(x_path) as img:
            x = np.array(img.convert("RGB"), dtype = np.uint8)

            if self.prepro_obj is not None:
                x = self.prepro_obj.process(x)
            x = x.astype(np.float32)
            x /= 255.0
            x = torch.from_numpy(x).permute(2, 0, 1)

            if self.args is not None and getattr(self.args, "downsample_factor", 1) > 1:
                x = F.interpolate(x.unsqueeze(0), scale_factor = 1 / self.args.downsample_factor, mode = 'bilinear', align_corners = False).squeeze(0)

        class_label = torch.tensor(class_idx, dtype=torch.long)

        return {"x": x, "class": class_label, "name": filename, "class_name": class_name}


# Backward-compatible alias for older imports.
SegmentationPairDataset = ClassificationDataset


def pad_collate_fn(batch, size_divisor=None):
    max_h = max(item["x"].shape[1] for item in batch)
    max_w = max(item["x"].shape[2] for item in batch)

    if size_divisor is not None and size_divisor > 1:
        max_h = ((max_h + size_divisor - 1) // size_divisor) * size_divisor
        max_w = ((max_w + size_divisor - 1) // size_divisor) * size_divisor

    xs, class_labels, names, class_names = [], [], [], []
    for item in batch:
        x = item["x"]
        _, h, w = x.shape
        pad_h, pad_w = max_h - h, max_w - w

        x = F.pad(x, (0, pad_w, 0, pad_h), value=0.0)

        xs.append(x)
        class_labels.append(item["class"])
        names.append(item["name"])
        class_names.append(item["class_name"])

    return {
        "x": torch.stack(xs, dim=0),
        "class": torch.stack(class_labels, dim=0),
        "name": names,
        "class_name": class_names,
    }


def _stratified_split_indices(dataset, train_ratio, seed):
    class_to_indices = {}

    for idx, (_, class_idx, _, _) in enumerate(dataset.samples):
        class_to_indices.setdefault(class_idx, []).append(idx)

    n_total = len(dataset)
    n_train_target = int(n_total * train_ratio)

    rng = np.random.default_rng(seed)
    shuffled_per_class = {}
    train_counts = {}
    fractional_parts = []

    for class_name, indices in class_to_indices.items():
        shuffled = list(rng.permutation(indices))
        shuffled_per_class[class_name] = shuffled

        exact = len(indices) * train_ratio
        base = int(np.floor(exact))
        train_counts[class_name] = base
        fractional_parts.append((exact - base, class_name))

    remaining = n_train_target - sum(train_counts.values())
    if remaining > 0:
        fractional_parts.sort(key=lambda t: t[0], reverse=True)
        for _, class_name in fractional_parts:
            if remaining <= 0:
                break
            if train_counts[class_name] < len(shuffled_per_class[class_name]):
                train_counts[class_name] += 1
                remaining -= 1

    train_indices = []
    test_indices = []
    for class_name, shuffled in shuffled_per_class.items():
        n_train_class = train_counts[class_name]
        train_indices.extend(shuffled[:n_train_class])
        test_indices.extend(shuffled[n_train_class:])

    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    return train_indices, test_indices


def make_train_test_loaders(
    cls_dataset,
    train_ratio=0.8,
    seed=42,
    batch_size=8,
    num_workers=0,
    size_divisor=32,
    stratify=False,
):
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be between 0 and 1 (exclusive).")

    n_total = len(cls_dataset)
    n_train = int(n_total * train_ratio)
    n_test = n_total - n_train

    if stratify:
        train_indices, test_indices = _stratified_split_indices(
            dataset=cls_dataset,
            train_ratio=train_ratio,
            seed=seed,
        )
        train_dataset = torch.utils.data.Subset(cls_dataset, train_indices)
        test_dataset = torch.utils.data.Subset(cls_dataset, test_indices)
    else:
        split_gen = torch.Generator().manual_seed(seed)
        train_dataset, test_dataset = torch.utils.data.random_split(
            cls_dataset, [n_train, n_test], generator=split_gen
        )

    collate = partial(pad_collate_fn, size_divisor=size_divisor)

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

    dataset = ClassificationDataset(raw_root=raw_root, taxonomy_csv_path="bovid_taxonomy.csv")
    print(f"Total classification samples found: {len(dataset)}")
    print(f"Num classes: {len(dataset.idx_to_class)}")

    train_loader, test_loader = make_train_test_loaders(dataset, stratify=True)

    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    print("\n--- Train loader ---")
    for i, batch in enumerate(train_loader):
        x, class_labels, names, class_names = batch["x"], batch["class"], batch["name"], batch["class_name"]
        print(f"  Batch {i}: x={tuple(x.shape)} dtype={x.dtype}, class={tuple(class_labels.shape)} dtype={class_labels.dtype}, names={names}, class_names={class_names}")

    print("\n--- Test loader ---")
    for i, batch in enumerate(test_loader):
        x, class_labels, names, class_names = batch["x"], batch["class"], batch["name"], batch["class_name"]
        print(f"  Batch {i}: x={tuple(x.shape)} dtype={x.dtype}, class={tuple(class_labels.shape)} dtype={class_labels.dtype}, names={names}, class_names={class_names}")