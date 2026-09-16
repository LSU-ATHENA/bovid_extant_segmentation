import argparse
import math
import csv
from collections import defaultdict
from tqdm import tqdm
import psutil

import torch as t
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast

from pathlib import Path
from dataloader import SegmentationPairDataset, make_train_test_loaders

import segmentation_models_pytorch as smp

import preprocess

def parse_args():
    parser = argparse.ArgumentParser(description = "Bovid segmentation training")
    parser.add_argument("--model_name", type = str, default = "UnetPlusPlus", help = "Model architecture to use")
    parser.add_argument("--encoder_name", type = str, default = "resnet18", help = "Encoder backbone for UNet++")
    parser.add_argument("--epochs", type = int, default = 100, help = "Number of training epochs")
    parser.add_argument("--batch_size", type = int, default = 4, help = "Batch size")
    parser.add_argument("--learning_rate", type = float, default = 1e-3, help = "Learning rate")
    parser.add_argument("--decay", type = float, default = 1e-5, help = "Weight decay for optimizer")
    parser.add_argument("--train_ratio", type = float, default = 0.8, help = "Train/test split ratio")
    parser.add_argument("--seed", type = int, default = 42, help = "Random seed for reproducibility")
    parser.add_argument("--num_workers", type = int, default = 4, help = "Number of DataLoader workers")
    parser.add_argument("--raw_data", type = str, default = "filt_res_data/raw_aligned", help = "Path to raw data")
    parser.add_argument("--mask_data", type = str, default = "filt_res_data/bw", help = "Path to mask data")
    parser.add_argument("--device", type = str, default = "cuda" if t.cuda.is_available() else "cpu", help = "Device to use")
    parser.add_argument("--downsample_factor", type = int, default = 1, help = "Factor to downsample images/masks for faster training (e.g. 2 for half size)")
    parser.add_argument("--dice_scalar", type = float, default = 1, help = "Dice loss scalar (lambda)")
    parser.add_argument("--exp_name", type = str, default = "my_exp")
    parser.add_argument("--contrast", type = str, default = "none")
    parser.add_argument("--clahe_clip", type = int, default = 5)
    parser.add_argument("--device_ids", nargs = '+', type = int, help = 'GPUs to use')
    parser.add_argument("--stratify", action="store_true", help="Whether to stratify the train/test split based on taxonomy")
    parser.add_argument("--taxonomy_csv", type = str, default = "bovid_taxonomy.csv", help = "Path to taxonomy CSV used for stratification and per-class metrics")

    
    return parser.parse_args()


def load_filename_to_class(taxonomy_csv_path):
    taxonomy_path = Path(taxonomy_csv_path)
    if not taxonomy_path.exists():
        raise FileNotFoundError(f"Taxonomy CSV not found: {taxonomy_path}")

    mapping = {}
    with taxonomy_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            filename = row[0].strip()
            class_name = row[1].strip()
            if filename:
                mapping[filename.lower()] = class_name

    return mapping


def pad_to_multiple(x, divisor = 32):
    _, _, h, w = x.shape
    pad_h = (divisor - h % divisor) % divisor
    pad_w = (divisor - w % divisor) % divisor

    return t.nn.functional.pad(x, (0, pad_w, 0, pad_h))

def dice_metric(logits, targets, eps = 1e-7):
    preds = (t.sigmoid(logits) > 0.5).float()
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum()
   
    return (2 * intersection + eps) / (union + eps)

def train_epoch(model, train_loader, optimizer, bce_loss, dice_loss, device, dice_scalar):
    model.train()
    total_bce_loss = 0.0
    total_dice_loss = 0.0
   
    pbar = tqdm(train_loader, desc = "Train", leave = False)

    for batch in pbar:
        images, masks = batch['x'], batch['y']
        images, masks = images.to(device), masks.to(device).float()

        optimizer.zero_grad()
       
        with t.autocast(dtype = t.bfloat16, device_type = device.type):
            outputs = model(images)
                
        bce = bce_loss(outputs.float(), masks.float())
        dice = dice_loss(outputs.float(), masks.float())
        loss = bce + dice_scalar * dice
        
        if t.isnan(bce).item():
            print("NaN BCE Loss Warning! \n")
        if t.isnan(dice).item():
            print("NaN Dice loss Warning \n")
       
        loss.backward()
        # gradient clipping to prevent exploding gradients.
        t.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
       
        total_bce_loss += bce.item()
        total_dice_loss += dice.item()
        pbar.set_postfix(
            total_loss=f"{loss.item():.4f}",
            bce_loss=f"{bce.item():.4f}",
            dice_loss=f"{dice.item():.4f}",
        )
       
    return total_bce_loss / len(train_loader), total_dice_loss / len(train_loader)


def eval_epoch(model, test_loader, bce_loss, dice_loss, device, class_by_filename = None):
    model.eval()

    total_bce_loss = 0.0
    total_dice_loss = 0.0
    total_dice_metric = 0.0
    class_stats = defaultdict(lambda: {"intersection": 0.0, "pred_sum": 0.0, "target_sum": 0.0, "union": 0.0})
    eps = 1e-7
   
    i = 0
    with t.no_grad():
        pbar = tqdm(test_loader, desc = "Eval", leave = False)
       
        for batch in pbar:
            images, masks, names = batch['x'], batch['y'], batch['name']
            images, masks = images.to(device), masks.to(device).float()
           
            with t.autocast(dtype = t.bfloat16, device_type = device.type):
                outputs = model(images)

            bce = bce_loss(outputs.float(), masks.float())
            dice = dice_loss(outputs.float(), masks.float())
            dice_m = dice_metric(outputs.float(), masks.float())
            
            if class_by_filename is not None:
                preds = (t.sigmoid(outputs) > 0.5).float()
                for i, sample_name in enumerate(names):
                    class_name = class_by_filename.get(sample_name.lower())
                    if class_name is None:
                        raise ValueError(f"Missing taxonomy class for filename: {sample_name}")

                    pred_i = preds[i]
                    target_i = masks[i]
                    intersection = (pred_i * target_i).sum().item()
                    pred_sum = pred_i.sum().item()
                    target_sum = target_i.sum().item()
                    union = pred_sum + target_sum - intersection

                    class_stats[class_name]["intersection"] += intersection
                    class_stats[class_name]["pred_sum"] += pred_sum
                    class_stats[class_name]["target_sum"] += target_sum
                    class_stats[class_name]["union"] += union
           
            total_bce_loss = total_bce_loss + bce.item()
            total_dice_loss = total_dice_loss + dice.item()
            total_dice_metric = total_dice_metric + dice_m.item()
           
            pbar.set_postfix(
                bce_loss=f"{bce.item():.4f}",
                dice_loss=f"{dice.item():.4f}",
                dice_metric=f"{dice_m.item():.4f}",
            )

    per_class_metrics = None
    if class_by_filename is not None:
        per_class_metrics = {}
        for class_name, stats in class_stats.items():
            iou = (stats["intersection"] + eps) / (stats["union"] + eps)
            dice = (2.0 * stats["intersection"] + eps) / (stats["pred_sum"] + stats["target_sum"] + eps)
            per_class_metrics[class_name] = {"miou": iou, "dice": dice}

    return (
        total_bce_loss / len(test_loader),
        total_dice_loss / len(test_loader),
        total_dice_metric / len(test_loader),
        per_class_metrics,
    )


if __name__ == "__main__":
    args = parse_args()
    #preprocess.model_args = args
    
    # CUDA is Nvidia GPU
    # MPS is Apple GPU (e.g. M1/M2)
    if args.device == "cuda" and not t.cuda.is_available():
        if hasattr(t.backends, "mps") and t.backends.mps.is_available():
            device = t.device("mps")
        else:
            device = t.device("cpu")
    elif args.device == "mps" and (not hasattr(t.backends, "mps") or not t.backends.mps.is_available()):
        device = t.device("cpu")
    else:
        device = t.device(args.device)
        
    if args.device_ids and device.type == "cuda":
        device = t.device(f"cuda:{args.device_ids[0]}")

    t.manual_seed(args.seed)
   
    raw_root = Path(args.raw_data)
    bw_root = Path(args.mask_data)
    
    prepro_obj = preprocess.PreProObj(args = args)
    prepro_obj.setup_stack(stack = [prepro_obj.apply_contrast])
    dataset = SegmentationPairDataset(raw_root, bw_root, binarize_mask = True, prepro_obj = prepro_obj, args = args)
    print(f"Total pairs found: {len(dataset)}")

    train_loader, test_loader = make_train_test_loaders(
        dataset,
        train_ratio = args.train_ratio,
        seed = args.seed,
        batch_size = args.batch_size,
        num_workers = args.num_workers,
        stratify = args.stratify,
        taxonomy_csv_path = args.taxonomy_csv,
    )

    class_by_filename = load_filename_to_class(args.taxonomy_csv) if args.stratify else None

    if args.model_name == "UnetPlusPlus":
        # binary mask, use logits + BCE/Dice
        model = smp.UnetPlusPlus(encoder_name = args.encoder_name, encoder_weights = "imagenet", in_channels = 3, classes = 1, activation = None)
    elif args.model_name == "Segformer":
        # binary mask with 2 classes (background vs foreground)
        model = smp.Segformer(encoder_name = args.encoder_name, encoder_weights = "imagenet", in_channels = 3, classes = 1, activation = None)
    else:
        raise ValueError(f"Unsupported model name: {args.model_name}")

    model = model.to(device)
    bce_loss = nn.BCEWithLogitsLoss()
    dice_loss = smp.losses.DiceLoss(mode = "binary", from_logits = True)
   
    optimizer = optim.AdamW(model.parameters(), lr = args.learning_rate, weight_decay = args.decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = args.epochs)

    best_test_metric = float("-inf")
    save_dir = Path("saved_models")
    save_dir.mkdir(parents = True, exist_ok = True)
    save_path = save_dir / f"{args.exp_name}.pt"
    
    model = nn.DataParallel(model, args.device_ids)

    print(f"Using device: {device}")
    for epoch in range(1, args.epochs + 1):
        train_bce_loss, train_dice_loss = train_epoch(model, train_loader, optimizer, bce_loss, dice_loss, device, args.dice_scalar)
        test_bce_loss, test_dice_loss, test_dice_metric, per_class_metrics = eval_epoch(
            model,
            test_loader,
            bce_loss,
            dice_loss,
            device,
            class_by_filename = class_by_filename,
        )

        print(
            f"Epoch [{epoch:03d}/{args.epochs:03d}] "
            f"train_bce_loss={train_bce_loss:.4f} train_dice_loss={train_dice_loss:.4f} "
            f"test_bce_loss={test_bce_loss:.4f} test_dice_loss={test_dice_loss:.4f} test_dice_metric={test_dice_metric:.4f}"
            f"free memory is {psutil.virtual_memory().available / (1024 ** 3)}"
        )

        if per_class_metrics is not None:
            mean_class_iou = sum(m["miou"] for m in per_class_metrics.values()) / len(per_class_metrics)
            mean_class_dice = sum(m["dice"] for m in per_class_metrics.values()) / len(per_class_metrics)
            print(f"  Stratified class means: mIoU={mean_class_iou:.4f} Dice={mean_class_dice:.4f}")
            for class_name in sorted(per_class_metrics):
                metrics = per_class_metrics[class_name]
                print(f"    {class_name}: mIoU={metrics['miou']:.4f} Dice={metrics['dice']:.4f}")

        scheduler.step()

        if test_dice_metric > best_test_metric:
            best_test_metric = test_dice_metric
            t.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "test_bce_loss": test_bce_loss,
                    "test_dice_loss": test_dice_loss,
                    "test_dice_metric": test_dice_metric,
                    "args": vars(args),
                },
                save_path,
            )
            print(f"Saved new best checkpoint to {save_path} (test_metric={best_test_metric:.4f})")

    print(f"Training complete. Best test metric: {best_test_metric:.4f}")
