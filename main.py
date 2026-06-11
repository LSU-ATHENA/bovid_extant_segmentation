import argparse
from tqdm import tqdm
import torch as t
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from dataloader import SegmentationPairDataset, make_train_test_loaders
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn


def parse_args():
    parser = argparse.ArgumentParser(description="Bovid segmentation training")
    parser.add_argument("--model_name", type=str, default="UnetPlusPlus", help="Model architecture to use")
    parser.add_argument("--encoder_name", type=str, default="resnet18", help="Encoder backbone for UNet++")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--decay", type=float, default=1e-5, help="Weight decay for optimizer")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train/test split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers")
    parser.add_argument("--raw_data", type=str, default="filt_res_data/raw", help="Path to raw data")
    parser.add_argument("--mask_data", type=str, default="filt_res_data/bw", help="Path to mask data")
    parser.add_argument("--device", type=str, default="cuda" if t.cuda.is_available() else "cpu", help="Device to use")
    return parser.parse_args()


def dice_metric(logits, targets, eps=1e-7):
    preds = (t.sigmoid(logits) > 0.5).float()
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum()
    return (2 * intersection + eps) / (union + eps)


def train_epoch(model, train_loader, optimizer, bce_loss, dice_loss, device):
    model.train()
    total_bce_loss = 0.0
    total_dice_loss = 0.0
    pbar = tqdm(train_loader, desc="Train", leave=False)
    for batch in pbar:
        images, masks = batch['x'], batch['y']
        images, masks = images.to(device), masks.to(device).float()
        
        optimizer.zero_grad()
        outputs = model(images)
        bce = bce_loss(outputs, masks)
        dice = dice_loss(outputs, masks)
        loss = bce + dice
        
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


def eval_epoch(model, test_loader, bce_loss, dice_loss, device):
    model.eval()
    total_bce_loss = 0.0
    total_dice_loss = 0.0
    total_dice_metric = 0.0
    with t.no_grad():
        pbar = tqdm(test_loader, desc="Eval", leave=False)
        for batch in pbar:
            images, masks = batch['x'], batch['y']
            images, masks = images.to(device), masks.to(device).float()
            outputs = model(images)
            bce = bce_loss(outputs, masks)
            dice = dice_loss(outputs, masks)
            dice_m = dice_metric(outputs, masks)
            total_bce_loss += bce.item()
            total_dice_loss += dice.item()
            total_dice_metric += dice_m.item()
            pbar.set_postfix(
                bce_loss=f"{bce.item():.4f}",
                dice_loss=f"{dice.item():.4f}",
                dice_metric=f"{dice_m.item():.4f}",
            )
    
    return total_bce_loss / len(test_loader), total_dice_loss / len(test_loader), total_dice_metric / len(test_loader)


if __name__ == "__main__":
    args = parse_args()
    
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

    t.manual_seed(args.seed)
    
    raw_root = Path(args.raw_data)
    bw_root = Path(args.mask_data)
    
    preprocessing_fn = get_preprocessing_fn(args.encoder_name, pretrained="imagenet")
    dataset = SegmentationPairDataset(raw_root, bw_root, binarize_mask=True, preprocess_fns=[preprocessing_fn])
    print(f"Total pairs found: {len(dataset)}")

    train_loader, test_loader = make_train_test_loaders(dataset, train_ratio=args.train_ratio, 
                                                        seed=args.seed, batch_size=args.batch_size, 
                                                        num_workers=args.num_workers)

    if args.model_name == "UnetPlusPlus":
        model = smp.UnetPlusPlus(
            encoder_name=args.encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,                    # binary mask
            activation=None               # use logits + BCE/Dice
    )
    elif args.model_name == "Segformer":
        model = smp.Segformer(
            encoder_name=args.encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,                    # binary mask with 2 classes (background vs foreground)
            activation=None
        )
    else:
        raise ValueError(f"Unsupported model name: {args.model_name}")

    model = model.to(device)
    bce_loss = nn.BCEWithLogitsLoss()
    dice_loss = smp.losses.DiceLoss(mode="binary", from_logits=True)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_test_loss = float("inf")
    save_dir = Path("saved_models")
    save_dir.mkdir(parents=True, exist_ok=True)
    arg_kv = "_".join(
        f"{k}-{str(v).replace('/', '-').replace(' ', '')}"
        for k, v in vars(args).items()
    )
    save_path = save_dir / f"{arg_kv}.pt"

    print(f"Using device: {device}")
    for epoch in range(1, args.epochs + 1):
        train_bce_loss, train_dice_loss = train_epoch(model, train_loader, optimizer, bce_loss, dice_loss, device)
        test_bce_loss, test_dice_loss, test_dice_metric = eval_epoch(model, test_loader, bce_loss, dice_loss, device)

        print(
            f"Epoch [{epoch:03d}/{args.epochs:03d}] "
            f"train_bce_loss={train_bce_loss:.4f} train_dice_loss={train_dice_loss:.4f} "
            f"test_bce_loss={test_bce_loss:.4f} test_dice_loss={test_dice_loss:.4f} test_dice_metric={test_dice_metric:.4f}"
        )

        scheduler.step()

        if test_dice_metric > best_test_loss:
            best_test_loss = test_dice_metric
            t.save(
                {
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
            print(f"Saved new best checkpoint to {save_path} (test_loss={best_test_loss:.4f})")

    print(f"Training complete. Best test loss: {best_test_loss:.4f}")