"""
Student exercise: Binary image segmentation training.

You are given:
- images
- binary masks (0 = background, 1 = object)

Your task:
- complete the missing pieces
- train a model that segments objects correctly
"""

import torch as t
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path
import segmentation_models_pytorch as smp

from dataloader import SegmentationPairDataset, make_train_test_loaders


# -------------------------
# TODO 1: Define Dice metric
# -------------------------
def dice_metric(logits, targets, eps=1e-7):
    """
    Compute Dice coefficient (NOT loss).

    logits: raw model outputs
    targets: ground-truth masks in {0,1}
    """
    # TODO:
    # 1. Convert logits to probabilities
    # 2. Threshold probabilities at 0.5
    # 3. Compute Dice coefficient
    pass


# -------------------------
# Training loop
# -------------------------
def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="Train"):
        images, masks = batch["x"], batch["y"]

        # TODO 2:
        # - move tensors to device
        # - ensure masks are float
        images = ...
        masks = ...

        optimizer.zero_grad()

        # TODO 3:
        # - forward pass
        outputs = ...

        # TODO 4:
        # - compute loss
        loss = ...

        # TODO 5:
        # - backward pass
        # - optimizer step
        ...

        total_loss += loss.item()

    return total_loss / len(loader)


# -------------------------
# Evaluation loop
# -------------------------
def eval_epoch(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0

    with t.no_grad():
        for batch in tqdm(loader, desc="Eval"):
            images, masks = batch["x"], batch["y"]

            # TODO 6:
            # - move tensors to device
            # - ensure masks are float
            images = ...
            masks = ...

            outputs = model(images)

            # TODO 7:
            # - compute loss
            loss = ...

            # TODO 8:
            # - compute Dice metric
            dice = ...

            total_loss += loss.item()
            total_dice += dice.item()

    return (
        total_loss / len(loader),
        total_dice / len(loader),
    )


# -------------------------
# Main
# -------------------------
def main():
    device = t.device("cuda" if t.cuda.is_available() else "cpu")

    # Dataset
    dataset = SegmentationPairDataset(
        Path("data/images"),
        Path("data/masks"),
        binarize_mask=True,
    )

    train_loader, val_loader = make_train_test_loaders(
        dataset,
        train_ratio=0.8,
        batch_size=4,
    )

    # Model
    model = smp.UnetPlusPlus(
        encoder_name="mobilenet_v2",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None,
    ).to(device)

    # TODO 9:
    # - define loss function
    #   (hint: BCEWithLogitsLoss)
    loss_fn = ...

    # TODO 10:
    # - define optimizer (Adam or AdamW)
    optimizer = ...

    for epoch in range(1, 21):
        train_loss = train_epoch(
            model, train_loader, optimizer, loss_fn, device
        )
        val_loss, val_dice = eval_epoch(
            model, val_loader, loss_fn, device
        )

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_dice={val_dice:.4f}"
        )


if __name__ == "__main__":
    main()