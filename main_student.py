"""
Student exercise: Binary image segmentation training.

You are given:
- images
- binary masks (0 = background, 1 = object)

Your task:
- complete the missing pieces
- train a model that segments objects correctly
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path
import segmentation_models_pytorch as smp

from dataloader import SegmentationPairDataset, make_train_test_loaders

# Define dice metric
def dice_metric(logits, targets, eps = 1e-7):
    """
    Compute Dice coefficient (NOT loss).

    logits: raw model outputs
    targets: ground-truth masks in {0,1}
    """
    # TODO:
    # 1. Convert logits to probabilities
    # 2. Threshold probabilities at 0.5
    # 3. Compute Dice coefficient
    
    f_logits = logits.flatten()
    f_targets = targets.flatten()
    
    probabilities = torch.sigmoid(f_logits)
    mask = torch.zeros_like(probabilities, dtype = torch.int32)
    
    num_intersections = 0
    
    threshold = 0.5
    mask = torch.where(mask >= threshold, 1, 0)
    
    num_intersections = torch.where(mask == f_targets, 1, 0).sum()
    
    
    """
    for i in range(len(mask)):
        
        if probabilities[i] >= 0.5:
            mask[i] = 1
        else:
            mask[i] = 0
    
        if mask[i] == f_targets[i]:
            num_intersections += 1"""
    
    dice_c = (2.0 * num_intersections) / (mask.sum() + f_targets.sum())
    return dice_c

# Training loop
def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0

    for i, batch in enumerate(tqdm(loader, desc = "Train")):
        images = batch["x"] 
        masks = batch["y"]

        # TODO 2:
        # Move tensors to device and ensure masks are a float
        images = images.to(device)
        
        masks = masks.to(device, torch.float)

        optimizer.zero_grad()

        # TODO 3:
        # Forward pass
        outputs = model(images)

        # TODO 4:
        # Compute loss
        loss = loss_fn(outputs, masks)

        # TODO 5:
        # Backward pass and optimizer step
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
        if i > 9:
            break

    return total_loss / len(loader)


# -------------------------
# Evaluation loop
# -------------------------
def eval_epoch(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval"):
            images, masks = batch["x"], batch["y"]

            # TODO 6:
            # - move tensors to device
            # - ensure masks are float
            images = images.to(device)
            
            masks = masks.to(device)
            masks = masks.to(torch.float)

            outputs = model(images)

            # TODO 7:
            # - compute loss
            loss = loss_fn(outputs, masks)

            # TODO 8:
            # - compute Dice metric
            dice = dice_metric(outputs, masks)

            total_loss += loss.item()
            total_dice += dice.item()

    return (total_loss / len(loader), total_dice / len(loader),)


# -------------------------
# Main
# -------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Dataset
    dataset = SegmentationPairDataset(Path("filt_res_data/raw"), Path("filt_res_data/bw"), binarize_mask = True,)
    train_loader, val_loader = make_train_test_loaders(dataset, train_ratio = 0.8, batch_size = 2,)
    
    # Model
    model = smp.UnetPlusPlus(encoder_name = "mobilenet_v2", encoder_weights = "imagenet", in_channels = 3, classes = 1, activation = None,).to(device)

    # TODO 9:
    # - define loss function
    #   (hint: BCEWithLogitsLoss)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # TODO 10:
    # - define optimizer (Adam or AdamW)
    optimizer = optim.AdamW(model.parameters())

    for epoch in range(1, 21):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_dice = eval_epoch(model, val_loader, loss_fn, device)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_dice={val_dice:.4f}"
        )


if __name__ == "__main__":
    main()