"""
Student exercise: Inference for a trained segmentation model.

Goal:
- load a trained model
- run it on new images
- save predicted masks
"""

import torch as t
import numpy as np
import cv2
from pathlib import Path
import segmentation_models_pytorch as smp


def main():
    device = t.device("cuda" if t.cuda.is_available() else "cpu")

    # TODO 1:
    # - load model architecture
    model = ...

    # TODO 2:
    # - load checkpoint weights
    checkpoint = t.load("model.pt", map_location=device)
    model.load_state_dict(...)

    model.to(device)
    model.eval()

    input_dir = Path("new_images")
    output_dir = Path("predicted_masks")
    output_dir.mkdir(exist_ok=True)

    with t.no_grad():
        for img_path in input_dir.glob("*.png"):
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # TODO 3:
            # - convert image to tensor
            # - add batch dimension
            img_tensor = ...

            # TODO 4:
            # - forward pass
            logits = ...

            # TODO 5:
            # - convert logits to binary mask
            mask = ...

            mask = (mask * 255).astype(np.uint8)
            cv2.imwrite(
                str(output_dir / f"{img_path.stem}_mask.png"),
                mask
            )


if __name__ == "__main__":
    main()