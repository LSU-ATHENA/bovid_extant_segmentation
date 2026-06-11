"""
Student exercise: Inference for a trained segmentation model.

Goal:
- load a trained model
- run it on new images
- save predicted masks
"""

import torch
import numpy as np
import cv2
from pathlib import Path
import segmentation_models_pytorch as smp


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TODO 1:
    # - load model architecture
    
    # By god I hope this is the architecture 
    model = smp.UnetPlusPlus(encoder_name = "mobilenet_v2", encoder_weights = "imagenet", in_channels = 3, classes = 1, activation = None,)

    # TODO 2:
    # - load checkpoint weights
    checkpoint = torch.load("model.pt", map_location = device)
    model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    input_dir = Path("new_images")
    output_dir = Path("predicted_masks")
    output_dir.mkdir(exist_ok = True)

    with torch.no_grad():
        for img_path in input_dir.glob("*.png"):
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # TODO 3:
            # - convert image to tensor
            # - add batch dimension
            img_tensor = torch.from_numpy(img)
            img_tensor = img_tensor.unsqueeze(0) # I am genuinely confused by why this needs a batch and I have a feeling im about to cause an error here
            
            
            # TODO 4:
            # - forward pass
            logits = model(img_tensor) # Im assuming the input isnt a 4D tensor but the 3D Image so ???????? I hope theres no error

            # TODO 5:
            # - convert logits to binary mask
            threshold = 0.5
            mask = torch.sigmoid(logits)
            mask = torch.where(mask >= threshold, 1, 0)

            mask = (mask * 255).astype(np.uint8)
            cv2.imwrite(str(output_dir / f"{img_path.stem}_mask.png"), mask)


if __name__ == "__main__":
    main()