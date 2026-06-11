import argparse
from pathlib import Path
import torch as t
import numpy as np
import cv2
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Inference for tooth segmentation")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory with input images")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save predicted masks")
    parser.add_argument("--model_name", type=str, default="UnetPlusPlus")
    parser.add_argument("--encoder_name", type=str, default="resnet18")
    parser.add_argument("--device", type=str,
                        default="cuda" if t.cuda.is_available() else "cpu")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Sigmoid threshold for binary mask")
    return parser.parse_args()


def load_model(args, device):
    if args.model_name == "UnetPlusPlus":
        model = smp.UnetPlusPlus(
            encoder_name=args.encoder_name,
            encoder_weights=None,   # weights come from checkpoint
            in_channels=3,
            classes=1,
            activation=None,
        )
    elif args.model_name == "Segformer":
        model = smp.Segformer(
            encoder_name=args.encoder_name,
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None,
        )
    else:
        raise ValueError(f"Unsupported model: {args.model_name}")

    checkpoint = t.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def main():
    args = parse_args()

    # Device logic (mirrors training)
    if args.device == "cuda" and not t.cuda.is_available():
        if hasattr(t.backends, "mps") and t.backends.mps.is_available():
            device = t.device("mps")
        else:
            device = t.device("cpu")
    else:
        device = t.device(args.device)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    preprocessing_fn = get_preprocessing_fn(
        args.encoder_name, pretrained="imagenet"
    )

    model = load_model(args, device)

    image_paths = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    )

    print(f"Running inference on {len(image_paths)} images")
    print(f"Saving masks to {output_dir}")
    print(f"Using device: {device}")

    with t.no_grad():
        for img_path in tqdm(image_paths):
            # Load image (BGR → RGB)
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                print(f"Warning: could not read {img_path}")
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, _ = img.shape

            # Preprocess
            img = preprocessing_fn(img)
            img = t.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
            img = img.float().to(device)

            # Forward
            logits = model(img)
            probs = t.sigmoid(logits)

            mask = (probs > args.threshold).float()
            mask = mask.squeeze().cpu().numpy()

            # Convert to 0–255 uint8
            mask = (mask * 255).astype(np.uint8)

            out_path = output_dir / f"{img_path.stem}_mask.png"
            cv2.imwrite(str(out_path), mask)

    print("Inference complete.")


if __name__ == "__main__":
    main()