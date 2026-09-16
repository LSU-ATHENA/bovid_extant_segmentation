import torch
from fvcore.nn import FlopCountAnalysis
import argparse
import segmentation_models_pytorch as smp

def count_flops(model, height = 512, width = 512):
    model.eval()
    device = next(model.parameters()).device
    x = torch.randn(1, 3, height, width, device = device)

    with torch.no_grad():
        flops = FlopCountAnalysis(model, x)

    print(f"FLOPs: {flops.total()/1e9:.3f} GFLOPs")
    print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.3f} M")
    
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type = str, default = "UnetPlusPlus")
parser.add_argument("--encoder_name", type = str)
parser.add_argument("--checkpoint", type = str)
args = parser.parse_args()
    
if args.model_name == "UnetPlusPlus":
    # binary mask, use logits + BCE/Dice
    model = smp.UnetPlusPlus(encoder_name = args.encoder_name, encoder_weights = "imagenet", in_channels = 3, classes = 1, activation = None)
elif args.model_name == "Segformer":
    # binary mask with 2 classes (background vs foreground)
    model = smp.Segformer(encoder_name = args.encoder_name, encoder_weights = "imagenet", in_channels = 3, classes = 1, activation = None)
else:
    raise ValueError(f"Unsupported model name: {args.model_name}")
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(args.checkpoint, map_location = device)
state_dict = checkpoint["model_state_dict"]

if any(k.startswith("module.") for k in state_dict.keys()):
    state_dict = { k.replace("module.", "", 1): v for k, v in state_dict.items() }

model.load_state_dict(state_dict)
model.to(device)
model.eval()

count_flops(model)