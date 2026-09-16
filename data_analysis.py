import matplotlib.pyplot as plt
import argparse
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--model_names", nargs = "+", help = "Models to generate stats for")
args = parser.parse_args()

valid_metrics = ["test_bce_loss", "test_dice_metric", "test_dice_loss", "mIoU", "Dice"]

for path in args.model_names:
    with open(path + ".txt", "r", encoding = "utf-8") as file:
        items = file.read().split()
        metric_map = { }
        epochs = 0
        
        for item in items:
            if item == "Epoch":
                epochs += 1
                
        for metric in valid_metrics:
            metric_map[metric] = np.zeros(epochs)
                
        stratify_guard = False
        stratify_params = ["mIoU", "Dice"]
        cur_epoch = 0
                
        for i, item in enumerate(items):
            if item == "Epoch":
                stratify_guard = False
                cur_epoch += 1
            if "=" in item:
                subitems = item.split("=")
                if subitems[0] not in valid_metrics:
                    continue
                
                if ((subitems[0] in stratify_params) and (not stratify_guard)):
                    stratify_guard = True
                    # Do stuff
                else:
                    metric_map[subitems[0]][cur_epoch - 1] = subitems[1].replace("free", "")

        new_path_name = "model_plots/" + path + "_graphs/"
        Path(new_path_name).mkdir(parents = True, exist_ok = True)
        
        for metric in valid_metrics:
            if metric_map.get(metric) is not None:
                    # Create a plot and save to dir
                    plt.plot(metric_map[metric], marker = "o", linestyle = "--", color = "b")
                    plt.title("epoch vs " + metric)
                    plt.xlabel("epoch")
                    plt.ylabel(metric)
                    
                    plt.savefig(new_path_name + metric, dpi = 300, bbox_inches = "tight")
                    