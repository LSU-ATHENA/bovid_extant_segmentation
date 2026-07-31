import os
import matplotlib.pyplot as plot
import numpy as np

NUM_EPOCHS = 100
USED_MODELS = ["resnet34_good_lr3_ld3", "resnet34_good_lr3_ld5", "resnet34_good_lr4_ld3", "resnet34_good_lr4_ld5", "resnet34_lr3_ld3", "resnet34_lr3_ld5", "resnet34_lr4_ld3", "resnet34_lr4_ld5"]

index_lib = { "train_bce_loss" : 0, "train_dice_loss" : 1, "test_bce_loss" : 2, "test_dice_loss" : 3, "test_dice_metric" : 4 }
label_lib = { value: key for key, value in index_lib.items() }

for model_idx in range(len(USED_MODELS)):
    curModel = USED_MODELS[model_idx]
    
    folder_name = curModel
    os.mkdir("plots/" + folder_name)
    
    used_epochs = 0
    loss_data = np.zeros((5, NUM_EPOCHS))

    # Read data into 2d array
    with open(curModel + ".txt", mode = "r", encoding = "utf-8") as file:
        lines = file.readlines()
        epoch = 0
        
        for line in lines:
            if "Epoch" not in line: 
                continue
            
            if epoch + 1 > NUM_EPOCHS:
                break
            
            epoch += 1
            parse = line.split()
            
            for word in parse:
                if "=" not in word:
                    continue
                
                # Assuming all numerical values will contain an equals sign
                data = word.split("=")
                category = data[0]
                num = float(data[1])
                
                loc = index_lib[category]
                loss_data[loc, epoch - 1] = num
        
        used_epochs = epoch

    # Generate plots
    for i in range(len(label_lib)):
        data = loss_data[i, :used_epochs]
        label = label_lib[i]
        figure, data_plot = plot.subplots()
        
        data_plot.set_title(label + " vs Epoch")
        data_plot.set_xlabel("Epochs (#)")
        data_plot.set_ylabel(label + " (#)")
        
        data_plot.plot(data)
        
        figure.tight_layout()
        figure.savefig("plots/" + folder_name + "/" + label + ".png")
        
        plot.close(figure)
