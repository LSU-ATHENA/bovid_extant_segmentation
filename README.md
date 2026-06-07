# TODO: Make this README better

## What each file does
```
data_filter.py
```
Takes 2 directories, e.g., the 'raw' and 'bw' folders, and looks to see how many files exist between both of them.

Sample usage:
```
python data_filter.py Tragelaphini\ raw/images/Extant/Tragelaphini/Tragelaphus/strepsiceros/ Tragelaphini\ bw/images/Extant/Tragelaphini/Tragelaphus/strepsiceros
```

`dataloader.py` - This does the work with our data. You should study it, but not change it. 

`main_student.py` Script for training a model. You will need to complete it. Later, Keith will add main.py which is more formal. Sample command:
```
python main.py --epochs 1 --device mps --batch_size 4 --seed 20 --encoder_name mobilenet_v2
```
`mps` is for Mac; you would not use it on Nvidia silicon.

`infer_student.py` Script for performing inference. You will need to complete to the best of your abilities and later Keith will provide a more formal script.

## Useful Commands
Given a directory (bw), go through it recursively and find the number of .jpg and .png files that exist:
```
find bw -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | wc -l
```