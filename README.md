# B.O.V.I.D. tooth segmentation and taxonomy classification

This repository trains models on bovid tooth photographs. The segmentation task predicts a binary mask of the occlusal (chewing) surface using U-Net++ or SegFormer. The photographs have hand-made masks originally prepared for morphometric analysis, so alignment and annotation quality matter. A separate training script classifies photographs by taxonomic tribe.

Paper: [Segmentation of Bovid Dentition Under Imperfect Annotations: A Comparative Study of Convolutional and Attention Models](https://arxiv.org/abs/2608.31052).

Run the commands below from the repository root. Examples are based on the previous student's recorded workflow, with corrected flag spelling, matching model/encoder settings, and explicit experiment names.

## Branches

- `main` is the primary branch for ATHENA-Metis.
- `eris-branch` is a variant from another lab machine. The classification code matches `main`.
- `evanbranch` is an older snapshot.

## Environment setup

The lab machine, ATHENA-Metis, is an NVIDIA DGX Spark with an ARM/aarch64 processor, a GB10 GPU, CUDA 13, and unified CPU/GPU memory. `environment.yml` is an exported environment for this platform. A generic `pip install torch` may not provide a working GPU build here; use the existing lab environment when available.

If Conda is already installed but shell activation has not been configured, run this **once**, then open a new terminal:

```bash
conda init bash
```

To create the environment when it does not already exist:

```bash
conda env create --name bovid_extant_env --file environment.yml
```

The explicit name selects the local environment rather than relying on the exported machine-specific `prefix`. Activate it in each new terminal or tmux session:

```bash
conda activate bovid_extant_env
```

Before training, verify that PyTorch sees CUDA:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"
```

If CUDA is unavailable, check with the lab before replacing packages. Use user-managed environments; do not use `sudo` or system-wide installs on the shared machine.

## Data

| Path | Contents / use |
| --- | --- |
| `filt_res_data/raw_aligned` | Aligned RGB tooth photographs; default image directory for both training scripts. |
| `filt_res_data/bw` | Black/white target masks; default mask directory for segmentation. |
| `bovid_taxonomy.csv` | Columns `filename`, `class`, and `path`; labels for classification and stratified segmentation. |

On the lab checkout, `filt_res_data` is a symlink to shared data. Check that it resolves and both directories are accessible:

```bash
ls -ld filt_res_data filt_res_data/raw_aligned filt_res_data/bw
```

The segmentation loader searches recursively and pairs photographs with masks by exact filename, including the extension and case. It converts black mask pixels to tooth = 1 and other pixels to background = 0. Images and masks are downsampled, then batches are padded to multiples of 32.

The taxonomy classes represent seven tribes: Alcelaphini, Antilopini, Bovini, Hippotragini, Neotragini, Reduncini, and Tragelaphini. CSV labels include a suffix such as `Neotragini raw`; scripts use these strings as written. Classification matches labels by filename without regard to case and skips images missing from the CSV. Stratified segmentation requires a CSV label for every paired image. The CSV `path` column is not used to locate images by either training script.

## Segmentation training: `mainbf16.py`

Training uses bfloat16 mixed precision, AdamW, cosine learning-rate scheduling, and the loss `BCE + dice_scalar * Dice`. The default train/test split is 80/20. The best checkpoint, selected by test Dice, is saved to `saved_models/<exp_name>.pt` and includes model weights, optimizer state, metrics, and training arguments.

These are long runs. Check resources and start a tmux session as described below before launching them. Pretrained ImageNet encoder weights may need to download on first use.

### U-Net++ with MobileNetV2

This uses the previous student's learning rate, Dice weight, epoch count, and downsampling, with batch size 8 as recorded in his later commands. The model and experiment name are explicit so the checkpoint and log names agree.

```bash
python -u mainbf16.py \
  --model_name UnetPlusPlus \
  --encoder_name mobilenet_v2 \
  --batch_size 8 \
  --epochs 100 \
  --downsample_factor 4 \
  --learning_rate 1e-3 \
  --dice_scalar 0.5 \
  --exp_name mbv2_lr3_ld5 \
  > mbv2_lr3_ld5.txt 2>&1
```

### SegFormer with MiT-B3 and CLAHE

This follows the previous student's recorded CLAHE clip-limit 10 run with batch size 5.

```bash
python -u mainbf16.py \
  --model_name Segformer \
  --encoder_name mit_b3 \
  --seed 7 \
  --batch_size 5 \
  --epochs 100 \
  --downsample_factor 4 \
  --learning_rate 1e-4 \
  --dice_scalar 0.3 \
  --contrast clahe \
  --clahe_clip 10 \
  --exp_name segformer_clahe10_lr4_ld3_b3 \
  > segformer_clahe10_lr4_ld3_b3.txt 2>&1
```

To compare CLAHE 25, change `--clahe_clip` to `25` and use a different experiment and log name. These examples use the previous student's recorded workflow settings; they are not a complete reproduction of the paper's experiments.

| Flag | Meaning |
| --- | --- |
| `--model_name` | Architecture: exactly `UnetPlusPlus` (default) or `Segformer`; spelling and case matter. |
| `--encoder_name` | Backbone, such as `mobilenet_v2` or `mit_b3`; default `resnet18`. |
| `--batch_size` | Images per batch; reduce this if memory runs out. Default 4. |
| `--epochs` | Number of training epochs; default 100. |
| `--downsample_factor` | Divide image and mask height/width by this factor; 4 means one-quarter of each dimension. Default 1. |
| `--learning_rate` | Initial AdamW learning rate; default `1e-3`. |
| `--dice_scalar` | Weight of Dice loss added to BCE; default 1. |
| `--seed` | Seed for PyTorch and the train/test split; default 42. |
| `--contrast` | `none` (default), `he` for global histogram equalization, or `clahe`. HE/CLAHE operate on luminance in YCrCb. |
| `--clahe_clip` | CLAHE clip limit; default 5. Used with `--contrast clahe`; tile grid is fixed at 8×8. |
| `--exp_name` | Checkpoint filename stem; default `my_exp`. Use a new name for each run to avoid overwriting checkpoints. |

Other supported options:

- `--raw_data` and `--mask_data`: override the image and mask directories shown above.
- `--decay`: AdamW weight decay; default `1e-5`.
- `--train_ratio`: fraction used for training; default `0.8`, strictly between 0 and 1.
- `--num_workers`: DataLoader workers; default 4.
- `--device`: default `cuda` when available, otherwise `cpu`. Use CUDA on Metis.
- `--device_ids`: space-separated GPU IDs for DataParallel; optional.
- `--stratify`: split by CSV class and print per-tribe segmentation Dice/IoU plus class means after each epoch. This still trains one binary segmentation model.
- `--taxonomy_csv`: taxonomy file for `--stratify`; default `bovid_taxonomy.csv`.

## Segmentation inference: `infer.py`

Run inference after the corresponding checkpoint exists. For the MobileNetV2 example:

```bash
python infer.py \
  --checkpoint saved_models/mbv2_lr3_ld5.pt \
  --input_dir filt_res_data/raw_aligned \
  --output_dir mbv2_lr3_ld5_infer \
  --model_name UnetPlusPlus \
  --encoder_name mobilenet_v2 \
  --use_heatmap 0
```

For the SegFormer example:

```bash
python infer.py \
  --checkpoint saved_models/segformer_clahe10_lr4_ld3_b3.pt \
  --input_dir filt_res_data/raw_aligned \
  --output_dir segformer_clahe10_lr4_ld3_b3_infer \
  --model_name Segformer \
  --encoder_name mit_b3 \
  --use_heatmap 1
```

`--checkpoint`, `--input_dir`, and `--output_dir` are required. **SegFormer checkpoints require `--model_name Segformer`; `--encoder_name` must match training.** Inference defaults to U-Net++ with `resnet18`; it does not choose the architecture automatically from the checkpoint. The previous student's recorded workflow includes attempts with mismatched encoders, so check these settings before running inference.

- `--use_heatmap 0` (default) writes binary masks with hole filling; `--use_heatmap 1` writes probability heatmaps. Use integers, not `True`/`False`.
- `--threshold` sets the binary sigmoid threshold; default `0.5`. It does not affect heatmaps.
- `--device` defaults to CUDA when available, otherwise CPU.

The script reads contrast settings and downsampling from the checkpoint's saved arguments. It scans only the immediate input directory for JPG/JPEG, PNG, TIF, or TIFF images and writes `<image_stem>_mask.png` in the output directory. Masks remain at the downsampled, padded resolution; they are not resized back to the original photograph size.

## Plotting segmentation logs

### `plot.py`

`plot.py` has **no argparse options**. It reads `<experiment>.txt` logs in the current directory and plots train/test BCE loss, train/test Dice loss, and test Dice score. Before running it:

1. Set `USED_MODELS` in `plot.py` to your log filename stems (without `.txt`); it currently lists older ResNet34 experiments.
2. Set `NUM_EPOCHS` to the maximum number of epochs to read (currently 100).
3. Ensure `plots/` exists. Each `plots/<experiment>/` subdirectory must not already exist because the script uses `os.mkdir`.

```bash
mkdir -p plots
python plot.py
```

Output files are `plots/<experiment>/<metric>.png`.

### `data_analysis.py`

This is another **segmentation log plotting** utility, despite its general name. Pass one or more log stems using `--model_names` (or `-s`):

```bash
python data_analysis.py --model_names mbv2_lr3_ld5 segformer_clahe10_lr4_ld3_b3
```

It reads the corresponding `.txt` files and saves graphs under `model_plots/<experiment>_graphs/`. It looks for `test_bce_loss`, `test_dice_metric`, `test_dice_loss`, `mIoU`, and `Dice`.

## Tribe classification: `mainbf16_class.py`

The current script performs **image-level tribe classification**, using `ClassificationDataset` from `dataloader_class.py` and the CSV `class` column. It trains a pretrained `timm` backbone with cross-entropy loss and bfloat16 on CUDA/CPU. It reports accuracy, balanced accuracy, macro precision, macro recall, and macro F1. Its `--stratify` flag stratifies the classification split by tribe. For stratified per-tribe **segmentation**, use `mainbf16.py --stratify` instead.

An example using the script's default ResNet18 backbone and workflow settings similar to the previous student's segmentation runs:

```bash
python -u mainbf16_class.py \
  --model_name resnet18 \
  --raw_data filt_res_data/raw_aligned \
  --taxonomy_csv bovid_taxonomy.csv \
  --stratify \
  --seed 7 \
  --batch_size 4 \
  --epochs 100 \
  --downsample_factor 4 \
  --learning_rate 1e-3 \
  --contrast clahe \
  --clahe_clip 25 \
  > resnet18_classification.txt 2>&1
```

This is a proposed classification command, not a recorded classification run from the previous student. `--model_name` is a `timm` classifier name (default `resnet18`), rather than the segmentation architecture name. The script has no `--encoder_name`, `--mask_data`, or `--dice_scalar` CLI options. It also accepts `--decay`, `--train_ratio`, `--num_workers`, `--device`, and `--device_ids`; their defaults match the segmentation script. `--stratify` is off unless supplied.

The best epoch is selected by test macro F1. Training writes the following outputs:

- `saved_models/best_checkpoint.pt`
- `saved_models/confusion_matrix.png`
- `saved_models/confusion_matrix.csv`
- `saved_models/classification_report.csv`

Each run writes to these same filenames, so copy `saved_models/` elsewhere before starting another classification run.

Classification checkpoints cannot be loaded by the segmentation-only `infer.py`; neither plotting script handles the current classification metrics.

## Running jobs on the shared machine

Check GPU activity and available system memory before launching a job:

```bash
nvidia-smi
free -h
```

Metis has unified memory, so `nvidia-smi` may not report GPU memory usage; use `free -h` for available memory and coordinate GPU use with the lab.

Start a session that survives SSH disconnection:

```bash
tmux new -s bovid
conda activate bovid_extant_env
```

Run a training command inside that session. Detach with **Ctrl-b**, then **d**. Reconnect later with:

```bash
tmux attach -t bovid
```

The examples use Python's `-u` for unbuffered output and `> experiment.txt 2>&1` to capture standard output and errors together. Redirection overwrites an existing log, so use a new filename for each experiment. Follow a running log from another terminal:

```bash
tail -f segformer_clahe10_lr4_ld3_b3.txt
```

If the process is OOM-killed or reports an out-of-memory error, check available memory and other jobs, then lower `--batch_size` before retrying. The previous student reduced batch sizes during his runs; the examples are starting points, not guaranteed to fit under current shared load. Reducing `--num_workers` can also help with host-memory pressure.

## Repository layout

| File / directory | Purpose |
| --- | --- |
| `mainbf16.py` | Binary segmentation training and optional per-tribe evaluation. |
| `mainbf16_class.py` | Tribe classification training, reports, and confusion matrices. |
| `dataloader.py`, `dataloader_class.py` | Pair/label loading, splitting, downsampling, and batch padding. |
| `preprocess.py`, `preprocess_class.py` | Contrast adjustment and model-specific preprocessing. |
| `infer.py` | Segmentation masks or probability heatmaps from a checkpoint. |
| `plot.py`, `data_analysis.py` | Segmentation log plots. |
| `main.py` | Additional segmentation trainer; also supplies `pad_to_multiple` imported by `infer.py`. |
| `flops.py` | Segmentation model FLOPs and parameter counting; imports `fvcore`. |
| `data_filter.py` | Deletes files without matching names in paired folders; destructive, not needed for normal training. |
| `environment.yml` | Conda environment export. |
| `bovid_taxonomy.csv` | Filename-to-tribe mapping. |
| `filt_res_data/` | Shared dataset symlink on the lab checkout. |
| `saved_models/` | Checkpoints and classification reports created by training. |
