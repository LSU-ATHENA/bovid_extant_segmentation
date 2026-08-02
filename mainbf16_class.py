import argparse
import csv
from contextlib import nullcontext
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import psutil
import timm
import torch as t
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
	accuracy_score,
	balanced_accuracy_score,
	confusion_matrix,
	precision_recall_fscore_support,
)
from tqdm import tqdm

from dataloader_class import ClassificationDataset, make_train_test_loaders
import preprocess_class


def parse_args():
	parser = argparse.ArgumentParser(description="Bovid taxonomy classification training")
	parser.add_argument("--model_name", type=str, default="resnet18", help="Any timm classification backbone")
	parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
	parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
	parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
	parser.add_argument("--decay", type=float, default=1e-5, help="Weight decay for optimizer")
	parser.add_argument("--train_ratio", type=float, default=0.8, help="Train/test split ratio")
	parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
	parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers")
	parser.add_argument("--raw_data", type=str, default="filt_res_data/raw_aligned", help="Path to raw data root")
	parser.add_argument("--taxonomy_csv", type=str, default="bovid_taxonomy.csv", help="Path to taxonomy CSV")
	parser.add_argument("--device", type=str, default="cuda" if t.cuda.is_available() else "cpu", help="Device to use")
	parser.add_argument("--downsample_factor", type=int, default=1, help="Factor to downsample images for faster training")
	parser.add_argument("--exp_name", type=str, default="my_exp")
	parser.add_argument("--contrast", type=str, default="none")
	parser.add_argument("--clahe_clip", type=int, default=5)
	parser.add_argument("--device_ids", nargs="+", type=int, help="GPUs to use")
	parser.add_argument("--stratify", action="store_true", help="Whether to stratify the train/test split")
	return parser.parse_args()


def autocast_context(device):
	if device.type in {"cuda", "cpu"}:
		return t.autocast(dtype=t.bfloat16, device_type=device.type)
	return nullcontext()


def train_epoch(model, train_loader, optimizer, ce_loss, device):
	model.train()
	total_loss = 0.0

	pbar = tqdm(train_loader, desc="Train", leave=False)
	for batch in pbar:
		images, labels = batch["x"], batch["class"]
		images, labels = images.to(device), labels.to(device).long()

		optimizer.zero_grad()
		with autocast_context(device):
			logits = model(images)
			loss = ce_loss(logits, labels)

		loss.backward()
		t.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
		optimizer.step()

		total_loss += loss.item()
		pbar.set_postfix(loss=f"{loss.item():.4f}")

	return total_loss / len(train_loader)


def eval_epoch(model, test_loader, ce_loss, device):
	model.eval()

	total_loss = 0.0
	all_true = []
	all_pred = []

	with t.no_grad():
		pbar = tqdm(test_loader, desc="Eval", leave=False)
		for batch in pbar:
			images, labels = batch["x"], batch["class"]
			images, labels = images.to(device), labels.to(device).long()

			with autocast_context(device):
				logits = model(images)
				loss = ce_loss(logits, labels)

			preds = t.argmax(logits, dim=1)
			all_true.extend(labels.cpu().tolist())
			all_pred.extend(preds.cpu().tolist())

			total_loss += loss.item()
			pbar.set_postfix(loss=f"{loss.item():.4f}")

	test_loss = total_loss / len(test_loader)
	test_accuracy = accuracy_score(all_true, all_pred)
	test_balanced_accuracy = balanced_accuracy_score(all_true, all_pred)
	test_macro_precision, test_macro_recall, test_macro_f1, _ = precision_recall_fscore_support(
		all_true,
		all_pred,
		average="macro",
		zero_division=0,
	)

	metrics = {
		"test_loss": test_loss,
		"test_accuracy": float(test_accuracy),
		"test_balanced_accuracy": float(test_balanced_accuracy),
		"test_macro_precision": float(test_macro_precision),
		"test_macro_recall": float(test_macro_recall),
		"test_macro_f1": float(test_macro_f1),
	}
	return metrics, all_true, all_pred


def save_confusion_matrix_csv(cm, class_names, output_path):
	with output_path.open("w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow(["class"] + class_names)
		for i, class_name in enumerate(class_names):
			writer.writerow([class_name] + cm[i].tolist())


def save_confusion_matrix_png(cm, class_names, output_path):
	fig, ax = plt.subplots(figsize=(8, 6))
	im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
	ax.figure.colorbar(im, ax=ax)
	ax.set(
		xticks=np.arange(len(class_names)),
		yticks=np.arange(len(class_names)),
		xticklabels=class_names,
		yticklabels=class_names,
		ylabel="True label",
		xlabel="Predicted label",
		title="Confusion Matrix",
	)
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

	thresh = cm.max() / 2.0 if cm.size > 0 else 0.0
	for i in range(cm.shape[0]):
		for j in range(cm.shape[1]):
			ax.text(
				j,
				i,
				format(cm[i, j], "d"),
				ha="center",
				va="center",
				color="white" if cm[i, j] > thresh else "black",
			)

	fig.tight_layout()
	fig.savefig(output_path, dpi=200)
	plt.close(fig)


def save_classification_report_csv(all_true, all_pred, class_names, output_path):
	labels = list(range(len(class_names)))
	precision, recall, f1, support = precision_recall_fscore_support(
		all_true,
		all_pred,
		labels=labels,
		average=None,
		zero_division=0,
	)

	with output_path.open("w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow(["class", "precision", "recall", "f1", "support"])
		for idx, class_name in enumerate(class_names):
			writer.writerow([
				class_name,
				f"{precision[idx]:.6f}",
				f"{recall[idx]:.6f}",
				f"{f1[idx]:.6f}",
				int(support[idx]),
			])


if __name__ == "__main__":
	args = parse_args()

    # This hack for compatibility with preprocess
	args.encoder_name = args.model_name

	if args.device == "cuda" and not t.cuda.is_available():
		if hasattr(t.backends, "mps") and t.backends.mps.is_available():
			device = t.device("mps")
		else:
			device = t.device("cpu")
	elif args.device == "mps" and (not hasattr(t.backends, "mps") or not t.backends.mps.is_available()):
		device = t.device("cpu")
	else:
		device = t.device(args.device)

	t.manual_seed(args.seed)
	np.random.seed(args.seed)

	raw_root = Path(args.raw_data)
	taxonomy_csv_path = Path(args.taxonomy_csv)

	prepro_obj = preprocess_class.PreProObj(args=args)
	prepro_obj.setup_stack(stack=[prepro_obj.apply_contrast])

	dataset = ClassificationDataset(
		raw_root=raw_root,
		taxonomy_csv_path=taxonomy_csv_path,
		prepro_obj=prepro_obj,
		args=args,
	)
	num_classes = len(dataset.idx_to_class)
	class_names = dataset.idx_to_class

	print(f"Total classification samples found: {len(dataset)}")
	print(f"Num classes: {num_classes}")

	train_loader, test_loader = make_train_test_loaders(
		dataset,
		train_ratio=args.train_ratio,
		seed=args.seed,
		batch_size=args.batch_size,
		num_workers=args.num_workers,
		stratify=args.stratify,
	)

	model = timm.create_model(
		args.model_name,
		pretrained=True,
		num_classes=num_classes,
	)
	model = model.to(device)

	if device.type == "cuda" and args.device_ids and len(args.device_ids) > 1:
		model = nn.DataParallel(model, device_ids=args.device_ids)

	ce_loss = nn.CrossEntropyLoss()
	optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.decay)
	scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

	save_dir = Path("saved_models")
	save_dir.mkdir(parents=True, exist_ok=True)
	best_checkpoint_path = save_dir / "best_checkpoint.pt"
	confusion_png_path = save_dir / "confusion_matrix.png"
	confusion_csv_path = save_dir / "confusion_matrix.csv"
	report_csv_path = save_dir / "classification_report.csv"

	best_macro_f1 = float("-inf")
	best_y_true = None
	best_y_pred = None

	print(f"Using device: {device}")
	for epoch in range(1, args.epochs + 1):
		train_loss = train_epoch(model, train_loader, optimizer, ce_loss, device)
		test_metrics, y_true, y_pred = eval_epoch(model, test_loader, ce_loss, device)

		print(
			f"Epoch [{epoch:03d}/{args.epochs:03d}] "
			f"train_loss={train_loss:.4f} "
			f"test_loss={test_metrics['test_loss']:.4f} "
			f"test_accuracy={test_metrics['test_accuracy']:.4f} "
			f"test_balanced_accuracy={test_metrics['test_balanced_accuracy']:.4f} "
			f"test_macro_precision={test_metrics['test_macro_precision']:.4f} "
			f"test_macro_recall={test_metrics['test_macro_recall']:.4f} "
			f"test_macro_f1={test_metrics['test_macro_f1']:.4f} "
			f"free_memory_gb={psutil.virtual_memory().available / (1024 ** 3):.2f}"
		)

		scheduler.step()

		if test_metrics["test_macro_f1"] > best_macro_f1:
			best_macro_f1 = test_metrics["test_macro_f1"]
			best_y_true = y_true
			best_y_pred = y_pred

			t.save(
				{
					"epoch": epoch,
					"model_state_dict": model.state_dict(),
					"optimizer_state_dict": optimizer.state_dict(),
					"train_loss": train_loss,
					"test_loss": test_metrics["test_loss"],
					"test_accuracy": test_metrics["test_accuracy"],
					"test_balanced_accuracy": test_metrics["test_balanced_accuracy"],
					"test_macro_precision": test_metrics["test_macro_precision"],
					"test_macro_recall": test_metrics["test_macro_recall"],
					"test_macro_f1": test_metrics["test_macro_f1"],
					"args": vars(args),
					"class_names": class_names,
				},
				best_checkpoint_path,
			)
			print(f"Saved new best checkpoint to {best_checkpoint_path} (macro_f1={best_macro_f1:.4f})")

	if best_y_true is None or best_y_pred is None:
		raise RuntimeError("No evaluation results were collected; cannot save confusion matrix/report.")

	labels = list(range(num_classes))
	cm = confusion_matrix(best_y_true, best_y_pred, labels=labels)

	save_confusion_matrix_csv(cm, class_names, confusion_csv_path)
	save_confusion_matrix_png(cm, class_names, confusion_png_path)
	save_classification_report_csv(best_y_true, best_y_pred, class_names, report_csv_path)

	print(f"Saved confusion matrix image to {confusion_png_path}")
	print(f"Saved confusion matrix csv to {confusion_csv_path}")
	print(f"Saved classification report csv to {report_csv_path}")
	print(f"Training complete. Best test macro F1: {best_macro_f1:.4f}")
