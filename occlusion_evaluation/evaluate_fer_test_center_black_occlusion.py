"""
Evaluate FER test accuracy after centered black-square occlusion.

The script:
1. Loads the FER2013 VGG19 private checkpoint.
2. Reads FER test images from a folder dataset.
3. Adds a black square in the center of each image in memory.
4. Evaluates the original and occluded images.
5. Reports original and occluded accuracy per class, macro accuracy, and overall accuracy.

It does not modify the original dataset.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image


THIS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = THIS_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import transforms.transforms as transforms
from model_architectures.vgg import VGG


# FER2013 label order used by preprocess_fer2013.py and PrivateTest_model.t7:
# 0=angry, 1=disgust, 2=fear, 3=happy, 4=sad, 5=surprise, 6=neutral
FER_CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}


def default_model_path():
    return str(PROJECT_DIR / 'checkpoints' / 'PrivateTest_model.t7')


def resolve_path(path):
    raw = Path(path)
    candidates = [raw, PROJECT_DIR / raw, THIS_DIR / raw]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return raw


class FERFolderDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = resolve_path(root_dir)
        self.transform = transform
        self.images = []
        self.labels = []

        for class_index, class_name in enumerate(FER_CLASS_NAMES):
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue
            for image_path in sorted(class_dir.iterdir()):
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.images.append(str(image_path))
                    self.labels.append(class_index)

        if not self.images:
            raise ValueError(f'No FER test images found in {self.root_dir}')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('L').convert('RGB')
        tensor = self.transform(image) if self.transform else image
        return tensor, self.labels[index], self.images[index]


def build_model(checkpoint_path, device):
    checkpoint_path = resolve_path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f'Model checkpoint not found: {checkpoint_path}')

    model = VGG('VGG19')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint['net'] if isinstance(checkpoint, dict) and 'net' in checkpoint else checkpoint
    if isinstance(state, nn.Module):
        state = state.state_dict()
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def make_transform():
    return transforms.Compose([
        transforms.TenCrop(44),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
    ])


def occlude_center_black_square(input_tensor, patch_size):
    occluded = input_tensor.clone()
    _, _, height, width = input_tensor.shape
    patch_size = min(patch_size, height, width)

    half = patch_size // 2
    center_y = height // 2
    center_x = width // 2
    y1 = max(0, center_y - half)
    y2 = min(height, y1 + patch_size)
    x1 = max(0, center_x - half)
    x2 = min(width, x1 + patch_size)
    y1 = max(0, y2 - patch_size)
    x1 = max(0, x2 - patch_size)

    occluded[:, :, y1:y2, x1:x2] = 0.0
    return occluded, (x1, y1, x2, y2)


def predict_tencrop(model, crops_tensor):
    outputs = model(crops_tensor)
    outputs_avg = outputs.mean(dim=0)
    return int(torch.argmax(outputs_avg).item())


def update_stats(stats, true_label, pred_label):
    stats['total'][true_label] += 1
    stats['confusion'][true_label, pred_label] += 1
    if true_label == pred_label:
        stats['correct'][true_label] += 1


def compute_summary(stats):
    rows = []
    per_class_accuracy = []
    total_correct = 0
    total_samples = int(stats['total'].sum())

    for index, class_name in enumerate(FER_CLASS_NAMES):
        total = int(stats['total'][index])
        correct = int(stats['correct'][index])
        accuracy = 100.0 * correct / total if total else 0.0
        rows.append({
            'class': class_name,
            'correct': correct,
            'total': total,
            'accuracy': accuracy,
        })
        per_class_accuracy.append(accuracy)
        total_correct += correct

    return {
        'rows': rows,
        'macro_accuracy': float(np.mean(per_class_accuracy)),
        'overall_accuracy': 100.0 * total_correct / total_samples if total_samples else 0.0,
        'correct': total_correct,
        'total': total_samples,
        'confusion': stats['confusion'],
    }


def write_summary_csv(output_path, original_summary, occluded_summary):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow([
            'class',
            'original_correct',
            'original_total',
            'original_accuracy',
            'occluded_correct',
            'occluded_total',
            'occluded_accuracy',
            'accuracy_drop',
        ])

        for original_row, occluded_row in zip(original_summary['rows'], occluded_summary['rows']):
            writer.writerow([
                original_row['class'],
                original_row['correct'],
                original_row['total'],
                '%.4f' % original_row['accuracy'],
                occluded_row['correct'],
                occluded_row['total'],
                '%.4f' % occluded_row['accuracy'],
                '%.4f' % (original_row['accuracy'] - occluded_row['accuracy']),
            ])

        writer.writerow([])
        writer.writerow(['metric', 'original', 'occluded', 'drop'])
        writer.writerow([
            'macro_accuracy',
            '%.4f' % original_summary['macro_accuracy'],
            '%.4f' % occluded_summary['macro_accuracy'],
            '%.4f' % (original_summary['macro_accuracy'] - occluded_summary['macro_accuracy']),
        ])
        writer.writerow([
            'overall_accuracy',
            '%.4f' % original_summary['overall_accuracy'],
            '%.4f' % occluded_summary['overall_accuracy'],
            '%.4f' % (original_summary['overall_accuracy'] - occluded_summary['overall_accuracy']),
        ])


def save_confusion_matrix(confusion, output_prefix, title):
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_prefix.with_suffix('.csv')
    png_path = output_prefix.with_suffix('.png')

    with open(csv_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['true/pred'] + FER_CLASS_NAMES)
        for index, class_name in enumerate(FER_CLASS_NAMES):
            writer.writerow([class_name] + confusion[index].astype(int).tolist())

    row_sums = confusion.sum(axis=1, keepdims=True).astype(np.float64)
    row_sums[row_sums == 0] = 1.0
    normalized = confusion / row_sums

    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(normalized, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_xticks(np.arange(len(FER_CLASS_NAMES)))
    ax.set_yticks(np.arange(len(FER_CLASS_NAMES)))
    ax.set_xticklabels(FER_CLASS_NAMES, rotation=45, ha='right')
    ax.set_yticklabels(FER_CLASS_NAMES)

    for row in range(len(FER_CLASS_NAMES)):
        for col in range(len(FER_CLASS_NAMES)):
            value = normalized[row, col]
            ax.text(
                col,
                row,
                '%.2f' % value,
                ha='center',
                va='center',
                color='white' if value > 0.55 else 'black',
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return csv_path, png_path


def print_summary(title, summary):
    print(f'\n{title}')
    print('  Overall accuracy: %.2f%% (%d/%d)' % (
        summary['overall_accuracy'], summary['correct'], summary['total']))
    print('  Macro accuracy:   %.2f%%' % summary['macro_accuracy'])
    print('  %-10s %10s %14s' % ('class', 'accuracy', 'correct/total'))
    for row in summary['rows']:
        print('  %-10s %9.2f%% %7d/%d' % (
            row['class'], row['accuracy'], row['correct'], row['total']))


def save_examples(output_dir, image_tensor, occluded_tensor, image_path, bbox, true_label, pred_original, pred_occluded):
    output_dir.mkdir(parents=True, exist_ok=True)

    image_np = image_tensor[0].permute(1, 2, 0).detach().cpu().numpy()
    occluded_np = occluded_tensor[0].permute(1, 2, 0).detach().cpu().numpy()
    x1, y1, x2, y2 = bbox

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image_np)
    axes[0].set_title('Original')
    axes[0].axis('off')

    axes[1].imshow(image_np)
    axes[1].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='black', facecolor='black', alpha=0.85))
    axes[1].set_title('Centered black square')
    axes[1].axis('off')

    axes[2].imshow(occluded_np)
    axes[2].set_title('Occluded')
    axes[2].axis('off')

    fig.suptitle(
        'true=%s | original=%s | occluded=%s' % (
            FER_CLASS_NAMES[true_label],
            FER_CLASS_NAMES[pred_original],
            FER_CLASS_NAMES[pred_occluded],
        )
    )
    fig.tight_layout()
    stem = Path(image_path).stem
    fig.savefig(output_dir / f'{stem}_occlusion.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate FER test accuracy after centered black-square occlusion.')
    parser.add_argument('--model', default=default_model_path())
    parser.add_argument('--fer-test-dir', default='FerDataset/test')
    parser.add_argument('--output-dir', default=str(THIS_DIR / 'outputs'))
    parser.add_argument('--batch-limit', default=None, type=int,
                        help='Optional limit for quick tests. Processes this many images total.')
    parser.add_argument('--patch-size', default=4, type=int,
                        help='Square occlusion size on the transformed 44x44 image.')
    parser.add_argument('--save-examples', default=12, type=int,
                        help='Number of example visualizations to save.')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                        choices=['cuda', 'cpu'])
    return parser.parse_args()


def main():
    opt = parse_args()
    if opt.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA not available. Use --device cpu')

    device = torch.device(opt.device)
    print(f'Device: {device}')
    print(f'Model: {resolve_path(opt.model)}')
    print(f'FER test dataset: {resolve_path(opt.fer_test_dir)}')

    model = build_model(opt.model, device)
    dataset = FERFolderDataset(opt.fer_test_dir, transform=make_transform())

    original_stats = {
        'correct': np.zeros(len(FER_CLASS_NAMES), dtype=np.int64),
        'total': np.zeros(len(FER_CLASS_NAMES), dtype=np.int64),
        'confusion': np.zeros((len(FER_CLASS_NAMES), len(FER_CLASS_NAMES)), dtype=np.int64),
    }
    occluded_stats = {
        'correct': np.zeros(len(FER_CLASS_NAMES), dtype=np.int64),
        'total': np.zeros(len(FER_CLASS_NAMES), dtype=np.int64),
        'confusion': np.zeros((len(FER_CLASS_NAMES), len(FER_CLASS_NAMES)), dtype=np.int64),
    }

    output_dir = Path(opt.output_dir)
    examples_dir = output_dir / 'examples'
    total_to_process = len(dataset) if opt.batch_limit is None else min(opt.batch_limit, len(dataset))

    for index in range(total_to_process):
        image_tensor, true_label, image_path = dataset[index]
        true_label = int(true_label)
        crops_tensor = image_tensor.to(device)

        with torch.no_grad():
            pred_original = predict_tencrop(model, crops_tensor)

        occluded_tensor, bbox = occlude_center_black_square(crops_tensor, opt.patch_size)

        with torch.no_grad():
            pred_occluded = predict_tencrop(model, occluded_tensor)

        update_stats(original_stats, true_label, pred_original)
        update_stats(occluded_stats, true_label, pred_occluded)

        if index < opt.save_examples:
            save_examples(
                examples_dir,
                crops_tensor[0:1],
                occluded_tensor[0:1],
                image_path,
                bbox,
                true_label,
                pred_original,
                pred_occluded,
            )

        if (index + 1) % 100 == 0 or index + 1 == total_to_process:
            print(f'Processed {index + 1}/{total_to_process}')

    original_summary = compute_summary(original_stats)
    occluded_summary = compute_summary(occluded_stats)

    print_summary('Original FER test accuracy', original_summary)
    print_summary('Centered black-square occluded FER test accuracy', occluded_summary)
    print('\nAccuracy drop:')
    print('  Overall: %.2f%%' % (
        original_summary['overall_accuracy'] - occluded_summary['overall_accuracy']))
    print('  Macro:   %.2f%%' % (
        original_summary['macro_accuracy'] - occluded_summary['macro_accuracy']))

    suffix = '' if opt.batch_limit is None else f'_first_{total_to_process}'
    summary_path = output_dir / f'fer_test_center_black_occlusion_accuracy{suffix}.csv'
    write_summary_csv(summary_path, original_summary, occluded_summary)
    original_cm_csv, original_cm_png = save_confusion_matrix(
        original_summary['confusion'],
        output_dir / f'original_confusion_matrix{suffix}',
        'Original FER Test Confusion Matrix',
    )
    occluded_cm_csv, occluded_cm_png = save_confusion_matrix(
        occluded_summary['confusion'],
        output_dir / f'occluded_confusion_matrix{suffix}',
        'Centered Black-Square Occluded Confusion Matrix',
    )
    print(f'\nSaved metrics: {summary_path}')
    print(f'Saved original confusion matrix: {original_cm_csv}, {original_cm_png}')
    print(f'Saved occluded confusion matrix: {occluded_cm_csv}, {occluded_cm_png}')
    if opt.save_examples:
        print(f'Saved examples: {examples_dir}')


if __name__ == '__main__':
    main()
