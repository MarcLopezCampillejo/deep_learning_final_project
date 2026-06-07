"""
Evaluate the fine-tuned Best_model.t7 on the FER test dataset.

Default model:
  fine_tuning/outputs/fine_tuned/Best_model.t7

Default dataset:
  datasets/FerDataset/test

Reports loss, overall accuracy, macro accuracy, and per-class accuracy.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

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


DEFAULT_CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}


def resolve_path(path):
    raw = Path(path)
    candidates = [raw, THIS_DIR / raw, PROJECT_DIR / raw]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return raw


def resolve_output_path(path):
    raw = Path(path)
    if raw.is_absolute():
        return raw
    return THIS_DIR / raw


class FERFolderDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, class_names, transform=None):
        self.root_dir = resolve_path(root_dir)
        self.class_names = class_names
        self.transform = transform
        self.images = []
        self.labels = []

        for class_index, class_name in enumerate(class_names):
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue
            for image_path in sorted(class_dir.iterdir()):
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.images.append(str(image_path))
                    self.labels.append(class_index)

        if not self.images:
            raise ValueError(f'No FER images found in {self.root_dir}')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('L').convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[index]


def make_transform():
    return transforms.Compose([
        transforms.Resize(48),
        transforms.CenterCrop(44),
        transforms.ToTensor(),
    ])


def load_model_and_class_names(model_path, device):
    model_path = resolve_path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f'Model not found: {model_path}')

    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint.get('class_names', DEFAULT_CLASS_NAMES) if isinstance(checkpoint, dict) else DEFAULT_CLASS_NAMES

    model = VGG('VGG19')
    state = checkpoint['net'] if isinstance(checkpoint, dict) and 'net' in checkpoint else checkpoint
    if isinstance(state, nn.Module):
        state = state.state_dict()
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    return model, class_names, model_path


def evaluate(model, loader, criterion, device, class_names):
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    per_class_correct = np.zeros(len(class_names), dtype=np.int64)
    per_class_total = np.zeros(len(class_names), dtype=np.int64)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs = inputs.to(device)
            targets = torch.as_tensor(targets, dtype=torch.long, device=device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_samples += targets.size(0)
            total_correct += predicted.eq(targets).sum().item()

            for true_label, pred_label in zip(targets.cpu().numpy(), predicted.cpu().numpy()):
                per_class_total[true_label] += 1
                if true_label == pred_label:
                    per_class_correct[true_label] += 1

            if (batch_idx + 1) % 10 == 0 or batch_idx + 1 == len(loader):
                print(f'Processed batch {batch_idx + 1}/{len(loader)}')

    per_class_accuracy = np.divide(
        per_class_correct,
        per_class_total,
        out=np.zeros(len(class_names), dtype=np.float64),
        where=per_class_total != 0,
    ) * 100.0

    return {
        'loss': total_loss / max(1, len(loader)),
        'overall_accuracy': 100.0 * total_correct / max(1, total_samples),
        'macro_accuracy': float(np.mean(per_class_accuracy)),
        'total_correct': total_correct,
        'total_samples': total_samples,
        'per_class_correct': per_class_correct,
        'per_class_total': per_class_total,
        'per_class_accuracy': per_class_accuracy,
    }


def print_results(results, class_names):
    print('\n' + '=' * 72)
    print('FINE-TUNED BEST MODEL ON FER TEST')
    print('=' * 72)
    print(f'Loss: {results["loss"]:.4f}')
    print(f'Overall accuracy: {results["overall_accuracy"]:.2f}% ({results["total_correct"]}/{results["total_samples"]})')
    print(f'Macro accuracy: {results["macro_accuracy"]:.2f}%')
    print()
    print('%-12s %12s %18s' % ('Class', 'Accuracy', 'Correct/Total'))
    print('-' * 46)
    for index, class_name in enumerate(class_names):
        print('%-12s %10.2f%% %8d/%d' % (
            class_name,
            results['per_class_accuracy'][index],
            int(results['per_class_correct'][index]),
            int(results['per_class_total'][index]),
        ))


def save_results(results, class_names, output_path):
    output_path = resolve_output_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['metric', 'value'])
        writer.writerow(['loss', '%.6f' % results['loss']])
        writer.writerow(['overall_accuracy', '%.6f' % results['overall_accuracy']])
        writer.writerow(['macro_accuracy', '%.6f' % results['macro_accuracy']])
        writer.writerow(['total_correct', results['total_correct']])
        writer.writerow(['total_samples', results['total_samples']])
        writer.writerow([])
        writer.writerow(['class', 'accuracy', 'correct', 'total'])
        for index, class_name in enumerate(class_names):
            writer.writerow([
                class_name,
                '%.6f' % results['per_class_accuracy'][index],
                int(results['per_class_correct'][index]),
                int(results['per_class_total'][index]),
            ])

    print(f'\nSaved results: {output_path}')


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate fine-tuned Best_model.t7 on FER test.')
    parser.add_argument('--model', default='outputs/fine_tuned/Best_model.t7')
    parser.add_argument('--fer-test-dir', default='datasets/FerDataset/test')
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--output', default='outputs/fine_tuned/fer_test_best_model_metrics.csv')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                        choices=['cuda', 'cpu'])
    return parser.parse_args()


def main():
    opt = parse_args()
    if opt.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA not available. Use --device cpu')

    device = torch.device(opt.device)
    print(f'Device: {device}')
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')

    model, class_names, model_path = load_model_and_class_names(opt.model, device)
    print(f'Model: {model_path}')
    print('Class order: %s' % ', '.join(class_names))

    dataset = FERFolderDataset(opt.fer_test_dir, class_names, transform=make_transform())
    loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=0)
    print(f'FER test dataset: {resolve_path(opt.fer_test_dir)}')
    print(f'Test samples: {len(dataset)}')

    criterion = nn.CrossEntropyLoss()
    results = evaluate(model, loader, criterion, device, class_names)
    print_results(results, class_names)
    save_results(results, class_names, opt.output)


if __name__ == '__main__':
    main()
