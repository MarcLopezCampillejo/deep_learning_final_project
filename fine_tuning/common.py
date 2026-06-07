import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import WeightedRandomSampler

try:
    from torch.amp import GradScaler, autocast
    _USE_NEW_AMP = True
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
    _USE_NEW_AMP = False


THIS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = THIS_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import transforms.transforms as transforms
import utils
from model_architectures.vgg import VGG


FER_CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
NUM_CLASSES = len(FER_CLASS_NAMES)

RAFCE_CLASS_NAMES = [
    'Happily Surprised', 'Happily Disgusted', 'Sadly Fearful', 'Sadly Angry',
    'Sadly Surprised', 'Sadly Disgusted', 'Fearfully Angry', 'Fearfully Surprised',
    'Angrily Surprised', 'Angrily Disgusted', 'Disgustedly Surprised',
    'Appalled', 'Hatred', 'Awed',
]

RAFCE_TO_FER = {
    0: 3,   # Happily Surprised -> happy
    1: 3,   # Happily Disgusted -> happy
    2: 5,   # Sadly Fearful -> sad
    3: 5,   # Sadly Angry -> sad
    4: 5,   # Sadly Surprised -> sad
    5: 5,   # Sadly Disgusted -> sad
    6: 2,   # Fearfully Angry -> fear
    7: 2,   # Fearfully Surprised -> fear
    8: 0,   # Angrily Surprised -> angry
    9: 0,   # Angrily Disgusted -> angry
    10: 1,  # Disgustedly Surprised -> disgust
    11: 1,  # Appalled -> disgust
    12: 0,  # Hatred -> angry
    13: 6,  # Awed -> surprise
}

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}
BLOCK5_START = 40


def resolve_path(path):
    raw = Path(path)
    candidates = [
        THIS_DIR / raw,
        raw,
        PROJECT_DIR / raw,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return THIS_DIR / raw


def default_model_path():
    local = THIS_DIR / 'PrivateTest_model.t7'
    if local.exists():
        return str(local)
    return str(PROJECT_DIR / 'checkpoints' / 'PrivateTest_model.t7')


def make_test_transform():
    return transforms.Compose([
        transforms.Resize(48),
        transforms.CenterCrop(44),
        transforms.ToTensor(),
    ])


def make_train_transform():
    return transforms.Compose([
        transforms.Resize(48),
        transforms.RandomCrop(44),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])


class RAFCEMappedDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, label_file, transform=None):
        self.img_dir = resolve_path(img_dir)
        self.label_file = resolve_path(label_file)
        self.transform = transform
        self.images = []
        self.labels = []
        self.compound_labels = []

        if not self.img_dir.exists():
            raise FileNotFoundError(f'RAF-CE image directory not found: {self.img_dir}')
        if not self.label_file.exists():
            raise FileNotFoundError(f'RAF-CE label file not found: {self.label_file}')

        label_map = {}
        with open(self.label_file, 'r', encoding='utf-8') as handle:
            for line in handle:
                parts = line.strip().split()
                if len(parts) >= 2:
                    label_map[parts[0]] = int(parts[1])

        for filename, compound_label in sorted(label_map.items()):
            img_path = self.img_dir / filename
            if img_path.exists() and img_path.suffix.lower() in IMAGE_EXTENSIONS:
                self.images.append(str(img_path))
                self.labels.append(RAFCE_TO_FER[compound_label])
                self.compound_labels.append(compound_label)

        if not self.images:
            raise ValueError(f'No RAF-CE images found in {self.img_dir}')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('L').convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[index]


class FERDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = resolve_path(root_dir)
        self.transform = transform
        self.images = []
        self.labels = []

        for class_index, class_name in enumerate(FER_CLASS_NAMES):
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue
            for path in sorted(class_dir.iterdir()):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.images.append(str(path))
                    self.labels.append(class_index)

        if not self.images:
            raise ValueError(f'No FER images found in {self.root_dir}')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('L').convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[index]


def build_vgg19(checkpoint_path, device):
    checkpoint_path = resolve_path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f'Model checkpoint not found: {checkpoint_path}')

    net = VGG('VGG19')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint['net'] if isinstance(checkpoint, dict) and 'net' in checkpoint else checkpoint
    if isinstance(state, nn.Module):
        state = state.state_dict()
    net.load_state_dict(state, strict=True)
    return net.to(device)


def set_phase(net, phase):
    if phase == 1:
        for param in net.features.parameters():
            param.requires_grad = False
        for param in net.classifier.parameters():
            param.requires_grad = True
        print('Phase 1: features frozen, head only')
    elif phase in (2, 3):
        for index, module in enumerate(net.features):
            for param in module.parameters():
                param.requires_grad = index >= BLOCK5_START
        for param in net.classifier.parameters():
            param.requires_grad = True
        print(f'Phase {phase}: block5 + head trainable, blocks 1-4 frozen')


def make_optimizer(net, lr):
    return optim.Adam([p for p in net.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)


def make_scaler(use_amp):
    if _USE_NEW_AMP:
        return GradScaler('cuda', enabled=use_amp)
    return GradScaler(enabled=use_amp)


def autocast_ctx(device, use_amp):
    if _USE_NEW_AMP:
        return autocast(device_type=device.type, enabled=use_amp)
    return autocast(enabled=use_amp)


def build_weighted_sampler(labels):
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float32)
    counts[counts == 0] = 1.0
    sample_weights = [1.0 / counts[label] for label in labels]
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )


def build_class_weights(labels, device):
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (NUM_CLASSES * counts)
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_one_epoch_mixed(net, loader, criterion, optimizer, scaler, device, use_amp):
    net.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device, non_blocking=True)
        targets = torch.as_tensor(targets, dtype=torch.long, device=device)

        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx(device, use_amp):
            outputs = net(inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        utils.clip_gradient(optimizer, 0.1)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().item()

        utils.progress_bar(batch_idx, len(loader),
            'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                total_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    return {'loss': total_loss / len(loader), 'accuracy': 100. * correct / total}


def validate_mixed(net, loader, criterion, device, use_amp):
    net.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    all_targets = []
    all_predicted = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs = inputs.to(device, non_blocking=True)
            targets = torch.as_tensor(targets, dtype=torch.long, device=device)
            with autocast_ctx(device, use_amp):
                outputs = net(inputs)
                loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum().item()

            for true_label, pred_label in zip(targets.cpu(), predicted.cpu()):
                true_label = int(true_label.item())
                pred_label = int(pred_label.item())
                class_total[true_label] += 1
                if true_label == pred_label:
                    class_correct[true_label] += 1
                all_targets.append(true_label)
                all_predicted.append(pred_label)

            utils.progress_bar(batch_idx, len(loader),
                'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                    total_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    per_class = {}
    for index, name in enumerate(FER_CLASS_NAMES):
        total_for_class = class_total[index]
        correct_for_class = class_correct[index]
        per_class[name] = {
            'accuracy': 100. * correct_for_class / total_for_class if total_for_class > 0 else 0.0,
            'correct': correct_for_class,
            'total': total_for_class,
        }

    macro = sum(row['accuracy'] for row in per_class.values()) / NUM_CLASSES
    return {
        'loss': total_loss / len(loader),
        'accuracy': 100. * correct / total,
        'macro_accuracy': macro,
        'correct': correct,
        'total': total,
        'per_class': per_class,
        'all_targets': all_targets,
        'all_predicted': all_predicted,
    }


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    class_total = np.zeros(NUM_CLASSES, dtype=np.int64)
    class_correct = np.zeros(NUM_CLASSES, dtype=np.int64)
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = torch.as_tensor(targets, dtype=torch.long, device=device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            for true_label, pred_label in zip(targets.cpu().numpy(), predicted.cpu().numpy()):
                class_total[true_label] += 1
                confusion[true_label, pred_label] += 1
                if true_label == pred_label:
                    class_correct[true_label] += 1

    per_class = {}
    for idx, name in enumerate(FER_CLASS_NAMES):
        acc = 100.0 * class_correct[idx] / class_total[idx] if class_total[idx] else 0.0
        per_class[name] = {
            'accuracy': acc,
            'correct': int(class_correct[idx]),
            'total': int(class_total[idx]),
        }

    return {
        'loss': total_loss / max(1, len(loader)),
        'accuracy': 100.0 * correct / max(1, total),
        'macro_accuracy': sum(v['accuracy'] for v in per_class.values()) / NUM_CLASSES,
        'correct': correct,
        'total': total,
        'per_class': per_class,
        'confusion': confusion,
    }


def print_metrics(title, metrics):
    print(f'\n[{title}] Overall: {metrics["accuracy"]:.2f}% | Macro: {metrics["macro_accuracy"]:.2f}% | Loss: {metrics["loss"]:.3f}')
    print('  %-10s %10s %14s' % ('class', 'accuracy', 'correct/total'))
    for name in FER_CLASS_NAMES:
        row = metrics['per_class'][name]
        print('  %-10s %9.2f%% %7d/%d' % (name, row['accuracy'], row['correct'], row['total']))


def save_confusion_matrix(confusion, output_path, title):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(confusion, cmap='Blues')
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_xticks(np.arange(NUM_CLASSES))
    ax.set_yticks(np.arange(NUM_CLASSES))
    ax.set_xticklabels(FER_CLASS_NAMES, rotation=45, ha='right')
    ax.set_yticklabels(FER_CLASS_NAMES)
    fig.colorbar(im, ax=ax)

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, str(confusion[i, j]), ha='center', va='center', color='black')

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def write_metrics_csv(path, metrics):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['metric', 'value'])
        writer.writerow(['loss', metrics['loss']])
        writer.writerow(['accuracy', metrics['accuracy']])
        writer.writerow(['macro_accuracy', metrics['macro_accuracy']])
        for name in FER_CLASS_NAMES:
            row = metrics['per_class'][name]
            writer.writerow([f'{name}_accuracy', row['accuracy']])
            writer.writerow([f'{name}_correct', row['correct']])
            writer.writerow([f'{name}_total', row['total']])


def add_common_args(parser):
    parser.add_argument('--model', default=default_model_path())
    parser.add_argument('--rafce-test-img', default='datasets/RafceDataset/test/img')
    parser.add_argument('--rafce-test-lbl', default='datasets/RafceDataset/test/pre-processing/RAFCE_emolabel.txt')
    parser.add_argument('--rafce-train-img', default='datasets/RafceDataset/train/augmented_img')
    parser.add_argument('--rafce-train-lbl', default='datasets/RafceDataset/train/after-processing/RAFCE_emolabel.txt')
    parser.add_argument('--output-dir', default=str(THIS_DIR / 'outputs'))
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', choices=['cuda', 'cpu'])
    return parser


def parse_common_args(description):
    parser = argparse.ArgumentParser(description=description)
    add_common_args(parser)
    return parser
