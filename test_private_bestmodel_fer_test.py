"""
Evaluate checkpoints/PrivateTest_model.t7 on FerDataset/test.

Uses the same TenCrop test-time evaluation style as mainpro_FER.py and reports:
- overall accuracy
- per-class accuracy
- macro per-class accuracy
- confusion matrix
"""

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

import transforms as transforms
from dataset_paths import FER_TEST_DIR, resolve_dataset_path
from model_architectures import VGG


FER_CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
FER_CLASS_TO_IDX = {name: index for index, name in enumerate(FER_CLASS_NAMES)}
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}
CUT_SIZE = 44


class FERFolderTestDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None, max_samples=None):
        self.root_dir = resolve_dataset_path(root_dir)
        self.transform = transform
        self.images = []
        self.labels = []

        for class_name in FER_CLASS_NAMES:
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue

            for image_path in sorted(class_dir.iterdir()):
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.images.append(str(image_path))
                    self.labels.append(FER_CLASS_TO_IDX[class_name])

        if not self.images:
            raise ValueError('No images found in %s' % self.root_dir)

        if max_samples is not None:
            self.images = self.images[:max_samples]
            self.labels = self.labels[:max_samples]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('L').convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[index]


def build_transform():
    return transforms.Compose([
        transforms.TenCrop(CUT_SIZE),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
    ])


def load_model(model_path, device):
    if not os.path.exists(model_path):
        raise FileNotFoundError('Model not found: %s' % model_path)

    model = VGG('VGG19')
    checkpoint = torch.load(model_path, map_location=device)
    state = checkpoint['net'] if isinstance(checkpoint, dict) and 'net' in checkpoint else checkpoint
    if isinstance(state, nn.Module):
        state = state.state_dict()
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def evaluate(model, loader, device):
    per_class_correct = np.zeros(len(FER_CLASS_NAMES), dtype=np.int64)
    per_class_total = np.zeros(len(FER_CLASS_NAMES), dtype=np.int64)
    confusion = np.zeros((len(FER_CLASS_NAMES), len(FER_CLASS_NAMES)), dtype=np.int64)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            batch_size, ncrops, channels, height, width = inputs.shape
            inputs = inputs.view(-1, channels, height, width).to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            outputs_avg = outputs.view(batch_size, ncrops, -1).mean(1)
            _, predicted = torch.max(outputs_avg.data, 1)

            for true_label, pred_label in zip(targets.cpu().numpy(), predicted.cpu().numpy()):
                per_class_total[true_label] += 1
                confusion[true_label, pred_label] += 1
                if true_label == pred_label:
                    per_class_correct[true_label] += 1

            if (batch_idx + 1) % 10 == 0 or batch_idx + 1 == len(loader):
                print('Processed batch %d/%d' % (batch_idx + 1, len(loader)))

    per_class_accuracy = np.divide(
        per_class_correct,
        per_class_total,
        out=np.zeros_like(per_class_correct, dtype=np.float64),
        where=per_class_total != 0,
    ) * 100.0

    overall_accuracy = 100.0 * per_class_correct.sum() / per_class_total.sum()
    macro_accuracy = float(np.mean(per_class_accuracy))

    return {
        'per_class_correct': per_class_correct,
        'per_class_total': per_class_total,
        'per_class_accuracy': per_class_accuracy,
        'overall_accuracy': overall_accuracy,
        'macro_accuracy': macro_accuracy,
        'confusion': confusion,
    }


def print_results(results):
    print('\n' + '=' * 72)
    print('PRIVATE BEST MODEL TEST RESULTS')
    print('=' * 72)
    print('Overall accuracy: %.2f%%' % results['overall_accuracy'])
    print('Macro per-class accuracy: %.2f%%' % results['macro_accuracy'])
    print()
    print('%-12s %12s %18s' % ('Class', 'Accuracy', 'Correct/Total'))
    print('-' * 46)

    for index, class_name in enumerate(FER_CLASS_NAMES):
        correct = int(results['per_class_correct'][index])
        total = int(results['per_class_total'][index])
        accuracy = results['per_class_accuracy'][index]
        print('%-12s %10.2f%% %8d/%d' % (class_name, accuracy, correct, total))

    print('-' * 46)
    print('%-12s %10.2f%% %8d/%d' % (
        'total',
        results['overall_accuracy'],
        int(results['per_class_correct'].sum()),
        int(results['per_class_total'].sum()),
    ))


def save_outputs(results, output_prefix):
    results_path = '%s_accuracy.csv' % output_prefix
    summary_path = '%s_summary.txt' % output_prefix
    confusion_path = '%s_confusion_matrix.csv' % output_prefix

    with open(results_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['class', 'accuracy', 'correct', 'total'])
        for index, class_name in enumerate(FER_CLASS_NAMES):
            writer.writerow([
                class_name,
                '%.4f' % results['per_class_accuracy'][index],
                int(results['per_class_correct'][index]),
                int(results['per_class_total'][index]),
            ])
        writer.writerow([])
        writer.writerow(['overall_accuracy', '%.4f' % results['overall_accuracy']])
        writer.writerow(['macro_per_class_accuracy', '%.4f' % results['macro_accuracy']])

    with open(confusion_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['true/pred'] + FER_CLASS_NAMES)
        for index, class_name in enumerate(FER_CLASS_NAMES):
            writer.writerow([class_name] + results['confusion'][index].astype(int).tolist())

    with open(summary_path, 'w', encoding='utf-8') as handle:
        handle.write('PRIVATE BEST MODEL TEST RESULTS\n')
        handle.write('=' * 72 + '\n')
        handle.write('Overall accuracy: %.2f%%\n' % results['overall_accuracy'])
        handle.write('Macro per-class accuracy: %.2f%%\n\n' % results['macro_accuracy'])
        for index, class_name in enumerate(FER_CLASS_NAMES):
            handle.write('%-12s %.2f%% %d/%d\n' % (
                class_name,
                results['per_class_accuracy'][index],
                int(results['per_class_correct'][index]),
                int(results['per_class_total'][index]),
            ))

    print('\nSaved:')
    print('  %s' % results_path)
    print('  %s' % summary_path)
    print('  %s' % confusion_path)


def parse_args():
    parser = argparse.ArgumentParser(description='Test PrivateTest_model.t7 on FerDataset/test.')
    parser.add_argument('--model', default='checkpoints/PrivateTest_model.t7')
    parser.add_argument('--test-dir', default=str(FER_TEST_DIR))
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--max-samples', default=None, type=int,
                        help='Optional limit for quick tests. By default evaluates all test images.')
    parser.add_argument('--output-prefix', default='private_bestmodel_fer_test')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                        choices=['cuda', 'cpu'])
    return parser.parse_args()


def main():
    opt = parse_args()
    if opt.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available. Use --device cpu.')

    device = torch.device(opt.device)
    print('Device: %s' % device)
    if device.type == 'cuda':
        print('GPU: %s' % torch.cuda.get_device_name(0))

    print('Model: %s' % opt.model)
    print('Test dataset: %s' % opt.test_dir)

    model = load_model(opt.model, device)
    dataset = FERFolderTestDataset(opt.test_dir, transform=build_transform(), max_samples=opt.max_samples)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=0,
    )

    print('Test samples: %d' % len(dataset))
    results = evaluate(model, loader, device)
    print_results(results)
    save_outputs(results, opt.output_prefix)


if __name__ == '__main__':
    main()
