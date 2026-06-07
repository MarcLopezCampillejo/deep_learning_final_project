import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from common import (
    FER_CLASS_NAMES,
    FERDataset,
    RAFCEMappedDataset,
    build_class_weights,
    build_vgg19,
    build_weighted_sampler,
    make_optimizer,
    make_scaler,
    make_test_transform,
    make_train_transform,
    parse_common_args,
    print_metrics,
    save_confusion_matrix,
    set_phase,
    train_one_epoch_mixed,
    validate_mixed,
)


def append_metrics(output_dir, epoch, phase, lr, train_m, fer_m, rafce_m):
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / 'training_metrics.csv'
    write_header = not metrics_path.exists()
    with open(metrics_path, 'a', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow([
                'epoch', 'phase', 'lr',
                'train_loss', 'train_acc',
                'fer_loss', 'fer_acc', 'fer_macro',
                'rafce_loss', 'rafce_acc', 'rafce_macro',
            ])
        writer.writerow([
            epoch, phase, lr,
            train_m['loss'], train_m['accuracy'],
            fer_m['loss'], fer_m['accuracy'], fer_m['macro_accuracy'],
            rafce_m['loss'], rafce_m['accuracy'], rafce_m['macro_accuracy'],
        ])


def save_checkpoint(net, optimizer, output_dir, epoch, phase, fer_m, rafce_m, best_score, source_checkpoint):
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / 'Best_model.t7'
    torch.save({
        'net': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model': 'VGG19_FER_RAFCE',
        'class_names': FER_CLASS_NAMES,
        'source_checkpoint': source_checkpoint,
        'epoch': epoch,
        'phase': phase,
        'best_score': best_score,
        'fer_macro': fer_m['macro_accuracy'],
        'rafce_macro': rafce_m['macro_accuracy'],
    }, checkpoint_path)
    print('Saved best model (score=%.2f%%)  FER macro=%.2f%%  RAF-CE macro=%.2f%%' % (
        best_score, fer_m['macro_accuracy'], rafce_m['macro_accuracy']))


def plot_curves(history, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = [row['epoch'] for row in history]
    train_loss = [row['train_loss'] for row in history]
    fer_loss = [row['fer_loss'] for row in history]
    rafce_loss = [row['rafce_loss'] for row in history]
    fer_macro = [row['fer_macro'] for row in history]
    rafce_macro = [row['rafce_macro'] for row in history]
    phase_changes = [
        idx for idx in range(1, len(history))
        if history[idx]['phase'] != history[idx - 1]['phase']
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(epochs, train_loss, label='Train loss (mixed)')
    ax1.plot(epochs, fer_loss, label='FER2013 test loss')
    ax1.plot(epochs, rafce_loss, label='RAF-CE test loss')
    for idx in phase_changes:
        ax1.axvline(x=epochs[idx], color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Cross-Entropy Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, fer_macro, label='FER2013 macro acc')
    ax2.plot(epochs, rafce_macro, label='RAF-CE macro acc (mapped)')
    for idx in phase_changes:
        ax2.axvline(
            x=epochs[idx],
            color='gray',
            linestyle='--',
            alpha=0.5,
            label='Phase change' if idx == phase_changes[0] else None,
        )
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Macro Accuracy (%)')
    ax2.set_title('Macro Accuracy - Both Datasets')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle('VGG19 Mixed Fine-tuning: FER2013 + RAF-CE', fontsize=13)
    fig.tight_layout()
    output_path = output_dir / 'loss_curves.png'
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f'Loss curves -> {output_path}')


def save_normalized_confusion(targets, predicted, output_dir, prefix):
    confusion = np.zeros((len(FER_CLASS_NAMES), len(FER_CLASS_NAMES)), dtype=np.int64)
    for true_label, pred_label in zip(targets, predicted):
        confusion[true_label, pred_label] += 1
    save_confusion_matrix(confusion, output_dir / f'{prefix}.png', prefix.replace('_', ' ').title())


def main():
    parser = parse_common_args('Fine-tune VGG19 on mixed FER2013 + RAF-CE using the exact three-phase strategy.')
    parser.add_argument('--fer-train-dir', default='FerDataset/train')
    parser.add_argument('--fer-test-dir', default='FerDataset/test')
    parser.add_argument('--epochs-phase1', default=15, type=int)
    parser.add_argument('--epochs-phase2', default=10, type=int)
    parser.add_argument('--epochs-phase3', default=10, type=int)
    parser.add_argument('--lr1', default=1e-3, type=float)
    parser.add_argument('--lr2', default=1e-4, type=float)
    parser.add_argument('--lr3', default=1e-5, type=float)
    parser.add_argument('--num-workers', default=0, type=int)
    parser.add_argument('--no-amp', dest='amp', action='store_false')
    parser.set_defaults(amp=True)
    opt = parser.parse_args()

    if opt.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA not available. Use --device cpu')

    device = torch.device(opt.device)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    use_amp = opt.amp and device.type == 'cuda'
    output_dir = Path(opt.output_dir) / 'fine_tuned'

    print(f'Device: {device}')
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')

    transform_train = make_train_transform()
    transform_test = make_test_transform()

    loader_kwargs = {'num_workers': opt.num_workers, 'pin_memory': device.type == 'cuda'}
    if opt.num_workers > 0:
        loader_kwargs['persistent_workers'] = True

    print('\n==> Loading FER2013 training set')
    fer_train = FERDataset(opt.fer_train_dir, transform=transform_train)
    print('\n==> Loading RAF-CE training set (mapped to FER labels)')
    rafce_train = RAFCEMappedDataset(opt.rafce_train_img, opt.rafce_train_lbl, transform=transform_train)
    print('\n==> Loading FER2013 test set')
    fer_test = FERDataset(opt.fer_test_dir, transform=transform_test)
    print('\n==> Loading RAF-CE test set (mapped to FER labels)')
    rafce_test = RAFCEMappedDataset(opt.rafce_test_img, opt.rafce_test_lbl, transform=transform_test)

    combined = torch.utils.data.ConcatDataset([fer_train, rafce_train])
    combined_labels = fer_train.labels + rafce_train.labels
    print('\nCombined training set: %d images (FER %d + RAF-CE %d)' % (
        len(combined), len(fer_train), len(rafce_train)))

    train_loader = torch.utils.data.DataLoader(
        combined,
        batch_size=opt.batch_size,
        shuffle=False,
        sampler=build_weighted_sampler(combined_labels),
        **loader_kwargs,
    )
    fer_loader = torch.utils.data.DataLoader(
        fer_test, batch_size=opt.batch_size, shuffle=False, **loader_kwargs)
    rafce_loader = torch.utils.data.DataLoader(
        rafce_test, batch_size=opt.batch_size, shuffle=False, **loader_kwargs)

    print('\n==> Building model')
    net = build_vgg19(opt.model, device)
    criterion = nn.CrossEntropyLoss(weight=build_class_weights(combined_labels, device))

    history = []
    best_score = 0.0
    global_epoch = 0
    phase_config = [
        (1, opt.epochs_phase1, opt.lr1),
        (2, opt.epochs_phase2, opt.lr2),
        (3, opt.epochs_phase3, opt.lr3),
    ]

    for phase, num_epochs, lr in phase_config:
        if num_epochs == 0:
            continue

        print('\n' + '=' * 72)
        print('PHASE %d  |  epochs=%d  |  lr=%.0e' % (phase, num_epochs, lr))
        print('=' * 72)

        set_phase(net, phase)
        optimizer = make_optimizer(net, lr)
        scaler = make_scaler(use_amp)

        for local_epoch in range(num_epochs):
            print('\nEpoch %d/%d  (global %d, phase %d)' % (
                local_epoch + 1, num_epochs, global_epoch, phase))

            train_m = train_one_epoch_mixed(net, train_loader, criterion, optimizer, scaler, device, use_amp)

            print('  -> FER2013 test:')
            fer_m = validate_mixed(net, fer_loader, criterion, device, use_amp)
            print('  -> RAF-CE test (mapped):')
            rafce_m = validate_mixed(net, rafce_loader, criterion, device, use_amp)

            print_metrics('FER2013', fer_m)
            print_metrics('RAF-CE (mapped)', rafce_m)

            append_metrics(output_dir, global_epoch, phase, lr, train_m, fer_m, rafce_m)
            history.append({
                'epoch': global_epoch,
                'phase': phase,
                'train_loss': train_m['loss'],
                'fer_loss': fer_m['loss'],
                'fer_macro': fer_m['macro_accuracy'],
                'rafce_loss': rafce_m['loss'],
                'rafce_macro': rafce_m['macro_accuracy'],
            })

            score = (fer_m['macro_accuracy'] + rafce_m['macro_accuracy']) / 2.0
            if score > best_score:
                best_score = score
                save_checkpoint(net, optimizer, output_dir, global_epoch, phase, fer_m, rafce_m, best_score, opt.model)

            global_epoch += 1

    print('\n' + '=' * 72)
    print('Training complete. Best combined macro: %.2f%%' % best_score)
    print('=' * 72)

    best_path = output_dir / 'Best_model.t7'
    if best_path.exists():
        print('\n==> Final evaluation with best model')
        checkpoint = torch.load(best_path, map_location=device)
        net.load_state_dict(checkpoint['net'])

        print('  -> FER2013 test:')
        fer_m = validate_mixed(net, fer_loader, criterion, device, use_amp)
        print_metrics('FER2013 (final)', fer_m)
        save_normalized_confusion(fer_m['all_targets'], fer_m['all_predicted'], output_dir, 'confusion_matrix_fer')

        print('  -> RAF-CE test (mapped):')
        rafce_m = validate_mixed(net, rafce_loader, criterion, device, use_amp)
        print_metrics('RAF-CE mapped (final)', rafce_m)
        save_normalized_confusion(rafce_m['all_targets'], rafce_m['all_predicted'], output_dir, 'confusion_matrix_rafce')

    if history:
        plot_curves(history, output_dir)

    print(f'\nOutputs saved to: {output_dir}/')


if __name__ == '__main__':
    main()
