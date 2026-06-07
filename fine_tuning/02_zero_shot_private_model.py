from pathlib import Path

import torch
import torch.nn as nn

from common import (
    RAFCEMappedDataset,
    build_vgg19,
    evaluate,
    make_test_transform,
    parse_common_args,
    print_metrics,
    save_confusion_matrix,
    write_metrics_csv,
)


def main():
    parser = parse_common_args('Zero-shot evaluation: FER2013 private model on RAF-CE mapped labels.')
    opt = parser.parse_args()

    device = torch.device(opt.device)
    model = build_vgg19(opt.model, device)
    criterion = nn.CrossEntropyLoss()

    test_dataset = RAFCEMappedDataset(opt.rafce_test_img, opt.rafce_test_lbl, transform=make_test_transform())
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)

    metrics = evaluate(model, test_loader, criterion, device)
    print_metrics('Zero-shot RAF-CE mapped labels', metrics)

    output_dir = Path(opt.output_dir) / 'zero_shot'
    write_metrics_csv(output_dir / 'metrics.csv', metrics)
    save_confusion_matrix(metrics['confusion'], output_dir / 'confusion_matrix.png', 'Zero-shot RAF-CE mapped labels')
    print(f'\nSaved outputs in: {output_dir}')


if __name__ == '__main__':
    main()
