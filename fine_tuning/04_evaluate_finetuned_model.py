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
    parser = parse_common_args('Evaluate fine-tuned VGG19 on RAF-CE mapped labels.')
    parser.add_argument('--fine-tuned-model', default=None)
    opt = parser.parse_args()

    output_dir = Path(opt.output_dir)
    model_path = opt.fine_tuned_model or str(output_dir / 'fine_tuned' / 'Best_model.t7')

    device = torch.device(opt.device)
    model = build_vgg19(model_path, device)
    criterion = nn.CrossEntropyLoss()

    test_dataset = RAFCEMappedDataset(opt.rafce_test_img, opt.rafce_test_lbl, transform=make_test_transform())
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)

    metrics = evaluate(model, test_loader, criterion, device)
    print_metrics('Fine-tuned RAF-CE mapped labels', metrics)

    eval_dir = output_dir / 'fine_tuned_eval'
    write_metrics_csv(eval_dir / 'metrics.csv', metrics)
    save_confusion_matrix(metrics['confusion'], eval_dir / 'confusion_matrix.png', 'Fine-tuned RAF-CE mapped labels')
    print(f'\nSaved outputs in: {eval_dir}')


if __name__ == '__main__':
    main()
