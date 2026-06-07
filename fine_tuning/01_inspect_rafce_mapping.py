from collections import Counter
from pathlib import Path

from common import (
    FER_CLASS_NAMES,
    RAFCE_CLASS_NAMES,
    RAFCE_TO_FER,
    RAFCEMappedDataset,
    parse_common_args,
)


def print_distribution(title, dataset):
    compound_counts = Counter(dataset.compound_labels)
    mapped_counts = Counter(dataset.labels)

    print(f'\n{title}')
    print(f'Images: {len(dataset)}')
    print('\nRAF-CE compound classes:')
    for idx, name in enumerate(RAFCE_CLASS_NAMES):
        print(f'  {idx:2d}  {name:24s} {compound_counts[idx]:5d} -> {FER_CLASS_NAMES[RAFCE_TO_FER[idx]]}')

    print('\nMapped FER classes:')
    for idx, name in enumerate(FER_CLASS_NAMES):
        print(f'  {idx}  {name:10s} {mapped_counts[idx]:5d}')


def main():
    parser = parse_common_args('Inspect RAF-CE labels and their FER2013 mapping.')
    opt = parser.parse_args()

    train = RAFCEMappedDataset(opt.rafce_train_img, opt.rafce_train_lbl, transform=None)
    test = RAFCEMappedDataset(opt.rafce_test_img, opt.rafce_test_lbl, transform=None)

    print_distribution('RAF-CE train mapping', train)
    print_distribution('RAF-CE test mapping', test)

    output_dir = Path(opt.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping_path = output_dir / 'rafce_to_fer_mapping.txt'
    with open(mapping_path, 'w', encoding='utf-8') as handle:
        for idx, name in enumerate(RAFCE_CLASS_NAMES):
            handle.write(f'{idx}\t{name}\t{FER_CLASS_NAMES[RAFCE_TO_FER[idx]]}\n')
    print(f'\nSaved mapping: {mapping_path}')


if __name__ == '__main__':
    main()
