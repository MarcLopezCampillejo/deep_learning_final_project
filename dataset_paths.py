from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent

FER_DATASET_DIR = PROJECT_DIR / 'FerDataset'
RAFCE_DATASET_DIR = PROJECT_DIR / 'RafceDataset'

FER_TRAIN_DIR = FER_DATASET_DIR / 'train'
FER_TEST_DIR = FER_DATASET_DIR / 'test'

RAFCE_TRAIN_IMG_DIR = RAFCE_DATASET_DIR / 'train' / 'augmented_img'
RAFCE_TRAIN_LABEL_FILE = RAFCE_DATASET_DIR / 'train' / 'after-processing' / 'RAFCE_emolabel.txt'
RAFCE_TEST_IMG_DIR = RAFCE_DATASET_DIR / 'test' / 'img'
RAFCE_TEST_LABEL_FILE = RAFCE_DATASET_DIR / 'test' / 'pre-processing' / 'RAFCE_emolabel.txt'


_PATH_ALIASES = (
    ('datasets/FerDataset/train', 'FerDataset/train'),
    ('datasets/FerDataset/test', 'FerDataset/test'),
    ('datasets/FerDataset', 'FerDataset'),
    ('datasets/RafceDataset/train', 'RafceDataset/train'),
    ('datasets/RafceDataset/test', 'RafceDataset/test'),
    ('datasets/RafceDataset/validation', 'RafceDataset/validation'),
    ('datasets/RafceDataset', 'RafceDataset'),
    ('datasets/new_data/test', 'FerDataset/test'),
    ('datasets/new_data/train', 'FerDataset/train'),
    ('datasets/new_data', 'FerDataset'),
    ('datasets/other_data/test', 'RafceDataset/test'),
    ('datasets/other_data/train', 'RafceDataset/train'),
    ('datasets/other_data', 'RafceDataset'),
)


def resolve_dataset_path(path, *base_dirs):
    raw = Path(path)
    if raw.is_absolute():
        return raw

    normalized = raw.as_posix()
    alias_candidates = []
    for old_prefix, new_prefix in _PATH_ALIASES:
        if normalized == old_prefix or normalized.startswith(old_prefix + '/'):
            suffix = normalized[len(old_prefix):].lstrip('/')
            alias_candidates.append(PROJECT_DIR / new_prefix / suffix)

    candidates = [*alias_candidates, *(Path(base_dir) / raw for base_dir in base_dirs), PROJECT_DIR / raw, raw]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else raw
