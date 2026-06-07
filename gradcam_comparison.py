"""GradCAM comparison: original FER model vs fine-tuned mixed model.

Generates a 2x3 figure:
  Rows    : FER2013 "happy" image  |  RAF-CE "Happily Surprised/Disgusted" image
  Columns : Original image  |  GradCAM (FER model)  |  GradCAM (fine-tuned model)

Usage:
  python gradcam_comparison.py
  python gradcam_comparison.py --fer-img datasets/new_data/test/happy/PrivateTest_10077120.jpg
                               --rafce-img datasets/other_data/test/img/0064.jpg
"""

import argparse
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as mcm
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import image_transforms.transforms as transforms
from model_architectures.vgg import VGG

# ── constants ──────────────────────────────────────────────────────────────
FER_CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

RAFCE_LABEL_NAMES = {
    0: 'Happily Surprised', 1: 'Happily Disgusted',
    2: 'Sadly Fearful',     3: 'Sadly Angry',
    4: 'Sadly Surprised',   5: 'Sadly Disgusted',
    6: 'Fearfully Angry',   7: 'Fearfully Surprised',
    8: 'Angrily Surprised', 9: 'Angrily Disgusted',
    10: 'Disgustedly Surprised', 11: 'Appalled',
    12: 'Hatred',           13: 'Awed',
}

# compound label → FER index (must match finetune_rafce.py)
RAFCE_TO_FER = {
    0: 3, 1: 3,   # happy
    2: 5, 3: 5, 4: 5, 5: 5,  # sad
    6: 2, 7: 2,   # fear
    8: 0, 9: 0, 12: 0,  # angry
    10: 1, 11: 1, # disgust
    13: 6,        # surprise
}
# FER index → compound labels that map to it
FER_TO_RAFCE = {}
for compound, fer in RAFCE_TO_FER.items():
    FER_TO_RAFCE.setdefault(fer, []).append(compound)

# emotions that have RAF-CE compound counterparts (neutral has none)
EMOTIONS_WITH_RAFCE = [e for e in FER_CLASS_NAMES if e != 'neutral']

# Last Conv2d in block5 of VGG19 features Sequential
GRADCAM_LAYER_IDX = 49

IMAGE_SIZE = 48
CROP_SIZE  = 44

try:
    BILINEAR = Image.Resampling.BILINEAR
except AttributeError:
    BILINEAR = Image.BILINEAR


# ── GradCAM ────────────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model      = model
        self.activations = None
        self.gradients   = None
        self._fh = target_layer.register_forward_hook(self._save_act)
        self._bh = target_layer.register_full_backward_hook(self._save_grad)

    def _save_act(self, _, __, output):
        self.activations = output.detach()

    def _save_grad(self, _, __, grad_output):
        self.gradients = grad_output[0].detach()

    def close(self):
        self._fh.remove(); self._bh.remove()

    def generate(self, input_tensor, target_class):
        self.model.eval()
        self.model.zero_grad()
        out = self.model(input_tensor)
        out[:, target_class].sum().backward()

        grads = self.gradients[0]
        acts  = self.activations[0]
        weights = grads.mean(dim=(1, 2))
        cam = sum(w * a for w, a in zip(weights, acts))
        cam = F.relu(cam)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.detach().cpu().numpy(), out.detach()


# ── model helpers ──────────────────────────────────────────────────────────
def load_model(checkpoint_path, device, num_classes=7):
    net = VGG('VGG19')
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt['net'] if isinstance(ckpt, dict) and 'net' in ckpt else ckpt
    if isinstance(state, torch.nn.Module):
        state = state.state_dict()
    # fine-tuned mixed model has same 7-class head — strict=True for both
    net.load_state_dict(state, strict=True)
    net.to(device).eval()
    return net


# ── image helpers ──────────────────────────────────────────────────────────
transform_infer = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.CenterCrop(CROP_SIZE),
    transforms.ToTensor(),
])


def load_image_tensor(path, device):
    img_pil = Image.open(path).convert('L').convert('RGB')
    tensor  = transform_infer(img_pil).unsqueeze(0).to(device)
    return tensor


def to_display(path):
    """Return a uint8 RGB numpy array sized for display."""
    img = Image.open(path).convert('RGB').resize((224, 224), BILINEAR)
    return np.array(img)


def overlay_gradcam(img_rgb, cam_np, alpha=0.45):
    """Blend GradCAM heatmap onto img_rgb (224×224 uint8 RGB) using PIL/matplotlib."""
    h, w = img_rgb.shape[:2]
    # resize cam with PIL
    cam_pil = Image.fromarray(np.uint8(255 * cam_np)).resize((w, h), BILINEAR)
    cam_resized = np.array(cam_pil) / 255.0
    # apply jet colormap via matplotlib (returns RGBA float)
    heatmap = np.uint8(mcm.jet(cam_resized)[:, :, :3] * 255)
    blended = np.uint8(alpha * heatmap + (1 - alpha) * img_rgb)
    return blended


def predict_label(logits):
    probs = torch.softmax(logits, dim=1)[0]
    idx   = int(probs.argmax())
    return FER_CLASS_NAMES[idx], float(probs[idx]) * 100


# ── find images ────────────────────────────────────────────────────────────
def pick_emotion(emotion=None):
    """Return a valid FER emotion name, random if not specified."""
    if emotion:
        if emotion not in EMOTIONS_WITH_RAFCE:
            raise ValueError('Emotion "%s" has no RAF-CE counterpart. Choose from: %s'
                             % (emotion, EMOTIONS_WITH_RAFCE))
        return emotion
    return random.choice(EMOTIONS_WITH_RAFCE)


def find_random_fer_image(emotion):
    folder = Path('datasets/new_data/test') / emotion
    candidates = [p for p in folder.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
    if not candidates:
        raise FileNotFoundError('No images in %s' % folder)
    return str(random.choice(candidates))


def find_random_rafce_image(emotion, label_file, img_dir):
    fer_idx = FER_CLASS_NAMES.index(emotion)
    matching = FER_TO_RAFCE.get(fer_idx, [])
    candidates = []
    with open(label_file, encoding='utf-8') as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) >= 2 and int(parts[1]) in matching:
                p = Path(img_dir) / parts[0]
                if p.exists():
                    candidates.append((str(p), int(parts[1])))
    if not candidates:
        raise FileNotFoundError('No RAF-CE images mapping to "%s"' % emotion)
    return random.choice(candidates)


# ── plotting ───────────────────────────────────────────────────────────────
def make_comparison_figure(
        fer_path, rafce_path, rafce_compound_label,
        emotion, model_orig, model_ft,
        device, output_path):

    target_idx = FER_CLASS_NAMES.index(emotion)

    # ---- run GradCAM for all 4 combinations ----
    gcam_orig = GradCAM(model_orig, model_orig.features[GRADCAM_LAYER_IDX])
    gcam_ft   = GradCAM(model_ft,   model_ft.features[GRADCAM_LAYER_IDX])

    fer_tensor   = load_image_tensor(fer_path,   device)
    rafce_tensor = load_image_tensor(rafce_path, device)

    cam_fer_orig,   logits_fo = gcam_orig.generate(fer_tensor,   target_idx)
    cam_rafce_orig, logits_ro = gcam_orig.generate(rafce_tensor, target_idx)
    cam_fer_ft,     logits_ff = gcam_ft.generate(fer_tensor,     target_idx)
    cam_rafce_ft,   logits_rf = gcam_ft.generate(rafce_tensor,   target_idx)

    gcam_orig.close(); gcam_ft.close()

    pred_fo, conf_fo = predict_label(logits_fo)
    pred_ro, conf_ro = predict_label(logits_ro)
    pred_ff, conf_ff = predict_label(logits_ff)
    pred_rf, conf_rf = predict_label(logits_rf)

    # ---- build display images ----
    fer_rgb   = to_display(fer_path)
    rafce_rgb = to_display(rafce_path)

    ov_fer_orig   = overlay_gradcam(fer_rgb,   cam_fer_orig)
    ov_rafce_orig = overlay_gradcam(rafce_rgb, cam_rafce_orig)
    ov_fer_ft     = overlay_gradcam(fer_rgb,   cam_fer_ft)
    ov_rafce_ft   = overlay_gradcam(rafce_rgb, cam_rafce_ft)

    # ---- figure layout: 2 rows × 3 cols ----
    fig = plt.figure(figsize=(13, 9))
    gs  = gridspec.GridSpec(2, 3, figure=fig,
                            hspace=0.25, wspace=0.08,
                            left=0.08, right=0.98, top=0.84, bottom=0.05)

    col_titles = ['Original image', 'GradCAM — FER model', 'GradCAM — Fine-tuned model']
    row_labels = [
        'FER2013\n"%s"' % emotion,
        'RAF-CE\n"%s"' % RAFCE_LABEL_NAMES.get(rafce_compound_label, emotion),
    ]

    images = [
        [fer_rgb,   ov_fer_orig,   ov_fer_ft],
        [rafce_rgb, ov_rafce_orig, ov_rafce_ft],
    ]
    subtitles = [
        ['',
         'predicts: %s  (%.0f%%)' % (pred_fo, conf_fo),
         'predicts: %s  (%.0f%%)' % (pred_ff, conf_ff)],
        ['',
         'predicts: %s  (%.0f%%)' % (pred_ro, conf_ro),
         'predicts: %s  (%.0f%%)' % (pred_rf, conf_rf)],
    ]

    for row in range(2):
        for col in range(3):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(images[row][col])
            ax.axis('off')
            # column header: use ax.set_title only on top row (no overlap with suptitle)
            if row == 0:
                ax.set_title(col_titles[col], fontsize=11, fontweight='bold', pad=6)
            # prediction text inside image at bottom
            if subtitles[row][col]:
                ax.text(0.5, 0.03, subtitles[row][col],
                        transform=ax.transAxes, ha='center', va='bottom',
                        fontsize=9, color='white',
                        bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.55))
            # row label on left side of first column
            if col == 0:
                ax.text(-0.12, 0.5, row_labels[row],
                        transform=ax.transAxes, ha='right', va='center',
                        fontsize=11, fontweight='bold', multialignment='center')

    fig.suptitle('GradCAM comparison — emotion: %s' % emotion,
                 fontsize=13, fontweight='bold', y=0.97)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved -> %s' % output_path)

    # print prediction summary
    fer_emotion = FER_CLASS_NAMES[target_idx]
    print('\n%-35s  %-12s  %-12s' % ('Image', 'FER model', 'Fine-tuned'))
    print('-' * 62)
    print('%-35s  %-12s  %-12s' % (
        'FER2013 "%s"' % fer_emotion,
        '%s %.0f%%' % (pred_fo, conf_fo),
        '%s %.0f%%' % (pred_ff, conf_ff)))
    print('%-35s  %-12s  %-12s' % (
        RAFCE_LABEL_NAMES.get(rafce_compound_label, '?'),
        '%s %.0f%%' % (pred_ro, conf_ro),
        '%s %.0f%%' % (pred_rf, conf_rf)))


# ── main ───────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--emotion',         default=None,
                   help='FER emotion to visualise (random if omitted). '
                        'Choices: %s' % EMOTIONS_WITH_RAFCE)
    p.add_argument('--fer-img',         default=None)
    p.add_argument('--rafce-img',       default=None)
    p.add_argument('--rafce-label',     default=None, type=int,
                   help='RAF-CE compound label index (optional, used with --rafce-img)')
    p.add_argument('--fer-checkpoint',  default='checkpoints/PrivateTest_model.t7')
    p.add_argument('--ft-checkpoint',   default='checkpoints/Best_model.t7')
    p.add_argument('--rafce-test-img',  default='datasets/other_data/test/img')
    p.add_argument('--rafce-test-lbl',  default='datasets/other_data/test/pre-processing/RAFCE_emolabel.txt')
    p.add_argument('--output',          default='outputs/gradcam_comparison.png')
    p.add_argument('--device',          default='cuda', choices=['cuda', 'cpu'])
    return p.parse_args()


def main():
    opt = parse_args()

    if opt.device == 'cuda' and not torch.cuda.is_available():
        opt.device = 'cpu'
        print('CUDA not available, using CPU')
    device = torch.device(opt.device)

    # pick emotion (random if not specified)
    emotion = pick_emotion(opt.emotion)
    print('Emotion: %s' % emotion)

    print('\nLoading original FER model...')
    model_orig = load_model(opt.fer_checkpoint, device)

    print('Loading fine-tuned mixed model...')
    model_ft = load_model(opt.ft_checkpoint, device)

    fer_path = opt.fer_img or find_random_fer_image(emotion)
    if opt.rafce_img:
        rafce_path = opt.rafce_img
        rafce_label = opt.rafce_label if opt.rafce_label is not None else \
                      FER_TO_RAFCE[FER_CLASS_NAMES.index(emotion)][0]
    else:
        rafce_path, rafce_label = find_random_rafce_image(
            emotion, opt.rafce_test_lbl, opt.rafce_test_img)

    print('FER image   : %s' % fer_path)
    print('RAF-CE image: %s  (%s)' % (rafce_path, RAFCE_LABEL_NAMES.get(rafce_label, '?')))

    make_comparison_figure(
        fer_path, rafce_path, rafce_label,
        emotion, model_orig, model_ft,
        device, opt.output)


if __name__ == '__main__':
    main()
