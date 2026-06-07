"""
Grad-CAM visualization for FER2013 using the original private-test VGG19 checkpoint.

Model: checkpoints/PrivateTest_model.t7
Dataset: FER2013 PrivateTest split from dataset/data.h5
Layer: net.features[33]
"""

import glob
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import image_transforms as transforms
from fer import FER2013
from model_architectures import VGG


CHECKPOINT_PATH = os.path.join('checkpoints', 'PrivateTest_model.t7')
OUTPUT_DIR = os.path.join('outputs', 'fer2013_private_gradcam')
SPLIT = 'PrivateTest'
MAX_IMAGES = int(os.environ['GRADCAM_MAX_IMAGES']) if os.environ.get('GRADCAM_MAX_IMAGES') else None
CROP_SIZE = 44

CLASS_NAMES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

transform_test = transforms.Compose([
    transforms.TenCrop(CROP_SIZE),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.clone().detach().cpu()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].clone().detach().cpu()

    def generate_cam(self, input_tensor, target_class):
        self.model.eval()
        self.model.zero_grad()
        output = self.model(input_tensor)
        output[0, target_class].backward()

        weights = self.gradients[0].mean(dim=(1, 2)).numpy()
        _, _, h, w = self.activations.shape
        cam = np.zeros((h, w), dtype=np.float32)
        activations = self.activations[0].numpy()

        for channel_index, weight in enumerate(weights):
            cam += weight * activations[channel_index]

        cam = np.maximum(cam, 0)
        cam = cam / (cam.max() + 1e-8)
        return cam


def load_private_model(device):
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f'Checkpoint not found: {CHECKPOINT_PATH}')

    model = VGG('VGG19')
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    state_dict = checkpoint['net'] if isinstance(checkpoint, dict) and 'net' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def get_raw_private_image(dataset, index):
    gray = np.asarray(dataset.PrivateTest_data[index]).astype(np.uint8)
    label = int(dataset.PrivateTest_labels[index])
    rgb = np.repeat(gray[:, :, np.newaxis], 3, axis=2)
    return gray, Image.fromarray(rgb), label


def save_gradcam_figure(output_path, gray, cam_resized, score, label, predicted_class):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(gray, cmap='gray')
    axes[0, 0].set_title('Original Image', fontsize=14)
    axes[0, 0].axis('off')

    heatmap = axes[0, 1].imshow(cam_resized, cmap='jet')
    axes[0, 1].set_title('Grad-CAM Map', fontsize=14)
    axes[0, 1].axis('off')
    plt.colorbar(heatmap, ax=axes[0, 1], fraction=0.046, pad=0.04)

    axes[0, 2].imshow(gray.astype(np.float32) / 255.0, cmap='gray')
    axes[0, 2].imshow(cam_resized, cmap='jet', alpha=0.5)
    axes[0, 2].set_title('Overlay', fontsize=14)
    axes[0, 2].axis('off')

    axes[1, 0].axis('off')
    result_text = f'Ground Truth: {CLASS_NAMES[label]}\n'
    result_text += f'Predicted: {CLASS_NAMES[predicted_class]}\n'
    result_text += f"Match: {'yes' if predicted_class == label else 'no'}\n\n"
    result_text += 'Confidence Scores:\n'
    for class_index, class_name in enumerate(CLASS_NAMES):
        result_text += f'{class_name}: {score.data.cpu().numpy()[class_index]:.4f}\n'
    axes[1, 0].text(
        0.1,
        0.5,
        result_text,
        fontsize=10,
        verticalalignment='center',
        family='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
    )

    colors = ['red', 'orangered', 'darkorange', 'limegreen', 'royalblue', 'navy', 'darkgreen']
    indices = np.arange(len(CLASS_NAMES))
    axes[1, 1].bar(indices, score.data.cpu().numpy(), color=colors)
    axes[1, 1].set_title('Classification Scores', fontsize=14)
    axes[1, 1].set_ylabel('Score', fontsize=12)
    axes[1, 1].set_xticks(indices)
    axes[1, 1].set_xticklabels(CLASS_NAMES, rotation=45, ha='right', fontsize=9)

    axes[1, 2].axis('off')
    axes[1, 2].text(0.5, 0.5, 'FER2013\nPrivateTest', ha='center', va='center', fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'Loading model: {CHECKPOINT_PATH}')

    model = load_private_model(device)
    cam_generator = GradCAM(model, model.features[33])

    dataset = FER2013(split=SPLIT, transform=None)
    total_images = len(dataset) if MAX_IMAGES is None else min(MAX_IMAGES, len(dataset))
    print(f'Processing {total_images} images from FER2013 {SPLIT}...')

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    emotion_dirs = {}
    for class_name in CLASS_NAMES:
        emotion_dir = os.path.join(OUTPUT_DIR, class_name.lower())
        os.makedirs(emotion_dir, exist_ok=True)
        emotion_dirs[class_name] = emotion_dir

    for index in range(total_images):
        try:
            print(f'  [{index + 1}/{total_images}] PrivateTest index {index}')
            gray, image, label = get_raw_private_image(dataset, index)
            inputs = transform_test(image)
            ncrops, c, h, w = inputs.shape

            with torch.no_grad():
                outputs = model(inputs.view(-1, c, h, w).to(device))
                outputs_avg = outputs.view(ncrops, -1).mean(0)
                score = F.softmax(outputs_avg, dim=0)
                _, predicted = torch.max(outputs_avg.data, 0)

            predicted_class = int(predicted.cpu())
            single_input = inputs[0:1].to(device).requires_grad_(True)
            cam = cam_generator.generate_cam(single_input, predicted_class)
            cam_resized = cv2.resize(cam, (48, 48))

            filename = f'{index:04d}_true-{CLASS_NAMES[label]}_pred-{CLASS_NAMES[predicted_class]}.png'
            output_path = os.path.join(emotion_dirs[CLASS_NAMES[label]], filename)
            save_gradcam_figure(output_path, gray, cam_resized, score, label, predicted_class)
        except Exception as exc:
            print(f'  Error: {exc}')
            plt.close('all')

    print(f'\nVisualizations saved in: {OUTPUT_DIR}/')
    print('Structure:')
    for class_name in CLASS_NAMES:
        class_dir = emotion_dirs[class_name]
        count = len(glob.glob(os.path.join(class_dir, '*.png')))
        print(f'  {class_name.lower()}/: {count} images')


if __name__ == '__main__':
    main()
