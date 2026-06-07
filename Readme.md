# VGG19 Facial Expression Recognition

## Overview

This folder contains the code, FER2013 data files, and model checkpoints needed to run the main experiments of the project:

- FER2013 preprocessing, training, and evaluation
- Grad-CAM visualization
- Center black-square occlusion evaluation
- RAF-CE zero-shot evaluation
- FER2013 + RAF-CE fine-tuning
- Fine-tuned model evaluation

## Requirements

Install the required Python packages:

```powershell
pip install torch torchvision numpy h5py pillow matplotlib scikit-learn opencv-python
```

The code was developed for Python 3 and PyTorch.

## Included Files

The submission includes:

```text
datasets/fer2013.csv
datasets/data.h5
checkpoints/PrivateTest_model.t7
checkpoints/Best_model.t7
```

Checkpoint meaning:

- `checkpoints/PrivateTest_model.t7`: original FER2013 model before fine-tuning.
- `checkpoints/Best_model.t7`: best fine-tuned model trained with the mixed FER2013 + RAF-CE strategy.

Some scripts also expect the following folders if RAF-CE or folder-based FER evaluation is executed:

```text
datasets/FerDataset/test/
datasets/RafceDataset/
```

## Folder Structure

```text
.
|-- datasets/
|-- checkpoints/
|-- model_architectures/
|-- image_transforms/
|-- occlusion_evaluation/
|-- fine_tuning/
|-- fer.py
|-- mainpro_FER.py
|-- preprocess_fer2013.py
|-- plot_fer2013_confusion_matrix.py
|-- plot_results.py
|-- test_private_bestmodel_fer_test.py
|-- gradcamVisualize.py
|-- gradcam_comparison.py
`-- utils.py
```

## How to Run

Run all commands from the root of this folder.

### 1. Preprocess FER2013

This step is only needed if `datasets/data.h5` must be regenerated from `datasets/fer2013.csv`.

```powershell
python preprocess_fer2013.py
```

### 2. Train VGG19 on FER2013

```powershell
python mainpro_FER.py --model VGG19 --bs 128 --lr 0.01
```

### 3. Plot Training Results

```powershell
python plot_results.py
```

### 4. Generate FER2013 Confusion Matrix

```powershell
python plot_fer2013_confusion_matrix.py --model VGG19 --split PrivateTest
```

### 5. Evaluate the Original FER2013 Model

```powershell
python test_private_bestmodel_fer_test.py --model ".\checkpoints\PrivateTest_model.t7"
```

This evaluates the model before fine-tuning.

### 6. Generate Grad-CAM Visualizations

```powershell
python gradcamVisualize.py
```

To limit the number of processed images:

```powershell
$env:GRADCAM_MAX_IMAGES=20
python gradcamVisualize.py
```

### 7. Run Center Black-Square Occlusion Evaluation

```powershell
python ".\occlusion_evaluation\evaluate_fer_test_center_black_occlusion.py"
```

Quick test:

```powershell
python ".\occlusion_evaluation\evaluate_fer_test_center_black_occlusion.py" --batch-limit 20
```

### 8. Inspect RAF-CE to FER2013 Label Mapping

```powershell
python ".\fine_tuning\01_inspect_rafce_mapping.py"
```

### 9. Run Zero-Shot Evaluation on RAF-CE

```powershell
python ".\fine_tuning\02_zero_shot_private_model.py"
```

### 10. Fine-Tune on FER2013 + RAF-CE

```powershell
python ".\fine_tuning\03_fine_tune_private_model.py"
```

### 11. Evaluate the Fine-Tuned Model on RAF-CE

```powershell
python ".\fine_tuning\04_evaluate_finetuned_model.py" --fine-tuned-model ".\checkpoints\Best_model.t7"
```

### 12. Evaluate the Fine-Tuned Model on FER

```powershell
python ".\fine_tuning\05_test_best_model_on_fer.py" --model ".\checkpoints\Best_model.t7"
```

### 13. Compare Grad-CAM Before and After Fine-Tuning

```powershell
python gradcam_comparison.py --fer-checkpoint ".\checkpoints\PrivateTest_model.t7" --ft-checkpoint ".\checkpoints\Best_model.t7"
```

## Notes

- `PrivateTest_model.t7` should be used for experiments before fine-tuning.
- `Best_model.t7` should be used for experiments after fine-tuning.
- All datasets must be placed inside the `datasets` folder.
- A new repository was created for the submission.
- The original datasets are not modified by the evaluation or occlusion scripts.
