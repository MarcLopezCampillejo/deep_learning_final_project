# FER Center Black Occlusion Evaluation

Evaluates how FER test accuracy changes when a centered black square is applied to every image.
The evaluation uses the FER2013 private checkpoint class order and TenCrop averaging, matching `mainpro_FER.py`.

Default inputs:

```text
Model: checkpoints/PrivateTest_model.t7
FER test dataset: datasets/FerDataset/test
```

Run full evaluation:

```powershell
python ".\occlusion_evaluation\evaluate_fer_test_center_black_occlusion.py"
```

Quick test with 20 images:

```powershell
python ".\occlusion_evaluation\evaluate_fer_test_center_black_occlusion.py" --batch-limit 20
```

Useful options:

```powershell
--patch-size 4
--save-examples 12
```

Outputs:

```text
occlusion_evaluation/outputs/fer_test_center_black_occlusion_accuracy.csv
occlusion_evaluation/outputs/original_confusion_matrix.csv
occlusion_evaluation/outputs/original_confusion_matrix.png
occlusion_evaluation/outputs/occluded_confusion_matrix.csv
occlusion_evaluation/outputs/occluded_confusion_matrix.png
occlusion_evaluation/outputs/examples/
```

The original dataset is not modified; occlusion is applied only in memory.
