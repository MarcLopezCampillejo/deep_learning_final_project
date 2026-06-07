# Fine Tune Definitive

Standalone scripts for RAF-CE testing and fine-tuning using the FER2013 VGG19 private model.

Expected local model path:

```text
fine_tuning/PrivateTest_model.t7
```

Expected local RAF-CE dataset path:

```text
fine_tuning/RafceDataset
```

If that file is not present, scripts fall back to:

```text
checkpoints/PrivateTest_model.t7
```

RAF-CE paths can be passed by argument. Defaults are resolved first from this folder:

```text
RafceDataset/train/augmented_img
RafceDataset/train/after-processing/RAFCE_emolabel.txt
RafceDataset/test/img
RafceDataset/test/pre-processing/RAFCE_emolabel.txt
```

## 1. Inspect RAF-CE Mapping

Shows how the 14 RAF-CE compound classes are mapped into the 7 FER classes.

```powershell
python ".\fine_tuning\01_inspect_rafce_mapping.py"
```

## 2. Zero-Shot Evaluation

Applies the FER2013 private model directly to RAF-CE, without retraining.

```powershell
python ".\fine_tuning\02_zero_shot_private_model.py"
```

## 3. Fine-Tune

Fine-tunes the private FER2013 model with the exact mixed FER2013 + RAF-CE three-phase strategy:

- Phase 1: features frozen, head only, `lr=1e-3`, 15 epochs.
- Phase 2: block5 + head trainable, `lr=1e-4`, 10 epochs.
- Phase 3: block5 + head trainable, `lr=1e-5`, 10 epochs.
- Best checkpoint is selected by average macro accuracy across FER2013 and RAF-CE.

```powershell
python ".\fine_tuning\03_fine_tune_private_model.py"
```

To override the exact defaults:

```powershell
python ".\fine_tuning\03_fine_tune_private_model.py" --epochs-phase1 15 --epochs-phase2 10 --epochs-phase3 10 --lr1 0.001 --lr2 0.0001 --lr3 0.00001
```

## 4. Evaluate Fine-Tuned Model

Evaluates the fine-tuned checkpoint and saves metrics plus confusion matrix.

```powershell
python ".\fine_tuning\04_evaluate_finetuned_model.py" --fine-tuned-model ".\checkpoints\Best_model.t7"
```

## 5. Test Fine-Tuned Best Model On FER

Evaluates `outputs/fine_tuned/Best_model.t7` on `FerDataset/test` and reports loss, overall accuracy, macro accuracy, and per-class accuracy.

```powershell
python ".\fine_tuning\05_test_best_model_on_fer.py" --model ".\checkpoints\Best_model.t7"
```

Outputs are written to:

```text
fine_tuning/outputs
```
