# Training Accuracy Graph Report

## Generated Figures
- Plant accuracy graph: plant_disease_training_accuracy.png
- Fruit accuracy graph: fruit_disease_training_accuracy.png
- Fruit loss graph: fruit_disease_training_loss.png

## Data Provenance
- Plant source: No epoch-history artifact found; inspected existing plant model/notebook placeholders
- Fruit source: backend\model\training_history.json

## Figure Captions
- Figure 1. Plant Disease Detection Model Training Accuracy. Epoch-wise plant training logs were not present in the repository, so no fabricated curve is shown.
- Figure 2. Fruit Disease Detection Model Training Accuracy. Training and validation accuracy across full two-phase EfficientNet-B0 training, with best-validation epoch marker and final metric annotation.
- Figure 3. Fruit Disease Detection Model Training Loss. Training and validation loss trend across full training for overfitting assessment.

## Metrics Summary
### Plant Disease Detection
- Final training accuracy: N/A
- Final validation accuracy: N/A
- Best validation accuracy: N/A
- Best validation epoch: N/A

### Fruit Disease Detection (EfficientNet-B0)
- Final training accuracy: 96.04%
- Final validation accuracy: 92.26%
- Best validation accuracy: 92.64%
- Best validation epoch: 16

## Interpretation
### Plant Disease Detection
- Training convergence: Epoch-level convergence cannot be assessed from this repository because only the trained .h5 artifact is present, without history logs.
- Validation stability: Not assessable without epoch-wise validation metrics.
- Overfitting/underfitting: Not assessable from available artifacts.
- Learning behavior: The deployment artifact confirms a trained CNN exists, but learning dynamics are unavailable.
- Final performance quality: Requires recovery of original history logs (CSVLogger/JSON/TensorBoard/Notebook outputs) for quantitative curve analysis.

### Fruit Disease Detection
- Training convergence: Accuracy rises quickly in early epochs and then improves gradually, indicating effective transfer learning convergence.
- Validation stability: Validation accuracy remains consistently high in later epochs with modest oscillation, suggesting stable generalization.
- Overfitting/underfitting: The train-validation gap is moderate and controlled; no severe divergence pattern is observed at the best epoch.
- Learning behavior: Two-phase training shows expected behavior: rapid feature-head learning followed by fine-tuned incremental gains.
- Final performance quality: Validation accuracy above 92% supports strong practical classification performance for the 17-class task.