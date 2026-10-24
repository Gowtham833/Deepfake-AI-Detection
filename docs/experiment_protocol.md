# Experiment Protocol

This protocol defines the standardized procedure for conducting deepfake detection experiments to ensure reproducibility and scientific integrity.

## Pre-experiment Checklist
- [ ] Ensure project environment is fully activated and dependencies are installed.
- [ ] Verify `configs/config.yaml` is set up with intended parameters for the specific experiment.
- [ ] Verify that GPU is available and recognized by PyTorch.
- [ ] Ensure sufficient disk space for datasets and model checkpoints.

## Dataset Preparation Steps
1. Download raw data (e.g., Celeb-DF v2 videos).
2. Organize raw data into the expected directory structure (REAL and DEEPFAKE folders).
3. Run `python scripts/prepare_data.py --config configs/config.yaml`.
4. Monitor the logs for MTCNN detection failures and frame extraction counts.

## Data Split Verification
- [ ] Verify that train, val, and test splits have been created.
- [ ] **CRITICAL**: Verify group-aware splitting. Ensure no source video ID exists in both the train set and the val/test set to prevent data leakage.
- [ ] Check class balance (REAL vs DEEPFAKE) across splits.

## Model Initialization
- [ ] Confirm EfficientNet-B0 backbone initialization with ImageNet weights.
- [ ] Confirm random seed is set (e.g., `set_seed(42)`) for reproducibility of weight initialization in the custom head.

## Training Phase 1: Frozen Backbone
- Run stage 1 training: Freeze EfficientNet-B0 layers, train only the classification head.
- Expected hyperparameter range: Higher Learning Rate (e.g., 1e-3).
- Monitor validation loss. Early stopping should apply.

## Training Phase 2: Fine-tuning
- Unfreeze the backbone layers.
- Expected hyperparameter range: Lower Learning Rate (e.g., 1e-4 or 1e-5).
- Continue training. Monitor validation loss carefully for overfitting.
- Early stopping based on validation loss is critical here.

## Hyperparameter Documentation
For every experiment, record the following (often captured in the config file):
- Image Size: 224x224
- Batch Size: [value]
- Learning Rate (Stage 1): [value]
- Learning Rate (Stage 2): [value]
- Optimizer: AdamW
- Weight Decay: [value]
- Scheduler: ReduceLROnPlateau (factor, patience)

## Checkpoint Selection
The final model used for testing MUST be the checkpoint that achieved the **best (lowest) validation loss** during training, not necessarily the weights at the final epoch.

## Threshold Selection
1. Evaluate the best checkpoint on the **Validation Set** to get raw probabilities.
2. Determine the optimal threshold (e.g., maximizing F1 score) on the Validation Set.
3. This threshold value is now frozen. **Do not adjust the threshold based on Test Set performance.**

## Final Evaluation on Test Set
- Run `python scripts/evaluate_model.py --checkpoint models/best_model.pth --split test`
- Generate and record all required metrics (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC).
- Generate the Confusion Matrix.

## Reporting Requirements
- Results must be reported strictly on the Test Set.
- Always report both threshold-dependent metrics (Accuracy, F1) and threshold-independent metrics (ROC-AUC).
- Explicitly state that metrics are self-generated from this pipeline, not claimed from original benchmark papers unless explicitly comparing.

## IMPORTANT NOTE
**PROTOCOL VS ACTUAL RESULTS**: This document outlines the *protocol*. Actual experimental results, metrics, and logs must be saved separately in experiment logs or MLflow/Weights & Biases (if integrated), or in a designated results markdown file. Do not hardcode experimental results into this protocol document.

---

### Template for Recording Experiment Results (Separate File/Entry)
* Experiment ID / Date: 
* Dataset Version: 
* Training Duration (Epochs): 
* Best Val Loss: 
* Optimal Threshold (from Val): 
* **Test Metrics**:
  * Accuracy: 
  * Precision: 
  * Recall: 
  * F1-Score: 
  * ROC-AUC: 
* Notes/Observations:
