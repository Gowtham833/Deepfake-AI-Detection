# Architecture Documentation

## System Overview
The Deepfake Detection System is designed to accurately classify facial images as either "REAL" (0) or "DEEPFAKE" (1) using a deep learning model based on the EfficientNet-B0 architecture. The system is split into two major environments: the training/evaluation environment (research) and the inference/deployment environment (production). It incorporates explainability via Grad-CAM to highlight manipulated regions, providing a user-friendly web interface for real-time inference.

## Data Flow Diagram

```text
[Dataset (Images/Videos)] 
        |
        v
[Data Pipeline] ---> (Frame Extraction) ---> (Face Detection & Cropping - MTCNN) ---> (Resizing & Normalization)
        |
        v
[Training Process] ---> (Feature Extractor: EfficientNet-B0) ---> (Binary Classification Head) 
        |                                                              |
        v                                                              v
[Model Artifacts] <-------------------------------------- (Loss Calculation & Optimization)
        |
        v
[Inference Service] <--- (Web Interface / FastAPI) <--- [User Uploads Image]
        |
        v
[Model Prediction + Grad-CAM Heatmap]
        |
        v
[Frontend Display (Result + Heatmap)]
```

## Component Descriptions

### Data Pipeline
Handles the ingestion, preprocessing, and augmentation of raw data. It ensures data is split properly (group-aware to prevent data leakage) and creates PyTorch DataLoaders for model consumption.

### Model
A PyTorch-based neural network utilizing an EfficientNet-B0 backbone pre-trained on ImageNet. The final classification layer is replaced with a custom binary classification head outputting a single logit.

### Training
Manages the two-stage transfer learning process (frozen backbone followed by fine-tuning). It tracks metrics, applies learning rate scheduling, handles early stopping, and saves the best model checkpoints.

### Evaluation
Evaluates model checkpoints against validation and test datasets. Calculates comprehensive metrics including Accuracy, Precision, Recall, F1-Score, ROC-AUC, and PR-AUC. Helps determine optimal probability thresholds.

### Grad-CAM
Gradient-weighted Class Activation Mapping (Grad-CAM) provides explainability by producing visual heatmaps that highlight which parts of the facial image strongly influenced the model's prediction.

### Inference Service
A dedicated service class that encapsulates model loading, preprocessing of single inference images, model execution, Grad-CAM generation, and formatting of outputs.

### FastAPI API
The backend web server providing RESTful endpoints for the frontend. It handles file uploads, routes requests to the Inference Service, and returns JSON responses containing predictions and base64-encoded heatmaps.

### Frontend
A vanilla HTML/CSS/JS web application that allows users to upload images, view predictions, probabilities, and Grad-CAM heatmaps in an interactive interface.

## Preprocessing Flow
1. **Input**: Raw image.
2. **Face Detection**: MTCNN detects bounding boxes of faces.
3. **Face Selection**: If multiple faces exist, the largest by area is selected.
4. **Cropping**: Face is cropped with a configurable margin.
5. **Resizing**: Cropped face is resized to 224x224 pixels.
6. **Normalization**: Pixel values are normalized using standard ImageNet mean and standard deviation.

## Training Flow
1. **Stage 1 (Frozen)**: The EfficientNet-B0 backbone is frozen. Only the new custom classification head is trained with a relatively higher learning rate to adapt to the binary classification task.
2. **Stage 2 (Fine-tune)**: The entire network (or upper layers) is unfrozen and trained with a much smaller learning rate. This allows the feature extractor to adjust to domain-specific features (forensic artifacts) without catastrophic forgetting.

## Inference Flow
1. User uploads an image via the Frontend.
2. FastAPI receives the image and passes it to the Inference Service.
3. Inference Service applies the Preprocessing Flow.
4. Preprocessed tensor is passed through the loaded model.
5. Model outputs a single logit.
6. Sigmoid activation converts the logit to a probability [0, 1].
7. Prediction (REAL/DEEPFAKE) is determined based on the optimal threshold.

## Grad-CAM Flow
1. During inference, gradients are allowed to flow to a target convolutional layer (usually the final conv layer of EfficientNet-B0).
2. The model's output (logit) is backpropagated.
3. Gradients of the target layer are global-average-pooled to obtain channel weights.
4. Target layer feature maps are multiplied by these weights and summed to create a 2D heatmap.
5. ReLU is applied to the heatmap to keep only features that positively contribute to the class.
6. Heatmap is upsampled to the original image size, colored, and blended with the original image.

## API Flow
`Client -> HTTP POST /api/predict (Multipart/form-data) -> FastAPI -> InferenceService.predict(image) -> Model Forward Pass -> GradCAM Generation -> InferenceService constructs response -> FastAPI returns JSON -> Client renders UI`

## Frontend-Backend Communication
Communication is achieved via standard HTTP REST over a predefined port (e.g., 8000). The frontend utilizes JavaScript `fetch` API to send FormData containing the image file to the backend's `/api/predict` endpoint. The backend responds with a JSON object containing the prediction class, confidence score, and a base64 string of the Grad-CAM visualization.

## Model Artifact Flow
Training generates checkpoint files (`.pth`), saving the state dictionary of the model when validation metrics improve. The `best_model.pth` is selected at the end of training. For inference, the Inference Service loads this specific checkpoint into memory once at startup to ensure fast request handling.

## Directory Structure Explanation
- `configs/`: YAML configuration files defining all hyperparameter and path settings.
- `docs/`: Markdown files detailing architecture, methodology, protocols, and model cards.
- `data/`: Local storage for datasets (raw, processed, splits).
- `src/`: Core Python modules (data handling, models, training, evaluation, utils).
- `scripts/`: Executable entry points for training, evaluation, and dataset prep.
- `tests/`: Unit and integration tests.
- `app/`: FastAPI backend application and Inference Service.
- `static/`: Frontend HTML, CSS, JS files.
- `models/`: Directory to store downloaded or trained model checkpoints.
- `notebooks/`: Jupyter notebooks for exploratory data analysis and prototyping.
