# Methodology Documentation

## Dataset Preparation
The primary dataset targeted is Celeb-DF v2, known for its high-quality, challenging deepfakes. Although Celeb-DF v2 is a video dataset, this project focuses on image-based detection. The data pipeline is designed to handle video-derived datasets by extracting frames, as well as generic image datasets. 

## Frame Extraction
For video-derived datasets, frames are extracted at a configurable frame rate (e.g., 1 frame per second) to reduce temporal redundancy. This ensures that the model sees diverse facial expressions and lighting conditions without being overwhelmed by near-identical consecutive frames.

## Group-aware Splitting
A critical methodological step is preventing data leakage. If frames from the same video (or identity) appear in both the training and validation/test sets, the model might learn the identity or background rather than the forensic artifacts of deepfakes. Splitting is done at the video level (or identity group level), ensuring that all frames from a single source video belong exclusively to one split (Train, Val, or Test).

## Face Detection using MTCNN
The Multi-task Cascaded Convolutional Networks (MTCNN) is used for face detection. It is robust and provides accurate bounding boxes. Identifying the face is crucial because deepfake artifacts are localized to the facial region. 

## Face Selection Strategy
In frames with multiple detected faces, the system selects the face with the largest bounding-box area. This heuristic assumes that the primary subject of the video is the largest, most prominent face.

## Face Cropping with Configurable Margin
Once the bounding box is found, the face is cropped. A configurable margin (e.g., 10%) is added around the bounding box to include context such as hair boundaries and the jawline, where blending artifacts often occur in deepfakes.

## Resizing to 224x224
The cropped face image is resized to 224x224 pixels. This is the standard input size for EfficientNet-B0 and ensures consistency across the dataset.

## Normalization
The resized image is normalized using standard ImageNet weights (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]). Since we are using an EfficientNet pre-trained on ImageNet, matching this normalization scheme is vital for the pre-trained weights to function correctly.

## Augmentation Strategy
Data augmentation is applied during training, but it is intentionally kept conservative. Overly aggressive augmentations (like heavy blurring or extreme color jitter) can destroy the subtle digital artifacts (like blending edges or noise patterns) that characterize deepfakes. Mild rotations, horizontal flips, and slight color adjustments are used to improve generalization without destroying critical signals.

## EfficientNet-B0 Architecture
EfficientNet-B0 is chosen for its excellent balance of parameter efficiency and accuracy. It uses compound scaling to uniformly scale depth, width, and resolution. Its relatively small footprint makes it suitable for faster training and efficient inference.

## Transfer Learning Approach
A two-stage transfer learning approach is employed:
1. **Stage 1 (Feature Extraction)**: The pre-trained EfficientNet-B0 backbone is frozen. Only the newly added classification head is trained. This allows the head to learn how to map the existing, generalized features to the specific real vs. deepfake problem without disrupting the pre-trained weights.
2. **Stage 2 (Fine-Tuning)**: The backbone is unfrozen, and the entire network is trained with a significantly lower learning rate. This allows the convolutional filters to adapt specifically to forensic facial artifacts.

## Binary Classification Head
The final fully connected layer of the EfficientNet is replaced with a custom head designed for binary classification. It outputs a single logit. 

## BCEWithLogitsLoss
The loss function used is `BCEWithLogitsLoss`. This combines a Sigmoid layer and the Binary Cross Entropy Loss in one single class, which is numerically more stable than using a plain Sigmoid followed by a BCELoss. REAL=0, DEEPFAKE=1.

## Optimizer and Scheduler
- **AdamW** is used as the optimizer, providing adaptive learning rates with decoupled weight decay, which helps prevent overfitting.
- **ReduceLROnPlateau** is used as the learning rate scheduler. It reduces the learning rate when the validation loss stops improving, allowing the model to converge more finely.

## Early Stopping
To prevent overfitting, training monitors the validation loss. If the validation loss does not improve for a configured number of epochs (patience), training is halted, and the best model weights are restored.

## Evaluation Metrics
The model is evaluated using a comprehensive suite of metrics:
- **Accuracy**: Overall correct prediction rate.
- **Precision**: Proportion of positive identifications (deepfakes) that were actually correct.
- **Recall**: Proportion of actual positive cases (deepfakes) that were identified correctly.
- **F1 Score**: Harmonic mean of precision and recall.
- **ROC-AUC**: Area under the Receiver Operating Characteristic curve; measures ability to separate classes across all thresholds.
- **PR-AUC**: Area under the Precision-Recall curve; particularly useful when classes are imbalanced.
- **Confusion Matrix**: Visualizes True Positives, True Negatives, False Positives, and False Negatives.

## Threshold Selection
The default threshold for deciding between REAL (0) and DEEPFAKE (1) is 0.5. However, an optimal threshold can be selected based *only on the validation set* to maximize a specific metric (like F1 score). This threshold is then fixed and applied to the test set to ensure an unbiased evaluation.

## Grad-CAM Methodology
Gradient-weighted Class Activation Mapping (Grad-CAM) is used for visual explainability. 
- It computes the gradient of the target output (the logit for the predicted class) with respect to the feature maps of a target convolutional layer (usually the last one before the classifier).
- These gradients are global-average-pooled over the spatial dimensions to obtain "importance weights" for each feature channel.
- A weighted combination of the feature maps is computed, followed by a ReLU activation to retain only the features that have a positive influence on the target class prediction.
- The resulting heatmap is overlaid onto the original image, highlighting the regions the model focused on.

## Limitations
- **Domain Shift**: The model may perform poorly on deepfake generation methods it was not trained on (e.g., trained on swap, tested on full synthesis).
- **Compression**: High levels of video compression (e.g., social media uploads) can destroy the subtle artifacts the model relies on.
- **Dataset Bias**: The model might inadvertently learn specific demographic features or background artifacts present in the training data if not perfectly balanced.
- **Explainability vs. Proof**: Grad-CAM shows *where* the model is looking, not *why*. It is an interpretability tool, not cryptographic proof of manipulation.
