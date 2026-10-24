# Model Card

## Model Details
* **Model Name:** EfficientNet-B0 Deepfake Detector
* **Architecture:** EfficientNet-B0 (pre-trained on ImageNet) with a custom single-node binary classification head.
* **Developer:** Project Developer
* **Version:** 1.0

## Intended Use
* **Primary Intended Uses:** Research, educational purposes, and demonstration of deep learning techniques applied to computer vision and image forensics.
* **Primary Intended Users:** Students, researchers, and developers exploring deepfake detection.
* **Out-of-Scope Use Cases:** 
  * **Non-intended use:** This model is **NOT** intended for use as forensic evidence in legal proceedings, automated production content moderation without human-in-the-loop and further extensive validation, or making definitive claims about the authenticity of media in critical scenarios.

## Training Data
* **Source:** Models trained using this pipeline are typically trained on datasets like Celeb-DF v2 (or subsets thereof) and generic facial image datasets.
* **Preprocessing:** Images undergo MTCNN face detection, cropping with a margin, resizing to 224x224, and ImageNet standardization (mean/std).

## Model Architecture and Configuration
* **Classes:** REAL = 0, DEEPFAKE = 1.
* **Threshold:** Configurable. Default is 0.5, but optimal thresholds should be derived from validation set performance.
* **Outputs:** A single probability score [0, 1] representing the likelihood of the image being a DEEPFAKE.

## Evaluation Metrics (Template)
*(To be filled after training an official release model)*
* **Test Accuracy:** TBD
* **Precision:** TBD
* **Recall:** TBD
* **F1-Score:** TBD
* **ROC-AUC:** TBD

## Limitations and Bias
* **Dataset Dependency:** The model's performance is highly dependent on the training data. If trained on Celeb-DF, it excels at finding artifacts specific to that generation method but may fail on entirely novel generative techniques (e.g., Diffusion models, if not in the training set).
* **Domain Shift:** Performance may degrade significantly on images with different lighting, resolutions, or compressions than the training set.
* **Compression Artifacts:** Heavy social media compression can obliterate the subtle visual artifacts the model relies upon, leading to false negatives.
* **Face Detection Errors:** The pipeline relies on MTCNN. If MTCNN fails to detect a face, or detects a non-face object, the downstream classification will be invalid.
* **Calibration Limitations:** The output probability is a confidence score of the neural network, not a statistically calibrated probability of truth.

## Ethical Considerations and Statement
**Important Statement:** Model probability is not equivalent to forensic certainty. Grad-CAM is an explanation mechanism showing where the model focused, not definitive proof of algorithmic manipulation. Misuse of this tool can lead to false accusations or the unwarranted dismissal of genuine media.

## Deployment Context
Designed to be deployed as part of an educational or research web application utilizing FastAPI and a standard HTML/JS frontend, running inference on CPU or single GPU setups.
