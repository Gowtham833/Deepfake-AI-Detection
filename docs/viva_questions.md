# Viva Questions and Answers

## PROJECT Section

**1. Why did you choose deepfake detection for your project?**
Deepfakes pose a significant threat to digital trust, enabling misinformation, identity theft, and fraud. By working on detection, we contribute to mitigating these risks. It also serves as an excellent practical application of advanced computer vision and transfer learning techniques.

**2. Why focus on image-based detection instead of video-based?**
While videos contain temporal information, image-based detection is computationally less expensive and forms the foundational step for many video pipelines (which often evaluate frame-by-frame). By mastering spatial artifact detection in individual frames, the system can be more easily deployed in resource-constrained environments like web APIs.

**3. Why use EfficientNet?**
EfficientNet provides state-of-the-art accuracy while being highly parameter-efficient. It uses a compound scaling method that balances network depth, width, and resolution, making it much more efficient than older models like ResNet or VGG for the same level of performance.

**4. Why specifically EfficientNet-B0?**
EfficientNet-B0 is the baseline model of the EfficientNet family, meaning it is the smallest and fastest. For an educational or research deployment, B0 offers a great trade-off, allowing for rapid training iterations and fast real-time inference on a web backend without needing massive GPU clusters.

**5. Why did you use transfer learning?**
Training a deep neural network from scratch requires massive datasets and compute resources. Transfer learning allows us to leverage weights pre-trained on ImageNet, meaning the model already understands basic visual features like edges, textures, and shapes. We only need to teach it to look for specific deepfake artifacts.

**6. Why is face detection a necessary first step?**
Deepfake manipulation (like face swapping) is localized to the facial region. If we feed the entire image into the model, the background introduces massive noise and the model might learn spurious correlations (e.g., learning the background of a specific dataset). Cropping to the face forces the model to focus on the relevant forensic artifacts.

**7. Why do you add a margin when cropping the face?**
Deepfake artifacts often occur at the boundaries where the fake face is blended into the original head or background. If we crop too tightly around the facial features (eyes, nose, mouth), we might cut off these crucial blending boundaries at the jawline or forehead. The margin ensures these edges are included in the analysis.

**8. Why use Grad-CAM in your project?**
Deep learning models are often considered "black boxes." Grad-CAM provides visual explainability by generating a heatmap that shows which parts of the image strongly influenced the model's prediction. This builds trust with the user and helps debug whether the model is looking at valid artifacts or just background noise.

**9. Why use PyTorch instead of TensorFlow/Keras?**
PyTorch offers a more Pythonic, dynamic computational graph, which makes debugging and custom implementations (like specific data loaders or Grad-CAM hooks) much easier. It has also become the dominant framework in the research community, providing access to the latest pre-trained models.

**10. Why use FastAPI for the backend?**
FastAPI is a modern, fast web framework for building APIs with Python. It provides automatic interactive documentation (Swagger UI), asynchronous capabilities for handling concurrent requests, and is highly performant, making it ideal for serving machine learning models.

**11. Why use vanilla JavaScript for the frontend?**
For a focused demonstration of the machine learning backend, a vanilla HTML/JS frontend minimizes complexity and dependencies. It proves that the API is standard and easily consumable without requiring heavy frontend frameworks like React or Angular, keeping the project lightweight and easy to understand.


## DATA Section

**1. What is Celeb-DF v2?**
Celeb-DF v2 is a high-quality deepfake video dataset. Unlike earlier datasets that had obvious artifacts, Celeb-DF v2 was created using improved synthesis methods, resulting in fewer noticeable visual flaws. This makes it a challenging and realistic dataset for training robust detection models.

**2. If Celeb-DF is a video dataset, how do you use it for an image project?**
We implement a data pipeline that extracts individual frames from the videos at a specified frame rate. We treat each extracted frame as an independent image for the purpose of training our spatial 2D CNN model.

**3. What is data leakage in the context of frame extraction?**
Data leakage occurs when highly similar data appears in both the training and testing sets. If we randomly split frames, two nearly identical frames from the same video could end up in different sets. The model would just memorize the specific person or background, artificially inflating test scores without actually learning to detect deepfakes.

**4. How do you solve the data leakage problem?**
We use "group-aware splitting" (or video-level splitting). We ensure that all extracted frames from a specific source video are kept together and assigned entirely to either the train, validation, or test set. An identity seen in training will not be seen in testing.

**5. How do you handle multiple faces in a single frame?**
When MTCNN detects multiple faces, our strategy is to select the face with the largest bounding box area. We assume the largest face is the primary subject of the video and the most likely target of manipulation.

**6. What happens if MTCNN fails to detect a face?**
If MTCNN fails to find a face, the frame is usually skipped during dataset preparation. During live inference, the API will return an error to the user stating that no face was detected, as the model requires a cropped face to function correctly.

**7. Why do you normalize the images before feeding them to the model?**
Normalization scales the pixel values to have a specific mean and standard deviation. We use the ImageNet normalization statistics because our EfficientNet backbone was pre-trained on ImageNet. For the pre-trained weights to activate correctly, the input data distribution must match what the model saw during its original training.

**8. Why be conservative with data augmentation?**
In standard image classification (like identifying dogs), heavy augmentation helps generalization. However, deepfakes are identified by subtle, high-frequency digital artifacts and noise patterns. Aggressive augmentation like heavy blurring or compression might destroy the very artifacts the model is trying to learn.


## MODEL Section

**1. What is a CNN and why use it here?**
A Convolutional Neural Network (CNN) is a deep learning architecture specifically designed to process grid-like data such as images. It uses convolutional layers to automatically and adaptively learn spatial hierarchies of features, making it ideal for finding the visual artifacts present in deepfakes.

**2. Explain the basic structure of EfficientNet.**
EfficientNet consists of a stem, multiple stages of Mobile Inverted Bottleneck Convolution (MBConv) blocks, and a head. It scales up baseline networks by uniformly increasing the depth (number of layers), width (number of channels), and resolution of the input image using a fixed set of scaling coefficients.

**3. What is a feature extractor vs. classification head?**
The feature extractor consists of the convolutional layers (the backbone) that learn to identify patterns, edges, and textures from the image. The classification head is the final fully connected layer that takes those learned features and maps them to a specific class probability (e.g., REAL or DEEPFAKE).

**4. Why does your model output a single logit?**
Since this is a binary classification problem, we only need one output node. The single logit represents the raw, unnormalized score for the positive class (DEEPFAKE=1). A negative logit pushes towards REAL, and a positive logit pushes towards DEEPFAKE.

**5. What is the role of the Sigmoid function?**
The Sigmoid function maps the raw logit output from the network into a value strictly between 0 and 1. This allows us to interpret the output as a probability score, where >0.5 typically indicates a DEEPFAKE.

**6. Why use BCEWithLogitsLoss instead of BCELoss?**
BCEWithLogitsLoss combines the Sigmoid activation and the Binary Cross Entropy loss into a single operation. This combined approach is mathematically more stable and less prone to numerical underflow/overflow issues compared to applying Sigmoid and then BCELoss separately.

**7. What is overfitting and how do you prevent it?**
Overfitting occurs when a model learns the training data too well, memorizing noise rather than generalizing to unseen data. We prevent it by using techniques like data augmentation, weight decay in the optimizer, and most importantly, early stopping based on validation loss.

**8. What is early stopping?**
Early stopping is a regularization technique where we monitor the validation loss during training. If the validation loss stops improving (or starts increasing) for a set number of epochs (patience), we halt training and restore the model weights from the epoch with the best validation score.

**9. Explain your two-stage fine-tuning process.**
In Stage 1, we freeze the pre-trained EfficientNet backbone and only train our new classification head with a higher learning rate. In Stage 2, we unfreeze the backbone and train the entire network with a much lower learning rate to adapt the convolutional filters to deepfake artifacts without destroying the pre-trained weights.

**10. Why use a smaller learning rate during fine-tuning?**
The backbone already possesses valuable pre-trained weights from ImageNet. If we use a high learning rate, large gradient updates might drastically alter and destroy these useful weights (catastrophic forgetting). A small learning rate allows for gentle adjustments.

**11. What is the AdamW optimizer?**
AdamW is a variation of the Adam optimizer that decouples weight decay from the gradient update. It provides the fast convergence benefits of Adam while handling regularization (weight decay) more correctly, leading to better generalization on unseen data.

**12. Why do we assign DEEPFAKE=1 and REAL=0?**
It is standard convention in binary classification to assign the "positive" class (the thing we are trying to detect) to 1. Since our primary goal is detecting deepfakes, DEEPFAKE is class 1.

**13. What does the learning rate scheduler do?**
We use ReduceLROnPlateau, which monitors a metric (like validation loss). If the loss stops decreasing for a few epochs, the scheduler reduces the learning rate. This allows the model to take smaller steps and settle into a finer minimum in the loss landscape.


## EVALUATION Section

**1. Define Accuracy in this context.**
Accuracy is the ratio of correctly predicted images (both true Reals and true Deepfakes) to the total number of images evaluated.

**2. Define Precision and why it matters.**
Precision is the ratio of correctly predicted Deepfakes to all images the model *claimed* were Deepfakes. High precision means when the model flags a Deepfake, it is highly likely to actually be one (few false positives).

**3. Define Recall and why it matters.**
Recall is the ratio of correctly predicted Deepfakes to all *actual* Deepfakes in the dataset. High recall means the model successfully finds most of the deepfakes (few false negatives).

**4. What is the F1-Score?**
The F1-Score is the harmonic mean of Precision and Recall. It provides a single metric that balances both concerns, which is especially useful if the dataset has a class imbalance (e.g., more Reals than Deepfakes).

**5. What is the ROC-AUC?**
ROC-AUC stands for Area Under the Receiver Operating Characteristic Curve. It measures the model's ability to distinguish between the REAL and DEEPFAKE classes across all possible probability thresholds. An AUC of 0.5 is random guessing; 1.0 is perfect separation.

**6. Why is ROC-AUC threshold-independent?**
Metrics like Accuracy and F1 require you to pick a specific cutoff (like 0.5) to decide if something is a deepfake. ROC-AUC evaluates the raw probabilities and ranks them, making it a robust metric of the model's overall discriminatory power regardless of the chosen threshold.

**7. In the context of your project, what is a False Positive?**
Since DEEPFAKE is 1, a False Positive is when the model predicts an image is a DEEPFAKE, but it is actually a REAL image.

**8. What is a False Negative?**
A False Negative is when the model predicts an image is REAL, but it is actually a DEEPFAKE.

**9. How do you handle class imbalance in evaluation?**
We look at metrics like PR-AUC (Precision-Recall Area Under Curve) and F1-score rather than just Accuracy. Accuracy can be misleading if 90% of the data is REAL; a model that always guesses REAL would get 90% accuracy but 0% recall for Deepfakes.

**10. How do you select the optimal threshold?**
Instead of blindly using 0.5, we evaluate the model on the Validation set and calculate the F1 score across various thresholds (e.g., from 0.1 to 0.9). The threshold that yields the highest F1 score on the Validation set is selected.

**11. Why select the threshold on the validation set and not the test set?**
Selecting the threshold on the test set is a form of data leakage or "peeking." The test set must remain completely unseen to provide an unbiased estimate of real-world performance. All tuning, including threshold selection, must be done on the validation set.

**12. What does a Confusion Matrix show?**
It is a 2x2 table showing True Positives (correctly flagged deepfakes), True Negatives (correctly flagged reals), False Positives, and False Negatives. It provides a clear visual breakdown of where the model is making errors.

**13. Which is worse: False Positives or False Negatives?**
It depends on the application. For an automated social media filter, False Positives (flagging a real video as fake) might cause user outrage. For a security clearance system, False Negatives (letting a deepfake through) are more dangerous.

**14. What is the difference between Validation and Test sets?**
The Validation set is used during training to monitor progress, tune hyperparameters, and trigger early stopping. The Test set is held out completely until the very end of the project to provide a final, unbiased evaluation of the fully trained model.


## EXPLAINABILITY Section

**1. What is Grad-CAM?**
Gradient-weighted Class Activation Mapping (Grad-CAM) is a technique used to produce visual explanations for decisions from CNN-based models. It creates a heatmap showing the regions of the input image that were most important for the model's prediction.

**2. How does Grad-CAM work conceptually?**
It works by looking at the gradients (the derivative of the loss) flowing back into the final convolutional layer. These gradients tell us which feature maps in that layer have the strongest positive influence on the final class score.

**3. Why do we target the convolutional feature maps?**
Convolutional layers retain spatial information (unlike fully connected layers, which flatten the data). The final convolutional layer provides the best compromise between high-level semantic information (what the object is) and detailed spatial information (where it is).

**4. Why is a ReLU activation applied to the heatmap in Grad-CAM?**
ReLU removes negative values. We apply it because we are only interested in the features that have a *positive* influence on the specific class we are querying. We want to know what makes the image a deepfake, not what makes it *less* of a deepfake.

**5. What does the heatmap actually mean?**
Hotter colors (like red or yellow) indicate regions of the image that strongly activated the model toward its final prediction. Cooler colors (blue) indicate regions that were less important.

**6. What can Grad-CAM NOT tell us?**
It cannot tell us the underlying logical reasoning. For example, it might highlight the jawline, indicating it found artifacts there, but it cannot verbally explain *what* the artifact is (e.g., "the pixels are blurred"). It is an interpretability tool, not a reasoning engine.

**7. Why is the model's output or Grad-CAM not proof of manipulation?**
Machine learning models are probabilistic and can be fooled by noise, unusual lighting, or adversarial attacks. Grad-CAM shows what the model focused on to make its guess, but since the model itself can be wrong, the heatmap is just an explanation of a potentially flawed decision, not cryptographic forensic proof.


## DEPLOYMENT Section

**1. How does the frontend communicate with the FastAPI backend?**
The frontend uses the JavaScript `fetch` API to send an asynchronous HTTP POST request to the backend's `/api/predict` endpoint. The image file is packaged inside a `FormData` object.

**2. How is the model loaded in the FastAPI app?**
The model is instantiated and its state dictionary (`best_model.pth`) is loaded into memory during the application's startup phase. We wrap this in a singleton InferenceService class.

**3. Why load the model only once at startup?**
Loading a PyTorch model from disk into RAM (or VRAM) takes time. If we loaded the model for every single user request, the API would be incredibly slow. Loading it once keeps it in memory, allowing for near-instantaneous inference on incoming requests.

**4. How do you handle file uploads in FastAPI?**
FastAPI provides an `UploadFile` class. When the endpoint receives a file, FastAPI handles the memory management (spooling large files to disk if necessary) and gives us an interface to read the byte stream directly into an OpenCV or PIL image format.

**5. How do you return the Grad-CAM image to the frontend?**
Instead of saving the generated Grad-CAM heatmap to the server's hard drive and serving a URL, we encode the image array into a JPEG format in memory, convert it to a Base64 string, and return that string inside the JSON payload. The frontend can render a Base64 string directly in an `<img>` tag.

**6. What kind of error handling is necessary in the API?**
The API must handle scenarios where the user uploads a non-image file, a corrupted image, or an image where MTCNN cannot detect a face. In these cases, it should catch the exception and return a clean HTTP 400 Bad Request error with a descriptive JSON message.

**7. How do you ensure the model runs efficiently during deployment if no GPU is available?**
The `device.py` utility automatically detects the environment. If no CUDA GPU is found, it falls back to the CPU. Since we use EfficientNet-B0, inference on a standard CPU takes only a fraction of a second per image, making it perfectly viable for a web backend without expensive GPU hosting.
