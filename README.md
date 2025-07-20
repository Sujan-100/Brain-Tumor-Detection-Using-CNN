# Brain-Tumor-Detection-Using-CNN
MRI Brain Tumor Detection using CNN Built and trained a Convolutional Neural Network (CNN) to classify MRI images into tumor types. Used image preprocessing, model evaluation, and accuracy comparison with AlexNet. Achieved high accuracy, demonstrating potential for aiding early diagnosis.


Detailed Explaination of the project:

Project Title: Brain Tumor Detection and Classification using Convolutional Neural Networks (CNN)
Project Overview
Brain tumors are abnormal cell growths inside the brain that can severely affect human health. Early and accurate detection is crucial for effective treatment and improved survival rates. Traditionally, radiologists analyze MRI (Magnetic Resonance Imaging) scans to detect tumors. However, manual analysis is time-consuming and subject to human error. In this project, we developed a deep learning-based system using Convolutional Neural Networks (CNN) to automatically detect and classify brain tumors from MRI images.

Objective
The main goal of the project is to:
Classify MRI brain scans into categories such as glioma, meningioma, pituitary tumor, or no tumor.
Build a model that can assist doctors and radiologists by offering fast and accurate results.
Compare the performance of a custom-built CNN model with AlexNet, a pre-defined deep learning architecture.

Tools and Technologies Used:
Programming Language: Python
Libraries: TensorFlow, Keras, NumPy, OpenCV, Matplotlib, scikit-learn
Environment: Google Colab / Jupyter Notebook
Model Architectures: Custom CNN and AlexNet

Dataset Description
The dataset is obtained from Kaggle and contains labeled MRI scan images. Each image belongs to one of the following classes:
Glioma Tumor
Meningioma Tumor
Pituitary Tumor
No Tumor
Images are grayscale and have varying sizes, so preprocessing is required to standardize them.

Data Preprocessing
Before feeding the data to the neural network, we performed several preprocessing steps:
Resizing: All images were resized to 128x128 pixels to maintain uniformity.
Normalization: Pixel values were scaled to a range between 0 and 1.
Grayscale Processing: Images were converted to grayscale if not already.
Label Encoding: Categorical labels were encoded using one-hot encoding.
Train-Test Split: Data was split into training and testing sets with an 80–20 ratio using stratification to maintain class balance.

Custom CNN Model Architecture
We built a custom CNN using the following layers:
Conv2D Layer (32 filters, 3x3 kernel) – detects features like edges and textures.
MaxPooling2D (2x2) – reduces the size of the feature maps.
Conv2D Layer (64 filters, 3x3 kernel) – detects more complex patterns.
MaxPooling2D – further downsampling.
Flatten Layer – converts 2D data to 1D.
Dense Layer (128 neurons) – fully connected layer to learn high-level features.
Dropout (0.3) – randomly drops some neurons to avoid overfitting.
Output Dense Layer with Softmax Activation – for multi-class classification.

Training the Model
Loss Function: Categorical Crossentropy (used for multi-class classification)
Optimizer: Adam (efficient and adaptive optimizer)
Metrics: Accuracy
Epochs: 10
Batch Size: 32
Validation Split: 20% of training data was used for validation during training

Model Evaluation
After training, the model was evaluated using the test dataset. Metrics such as test accuracy and loss were calculated. Additionally, training and validation accuracy/loss graphs were plotted to monitor performance and overfitting.

Comparison with AlexNet
To compare performance, we also implemented AlexNet, a well-known deep learning model designed for image classification. AlexNet was modified slightly to work with grayscale MRI images. Both models were trained and evaluated on the same dataset to observe differences in:
Accuracy
Training time
Generalization performance

Results and Conclusion
Our custom CNN model performed well with high test accuracy, indicating effective learning and generalization.
AlexNet also showed strong performance and served as a benchmark.
The deep learning approach can be a valuable tool for radiologists, helping with early tumor detection and reducing diagnosis time.
This project demonstrates the power of deep learning in medical image analysis and its potential to make healthcare smarter and faster.
