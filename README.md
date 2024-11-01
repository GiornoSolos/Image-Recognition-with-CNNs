# Image-Recognition-with-CNNs
This project implements convolutional neural networks (CNNs) for image classification. It uses transfer learning with a pre-trained ResNet18 model and a custom CNN model to distinguish between image classes (tomato, cherry, and strawberries) in a dataset with varied photo quality. The project achieved a 92% accuracy rate on unseen data.

Features:
Data Augmentation: The dataset was preprocessed with random rotations, horizontal flips, and color jitter to improve model robustness.
Image Quality Control: Code was included to detect blurry images and outliers based on brightness, which were then enhanced with contrast adjustments or sharpening filters.
Models Implemented:
Multi-layer Perceptron (MLP): For comparison, a simple MLP was trained on flattened image data.
Custom CNN: A custom CNN model fine-tuned from a pre-trained ResNet18 model, achieving the best results with an accuracy of 92%.

Requirements:
Python 3.7+
PyTorch
torch-vision
OpenCV
PIL
Matplotlib

Can be applied to three variations of fruit applied to the test directory.
Project made for COMP309 Victoria University of Wellington.
