#!/usr/bin/env python
# coding: utf-8

# In[10]:


import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os


class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        
        self.base_model = models.resnet18(pretrained=False)  
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.base_model(x)

# Function to load the model
def load_model(model_path, num_classes):
    model = CustomCNN(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set to evaluation mode
    return model

# Function to run predictions and calculate accuracy
def run_inference(model, test_loader, class_names):
    model.eval()
    results = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, image_paths in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            # Collect results
            for i in range(len(predicted)):
                results.append((image_paths[i], class_names[predicted[i].item()]))
            
            # Calculate accuracy
            labels = [os.path.basename(os.path.dirname(path)) for path in image_paths]
            labels = torch.tensor([class_names.index(label) for label in labels])
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    return results, accuracy

# Custom Dataset class to load images from the testdata directory
class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        return original_tuple[0], path  # add path to the output

# Main function to load the model, data, and run predictions
def main():
    model_path = "model.pth"
    test_data_dir = "testdata"
    num_classes = 3

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((300, 300)),   # Resize to match training
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize as in training
    ])

    # Load the test dataset
    test_dataset = ImageFolderWithPaths(root=test_data_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    class_names = test_dataset.classes  # Get class names

    # Load model
    model = load_model(model_path, num_classes)
    model.eval()

    # Run inference and calculate accuracy
    results, accuracy = run_inference(model, test_loader, class_names)

    
    # Print accuracy
    print(f"Overall Accuracy on Test Data: {accuracy:.2f}%")

if __name__ == "__main__":
    main()

