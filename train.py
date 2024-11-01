#!/usr/bin/env python
# coding: utf-8

# In[26]:


import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import os
import matplotlib.pyplot as plt
from PIL import Image
import cv2


data_dir = data_dir = r'C:\Users\Administrator\Downloads\train_data\train_data'  


# Define transformations (resize, normalize)
train_transform = transforms.Compose([
    transforms.Resize((300, 300)),   
    transforms.RandomHorizontalFlip(),  # Random horizontal flip
    transforms.RandomRotation(15),  # Random rotation within 15 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),           # Convert to tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize RGB values to [-1, 1]
])

# Load the dataset using ImageFolder
train_data = datasets.ImageFolder(root=data_dir, transform=train_transform)

# Use DataLoader to batch the data
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)


images, labels = next(iter(train_loader))
print(f"Image batch shape: {images.shape}")
print(f"Label batch shape: {labels.shape}")



# In[27]:


# Classes
classes = ['tomato', 'strawberry', 'cherry']

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

for i, cls in enumerate(classes):
    img_dir = os.path.join(data_dir, cls)
    
    #Validity Check
    valid_extensions = ('.jpg')
    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(valid_extensions)]
    
    
    if img_files:
        img_path = os.path.join(img_dir, img_files[0])
        img = Image.open(img_path)

        axs[i].imshow(img)
        axs[i].set_title(cls)
        axs[i].axis('off')
    else:
        print(f"No valid images found in {img_dir}")

plt.show()


# In[28]:


def resize_and_normalize(image, size=(300, 300)):
    resized = cv2.resize(image, size)
    normalized = resized / 255.0  # Scale pixel values
    return resized, normalized


# In[29]:


bad_image_counts = {cls: 0 for cls in classes}

# Iterate through each class folder
for cls in classes:
    img_dir = os.path.join(data_dir, cls)
    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith('.jpg')]
    
    for img_file in img_files:
        img_path = os.path.join(img_dir, img_file)
        image = cv2.imread(img_path)
        # Check size
        if image is not None and image.shape[:2] != (300, 300):
            print(f"Inconsistent size found: {cls} - {img_file}, Size: {image.shape}")
            bad_image_counts[cls] += 1  # Increment the counter for this class

# Print the total count of bad images for each class
for cls, count in bad_image_counts.items():
    print(f"Total inconsistent images in class '{cls}': {count}")


# In[ ]:





# In[30]:


import cv2
import numpy as np
#Based on image brightness
def is_outlier(image, threshold_low=20, threshold_high=230):
    avg_pixel_value = np.mean(image)
    return avg_pixel_value < threshold_low or avg_pixel_value >threshold_high


# In[31]:


def is_blurry(image, threshold=50):
    # Laplacian operator sensitive to sharp changes in the image
    # Therefore, less variance = more blurry
    variance = cv2.Laplacian(image, cv2.CV_64F).var() #outputs 64bit float point
    return variance < threshold


# In[32]:


outliers = {cls: [] for cls in classes}
blurry_images = {cls: [] for cls in classes}

# Iterate through each class folder
for cls in classes:
    img_dir = os.path.join(data_dir, cls)
    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(valid_extensions)]
    
    # Check each image for outliers and blurriness
    for idx, img_file in enumerate(img_files):
        img_path = os.path.join(img_dir, img_file)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            # Check if the image is blurry
            if is_blurry(image):
                blurry_images[cls].append((idx, img_path))
            # Check if the image is an outlier
            if is_outlier(image):
                outliers[cls].append((idx, img_path))

# Display blurry images
for cls, img_data in blurry_images.items():
    if img_data:
        print(f"Found {len(img_data)} blurry images in class '{cls}':")
        fig, axs = plt.subplots(1, min(len(img_data), 5), figsize=(15, 5))
        # If there's only one axis, convert axs to a list to maintain consistency
        if len(img_data) == 1:
            axs = [axs]
        for i, (index, img_path) in enumerate(img_data[:5]):
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axs[i].imshow(img_rgb)
            axs[i].set_title(f"{cls} - Index {index}")
            axs[i].axis('off')
        plt.show()
    else:
        print(f"No blurry images found in class '{cls}'.")

# Display outliers
for cls, img_data in outliers.items():
    if img_data:
        print(f"Found {len(img_data)} outliers in class '{cls}':")
        fig, axs = plt.subplots(1, min(len(img_data), 5), figsize=(15, 5))
        if len(img_data) == 1:
            axs = [axs]
        for i, (index, img_path) in enumerate(img_data[:5]):
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axs[i].imshow(img_rgb)
            axs[i].set_title(f"{cls} - Index {index}")
            axs[i].axis('off')
        plt.show()
    else:
        print(f"No outliers found in class '{cls}'.")


# In[33]:


def sharpen_image(image):
    # Define a sharpening kernel
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    # Apply the sharpening filter
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened


# In[34]:


for cls, img_data in blurry_images.items():
    if img_data:
        print(f"Sharpening {len(img_data)} blurry images in class '{cls}':")
        for index, img_path in img_data:
            # Read the image in color (not grayscale) for sharpening
            image = cv2.imread(img_path)
            
            if image is not None:
                # Apply the sharpening filter
                sharpened_image = sharpen_image(image)
                
                # Save the sharpened image back to the same path (overwrites original image)
                cv2.imwrite(img_path, sharpened_image)
                print(f"Sharpened image saved: {img_path}")
            else:
                print(f"Error reading image: {img_path}")
    else:
        print(f"No blurry images to sharpen in class '{cls}'.")


# In[35]:


def enhance_contrast(image):
    # Convert to grayscale if it's a color image
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply histogram equalization
    equalized = cv2.equalizeHist(gray)
    return equalized



# In[51]:


for cls, img_data in outliers.items():
    if img_data:
        print(f"Enhancing {len(img_data)} outliers in class '{cls}':")
        for index, img_path in img_data:
            # Read the image in color (not grayscale) for sharpening
            image = cv2.imread(img_path)
            
            if image is not None:
                # Apply the sharpening filter
                enhance_image = enhance_contrast(image)
                
                # Save the sharpened image back to the same path (overwrites the original image)
                cv2.imwrite(img_path, enhance_image)
                print(f"Enhanced image saved: {img_path}")
            else:
                print(f"Error reading image: {img_path}")
    else:
        print(f"No outliers to enhance in class '{cls}'.")


# In[37]:


val_transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)

# Split the dataset into training and validation sets (80% training, 20% validation)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])


val_dataset.dataset.transform = val_transform

# Create DataLoaders for training and validation
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# In[38]:


import torch.nn as nn
import torch.optim as optim

#Vanilla MLP model
class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleMLP, self).__init__()
        self.flatten = nn.Flatten()  # Flatten the input image
        self.fc1 = nn.Linear(input_size, hidden_size)  # First fully connected layer
        self.relu = nn.ReLU()  # ReLU activation function
        self.fc2 = nn.Linear(hidden_size, num_classes)  # Output layer

    def forward(self, x):
        x = self.flatten(x)  # Flatten the image for input into the MLP
        x = self.fc1(x)  # First layer
        x = self.relu(x)  # Activation
        x = self.fc2(x)  # Output layer
        return x


# In[41]:


# Initialize the model, loss function, and optimizer
input_size = 300 * 300 * 3  # 300x300 images with 3 channels (RGB)
hidden_size = 128  # Number of neurons in the hidden layer
num_classes = len(full_dataset.classes)

MLPmodel = SimpleMLP(input_size, hidden_size, num_classes) #Initialize the model
criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for classification
optimizer = optim.Adam(MLPmodel.parameters(), lr=0.001)  # Adam optimizer


# In[42]:


num_epochs = 10
for epoch in range(num_epochs):
    MLPmodel.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for images, labels in train_loader:
        outputs = MLPmodel(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

        # Calculate training accuracy
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
    
    train_accuracy = 100 * correct_train / total_train
    epoch_loss = running_loss / len(train_loader)

    # Evaluate on the validation set
    MLPmodel.eval()
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = MLPmodel(images)
            _, predicted = torch.max(outputs, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_accuracy = 100 * correct_val / total_val

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Validation Accuracy: {val_accuracy:.2f}%")


# In[43]:


print(MLPmodel)


# In[ ]:





# In[44]:


#CNN Model implementation

class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        # Load a pre-trained ResNet18 model
        self.base_model = models.resnet18(pretrained=True)
        # Freeze all layers except the last one
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Replace the final layer to match the number of classes
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)




# In[45]:


# Initialize the model
num_classes = len(full_dataset.classes)
model = CustomCNN(num_classes)

# Define loss function (CrossEntropy with label smoothing)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Define optimizer with weight decay (regularization) and an alternative optimizer (SGD with momentum)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)


# In[46]:


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Training the model
num_epochs = 10
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Calculate training accuracy
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
    
    train_accuracy = 100 * correct_train / total_train
    epoch_loss = running_loss / len(train_loader)

    # Evaluate on the validation set
    model.eval()
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_accuracy = 100 * correct_val / total_val

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Validation Accuracy: {val_accuracy:.2f}%")

# Print the model summary
print(model)


# In[52]:


model_path = 'model.pth'
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

