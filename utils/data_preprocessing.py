# Import the os module to work with file paths and directories
import os

# Import the torch library for PyTorch model development
import torch

# Import datasets and transforms modules from torchvision for loading and transforming image data
from torchvision import datasets, transforms

# Import TensorFlow library for building and training deep learning models
import tensorflow as tf

# Define a function to get PyTorch dataloaders for training and testing
def get_pytorch_dataloaders(data_dir, batch_size=32):
    # Define a sequence of image transformations: resize to 150x150 and convert to tensor
    transform = transforms.Compose([
        transforms.Resize((150, 150)),  # Resize images to 150x150 pixels
        transforms.ToTensor()           # Convert images to PyTorch tensors
    ])
    
    # Load training images from 'train' subdirectory using the defined transformations
    train_data = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
    
    # Load testing images from 'test' subdirectory using the same transformations
    test_data = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform)

    # Create PyTorch DataLoader objects for both training and testing datasets
    dataloaders = {
        'train': torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True),  # Shuffle training data
        'test': torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)    # Do not shuffle test data
    }
    
    # Return the dictionary containing both dataloaders
    return dataloaders

# Define a function to get TensorFlow datasets for training and validation
def get_tf_datasets(data_dir, batch_size=32):
    # Load training dataset from 'train' subdirectory, resizing images and batching
    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, 'train'),   # Path to training data
        image_size=(150, 150),             # Resize images to 150x150
        batch_size=batch_size              # Set batch size
    )
    
    # Load validation dataset from 'test' subdirectory, using the same settings
    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, 'test'),    # Path to test/validation data
        image_size=(150, 150),             # Resize images to 150x150
        batch_size=batch_size              # Set batch size
    )
    
    # Return the training and validation datasets
    return train_ds, val_ds
