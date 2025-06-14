import torch                              # Import PyTorch library for tensor operations and deep learning
import torch.nn as nn                     # Import the neural network module from PyTorch
import torch.optim as optim               # Import optimizers module for training
from torchvision import models           # Import pretrained models from torchvision

class PyTorchCNN(nn.Module):              # Define a custom CNN class that inherits from nn.Module
    def __init__(self):                   # Initialization method for the class
        super(PyTorchCNN, self).__init__()  # Initialize the parent nn.Module class
        self.model = models.resnet18(pretrained=True)  # Load ResNet18 pretrained on ImageNet
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)  # Replace last FC layer for 2 classes

    def forward(self, x):                 # Forward pass method to define computation
        return self.model(x)              # Pass input through the ResNet model and return output

def train_model(model, dataloaders, device, epochs=5, lr=0.001):  # Function to train the model
    model.to(device)                     # Move model to specified device (CPU or GPU)
    criterion = nn.CrossEntropyLoss()   # Define loss function for classification
    optimizer = optim.Adam(model.parameters(), lr=lr)  # Use Adam optimizer with learning rate lr

    for epoch in range(epochs):          # Loop over number of epochs
        print(f"Epoch {epoch+1}/{epochs}")  # Print current epoch number
        model.train()                    # Set model to training mode
        total_loss = 0                   # Initialize loss accumulator
        correct = 0                     # Initialize correct predictions counter
        total = 0                       # Initialize total samples counter

        for inputs, labels in dataloaders['train']:  # Loop over batches in training data
            inputs, labels = inputs.to(device), labels.to(device)  # Move batch data to device
            optimizer.zero_grad()       # Clear gradients from previous step

            outputs = model(inputs)     # Forward pass: compute predictions
            loss = criterion(outputs, labels)  # Compute loss comparing output and true labels
            loss.backward()             # Backpropagation to compute gradients
            optimizer.step()            # Update model weights based on gradients

            total_loss += loss.item()  # Accumulate batch loss
            _, preds = torch.max(outputs, 1)  # Get predicted class indices
            correct += (preds == labels).sum().item()  # Count correct predictions
            total += labels.size(0)    # Count total samples in this batch

        accuracy = correct / total      # Calculate accuracy for the epoch
        avg_loss = total_loss / len(dataloaders['train'])  # Calculate average loss per batch
        print(f"Train Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")  # Print loss and accuracy
