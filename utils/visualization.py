
# Import the pyplot module from matplotlib for plotting graphs
import matplotlib.pyplot as plt

# Import seaborn for advanced visualizations like heatmaps
import seaborn as sns

# Import confusion_matrix function to compute classification results
from sklearn.metrics import confusion_matrix

# Import NumPy for numerical operations (though not used directly here)
import numpy as np


# Define a function to plot training and validation accuracy over epochs
def plot_history(history):
    # Plot training accuracy from the model's history object
    plt.plot(history.history['accuracy'], label='Train Accuracy')

    # Plot validation accuracy from the model's history object
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')

    # Label the x-axis as "Epochs"
    plt.xlabel("Epochs")

    # Label the y-axis as "Accuracy"
    plt.ylabel("Accuracy")

    # Set the title of the plot
    plt.title("Training Accuracy")

    # Display the legend to show which line is which
    plt.legend()

    # Show the plot
    plt.show()


# Define a function to plot a confusion matrix using seaborn
def plot_confusion_matrix(y_true, y_pred, class_names):
    # Generate the confusion matrix from true and predicted labels
    cm = confusion_matrix(y_true, y_pred)

    # Create a heatmap of the confusion matrix with class names as labels
    sns.heatmap(cm, annot=True, fmt='d', 
                xticklabels=class_names, 
                yticklabels=class_names, 
                cmap='Blues')

    # Label the x-axis as "Predicted"
    plt.xlabel("Predicted")

    # Label the y-axis as "True"
    plt.ylabel("True")

    # Set the title of the plot
    plt.title("Confusion Matrix")

    # Show the heatmap
    plt.show()
