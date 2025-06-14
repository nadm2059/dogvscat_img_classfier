import tensorflow as tf  # Import TensorFlow library with alias tf

def build_tf_model():
    # Create a Sequential model (a linear stack of layers)
    model = tf.keras.models.Sequential([
        # Rescale input pixel values from [0, 255] to [0, 1], and specify input shape as 150x150 RGB images
        tf.keras.layers.Rescaling(1./255, input_shape=(150, 150, 3)),
        # Apply a 2D convolution layer with 32 filters, kernel size 3x3, using ReLU activation
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        # Apply max pooling to reduce spatial dimensions (default pool size 2x2)
        tf.keras.layers.MaxPooling2D(),
        # Another convolution layer with 64 filters and 3x3 kernel, ReLU activation
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        # Another max pooling layer to downsample the feature maps
        tf.keras.layers.MaxPooling2D(),
        # Flatten the 3D feature maps into 1D feature vector for the dense layers
        tf.keras.layers.Flatten(),
        # Fully connected dense layer with 64 units and ReLU activation
        tf.keras.layers.Dense(64, activation='relu'),
        # Output layer with 2 units (for 2 classes) and softmax activation for classification probabilities
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    return model  # Return the constructed model

def train_tf_model(model, train_ds, val_ds, epochs=5):
    # Compile the model specifying optimizer, loss function, and metrics to track
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',  # Use sparse categorical crossentropy for integer labels
                  metrics=['accuracy'])  # Track accuracy metric during training
    # Train the model on training dataset, validate on validation dataset, for given number of epochs
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    return history  # Return training history object containing loss and accuracy values
