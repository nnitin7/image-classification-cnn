import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import pickle

# Load dataset
(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()

# Normalize the pixel values
X_train, X_test = X_train / 255.0, X_test / 255.0

# Build CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Save the model
model.save('cnn_model.h5')
