import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# 1. Load and preprocess the MNIST dataset
print("Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape to (samples, height, width, channels) and normalize pixels to [0,1]
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test  = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# Convert labels to one-hot encoding (10 classes: 0-9)
y_train = to_categorical(y_train, 10)
y_test  = to_categorical(y_test, 10)

print(f"Training samples: {x_train.shape[0]}")
print(f"Test samples: {x_test.shape[0]}")

# 2. Build a simple CNN model
print("\nBuilding CNN model...")
model = keras.Sequential([
    # Convolutional layer 1: learns basic features (edges, lines)
    layers.Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)),
    layers.MaxPooling2D(pool_size=(2, 2)),

    # Convolutional layer 2: learns more complex patterns
    layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),

    # Flatten and fully connected layers for classification
    layers.Flatten(),
    layers.Dropout(0.5),  # Prevents overfitting
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax")  # Output: probabilities for 10 digits
])

model.summary()

# 3. Compile the model
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# 4. Train the model
print("\nTraining the model...")
history = model.fit(
    x_train, y_train,
    batch_size=128,
    epochs=10,
    validation_split=0.1,  # Use 10% of training data for validation
    verbose=1
)

# 5. Evaluate on the test set
print("\nEvaluating on test set...")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
print(f"\nTest accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"Test loss: {test_loss:.4f}")

# 6. Visualize original images with predictions
print("\nDisplaying 9 original MNIST test images with predictions:")
predictions = model.predict(x_test)

plt.figure(figsize=(12, 12))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_test[i].squeeze(), cmap="gray")  # Original 28×28 grayscale image
    pred_label = np.argmax(predictions[i])
    true_label = np.argmax(y_test[i])
    color = "green" if pred_label == true_label else "red"
    plt.title(f"Pred: {pred_label}\nTrue: {true_label}", fontsize=14, color=color)
    plt.axis("off")

plt.suptitle("Original MNIST Test Images\n(Handwritten digits 0-9)", fontsize=18)
plt.tight_layout()
plt.show()
