import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Path to the dataset folder
male_folder = "./Gender Data/male"
female_folder = "./Gender Data/female"

# Function to load images from a folder (with support for JPG and BMP formats)
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        # Check if the file is a JPEG or BMP image
        if filename.lower().endswith(('.jpg', '.jpeg', '.bmp')):
            # Read image and convert to RGB
            image_path = os.path.join(folder, filename)
            try:
                image = cv2.imread(image_path)
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    # Resize image to desired dimensions (e.g., 48x48)
                    image = cv2.resize(image, (48, 48))
                    images.append(image)
                else:
                    print(f"Error: Unable to read image '{image_path}'")
            except Exception as e:
                print(f"Error: {e} while reading image '{image_path}'")
    return images

# Load male and female images
male_images = load_images_from_folder(male_folder)
female_images = load_images_from_folder(female_folder)

# Combine male and female images into a single dataset
images = np.array(male_images + female_images)

# Create labels for male (0) and female (1)
male_labels = np.zeros(len(male_images))
female_labels = np.ones(len(female_images))
labels = np.concatenate([male_labels, female_labels])

# Preprocess images (normalize pixel values)
images = images / 255.0

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.25, random_state=42)

# Define the gender detection model
def gender_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)  # Output layer for gender (binary classification)
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Create the gender detection model
model = gender_model(input_shape=(48, 48, 3))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define callbacks (e.g., model checkpoint to save the best model during training)
checkpoint_path = 'gender_pedestrian_model.keras'  # Change filepath extension to .keras
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks = [checkpoint]

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test), callbacks=callbacks)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Plot training history (e.g., loss and accuracy)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Predictions and metrics
y_pred = (model.predict(X_test) > 0.5).astype("int32")

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Male", "Female"]))

# Print confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Test the model on a sample image
def test_image(index, X_test, model):
    sample_image = X_test[index]
    plt.imshow(sample_image)
    plt.axis('off')
    plt.show()
    prediction = model.predict(np.expand_dims(sample_image, axis=0))[0][0]  # Extract scalar value from array
    gender = "Male" if prediction < 0.5 else "Female"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    print(f"Predicted Gender: {gender} with confidence {confidence:.2f}")

# Test a sample image
test_image(1, X_test, model)
