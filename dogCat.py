from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import numpy as np

#load image
image = Image.open("cat_1.jpg")
#convert the image to numpy array
image_array = np.array(image)
#normalize the pixel values to the range [0,1]
image_normalized  = image_array / 255.0 
#Make sure the image has the correct data type(float32)
image_normalized = image_normalized.astype('float32')
#verify
print(image_normalized)


image_paths = ["cat_1.jpg","cat_2.jpg","cat_3.jpg","cat_4.jpg","cat_5.jpg","dog_1.jpg", "dog_2.jpg","dog_3.jpg", "dog_4.jpg","dog_5.jpg"]
labels = [0,0,0,0,0,1,1,1,1,1]


# Initialize empty lists to store images and labels
images = []
encoded_labels = []
# Load and preprocess each image
for image_path, label in zip(image_paths, labels):
    image = Image.open(image_path)
    image_array = np.array(image)
    image_normalized = image_array / 255.0
    image_normalized = image_normalized.astype('float32')

    images.append(image_normalized)
    encoded_labels.append(label)
# Convert lists to NumPy arrays after the loop
images = np.array(images)
encoded_labels = np.array(encoded_labels)

    # Verify the shapes of the arrays
print("Images shape:", images.shape)
print("Labels shape:", encoded_labels.shape)


model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(2, activation='softmax')  # Adjust the number of output neurons for your classes
])


model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',  # Use 'categorical_crossentropy' for one-hot encoded labels
            metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)