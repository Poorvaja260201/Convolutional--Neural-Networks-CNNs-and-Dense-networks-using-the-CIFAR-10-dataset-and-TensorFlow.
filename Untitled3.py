#!/usr/bin/env python
# coding: utf-8

# In[22]:


# Load the CIFAR-10 dataset
from tensorflow.keras.datasets import cifar10
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[23]:


# Finding the size of the data
print('Training data:', train_images.shape, train_labels.shape)
print('Test data:', test_images.shape, train_labels.shape) 


#Converting labels single dimension
train_labels=np.ravel(train_labels)
print(train_labels.shape)

test_labels=np.ravel(test_labels)
print(test_labels.shape)


# In[24]:


#Assigning class names to output variables
class_names=['Airplane','Automobile','Bird','Cat','Deer','Dog','Frog','Horse','Ship','Truck']


# In[25]:


#Normalization of the data
train_images = train_images / 255.0
test_images = test_images/255.0
print(train_images.min(),train_images.max())


# In[26]:


#Implementing baseline cnn model
import tensorflow as tf
from tensorflow.keras import layers,models

#To prevent kernal from crashing
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

#define the CNN model
model = tf.keras.models.Sequential([
tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
tf.keras.layers.MaxPooling2D(2, 2),
tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
tf.keras.layers.MaxPooling2D(2,2),
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(128, activation='relu'),
tf.keras.layers.Dense(10, activation='softmax')
])


# In[27]:


#Compiling the model and printing the summary of the baseline
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()


# In[96]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Create an instance of ImageDataGenerator with data augmentation options
datagen = ImageDataGenerator(
    rescale=1./255,                # Rescale pixel values to [0, 1]
    rotation_range=40,            # Random rotation within -40 to 40 degrees
    width_shift_range=0.2,        # Random horizontal shift within 20% of the image width
    height_shift_range=0.2,       # Random vertical shift within 20% of the image height
    horizontal_flip=True          # Random horizontal flipping
)

# Example usage for loading and augmenting images from a directory
train_generator = datagen.flow_from_directory(
    r"'C:\Users\Poorvaja\.keras\datasets\cifar-10-batches-py.tar.gz',
    target_size=(32, 32),       # Resize the images to a specific target size
    batch_size=32,
    class_mode='categorical'    # Use 'categorical' for multi-class classification
)


# In[98]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import numpy as np

# Create an instance of ImageDataGenerator with data augmentation options
datagen = ImageDataGenerator(
    rescale=1./255,                # Rescale pixel values to [0, 1]
    rotation_range=40,            # Random rotation within -40 to 40 degrees
    width_shift_range=0.2,        # Random horizontal shift within 20% of the image width
    height_shift_range=0.2,       # Random vertical shift within 20% of the image height
    horizontal_flip=True          # Random horizontal flipping
)

# Load an example image for augmentation
img_path = '.jpg'  # Replace with the path to your image
img = image.load_img(img_path)
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

# Generate augmented images
i = 0
for batch in datagen.flow(x, batch_size=1):
    # Display the augmented image (you can save or process it as needed)
    augmented_image = image.array_to_img(batch[0])
    augmented_image.save(f'augmented_image_{i}.jpg')
    i += 1
    if i >= 20:  # Generate 20 augmented images
        break


# In[99]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[100]:


datagen = ImageDataGenerator(
    rescale=1./255,                # Rescale pixel values to [0, 1]
    rotation_range=40,            # Random rotation within -40 to 40 degrees
    width_shift_range=0.2,        # Random horizontal shift within 20% of the image width
    height_shift_range=0.2,       # Random vertical shift within 20% of the image height
    horizontal_flip=True          # Random horizontal flipping
)


# In[101]:


train_data_generator = datagen.flow_from_directory(
    'path_to_training_data_directory',
    target_size=(150, 150),     # Resize the images to a specific target size
    batch_size=32,
    class_mode='categorical'    # Use 'categorical' for multi-class classification
)



# In[102]:


model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))


# In[103]:


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[104]:


model.fit(train_data_generator, epochs=10)


# In[105]:


# Define the baseline model
baseline_model = ...

# Compile the model with the appropriate loss and metrics
baseline_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the baseline model
history_baseline = baseline_model.fit(
    train_data_generator,  # Use your training data generator or loaded data
    epochs=10,
    validation_data=validation_data_generator  # If applicable
)

# Evaluate the baseline model on the test data
test_accuracy_baseline = baseline_model.evaluate(test_data_generator)[1]

# Print and plot the training and validation accuracy and loss
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

# Plot training & validation accuracy
plt.subplot(1, 2, 1)
plt.plot(history_baseline.history['accuracy'], label='Train')
plt.plot(history_baseline.history['val_accuracy'], label='Validation')
plt.title('Baseline Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('


# In[106]:


# Define the augmented model
augmented_model = ...

# Compile the augmented model with the same loss and metrics as the baseline model
augmented_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the augmented model using the same data loading procedure as the baseline
history_augmented = augmented_model.fit(
    train_data_generator,  # Use your training data generator or loaded data
    epochs=10,
    validation_data=validation_data_generator  # If applicable
)

# Evaluate the augmented model on the test data
test_accuracy_augmented = augmented_model.evaluate(test_data_generator)[1]

# Print and plot the training and validation accuracy and loss
plt.figure(figsize=(12, 4))

# Plot training & validation accuracy
plt.subplot(1, 2, 1)
plt.plot(history_augmented.history['accuracy'], label='Train')
plt.plot(history_augmented.history['val_accuracy'], label='Validation')
plt.title('Augmented Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot training & validation loss
plt.subplot(1, 2, 2)
plt.plot(history_augmented.history['loss'], label='Train')
plt.plot(history_augmented.history['val_loss'], label='Validation')
plt.title('Augmented Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

print(f"Test Accuracy (Augmented Model): {test_accuracy_augmented}")


# In[107]:


import matplotlib.pyplot as plt
import numpy as np

# Assuming you have a list of original images and their labels
original_images = ...
labels = ...

# Display a few original images and their augmented versions
num_samples = 5  # Number of examples to display

for i in range(num_samples):
    original_image = original_images[i]
    label = labels[i]

    # Create a subplot with 1 row and 2 columns
    plt.subplot(num_samples, 2, 2 * i + 1)
    plt.imshow(original_image)
    plt.title(f'Original - Label: {label}')
    plt.axis('off')

    # Generate augmented versions of the image
    augmented_images = []
    num_augmentations = 3  # Number of augmented versions to generate
    for j in range(num_augmentations):
        augmented_image = datagen.random_transform(original_image)
        augmented_images.append(augmented_image)

    # Display augmented images
    for j,


# In[ ]:




