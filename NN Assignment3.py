#!/usr/bin/env python
# coding: utf-8

# In[72]:


import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
# Load the dataset
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
from tensorflow.keras.models import load_model
from tensorflow.keras import models


# In[73]:


train_images.shape,test_images.shape,train_labels.shape,test_labels.shape


# In[74]:


# Preprocess data
train_images = train_images/255.0
test_images = test_images/255.0


# In[75]:


import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


# In[76]:


# Define and compile the CNN model
# define the model
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()


# In[77]:


# Train the CNN model
history = model.fit(train_images, train_labels, epochs=6, batch_size=32, validation_split=0.2)


# In[78]:


# Save the model's architecture and weights
model.save("cnn_model.h5")


# In[79]:


# Visualize Model Training History
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[80]:


# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[81]:


# Step 2: Evaluate CNN Model
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print("Test Loss:", test_loss)
print("Test Accuracy:",test_accuracy)


# In[83]:


predictions = model.predict (test_images)
print(predictions.shape, test_images.shape)
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(test_labels, np.argmax(predictions, axis=1))
print(cm)
cr = classification_report(test_labels, np.argmax(predictions, axis=1))
print(cr)



# In[85]:


layer_outputs = [layer.output for layer in model.layers]

activation_model = tf.keras.models.Model(inputs = model.input, outputs = layer_outputs)

list = [i for i in range(len(train_images)) if train_labels[i] == 9]
print(len(list))


print(list[0:10])
train_labels[list[0:10]]
np.argmax(model.predict(train_images[list[0:10]]), axis=1)

f, ax = plt.subplots(10,5)
f.set_figheight(15)
f.set_figwidth(15)
plt.setp(ax, xticks=[], yticks=[])

for idx, k in enumerate(list[0:10]):
    ax[idx,0].imshow(train_images[k], cmap='gray')
    ax[idx,0].set_xlabel('image:'+str(k))
    for lay_ind in range(0,4):
        f = activation_model.predict(train_images[k].reshape(1, 32, 32, 3))[lay_ind]
        ax[idx,lay_ind+1].imshow(f[0, : , :, 1],cmap='gray')


# In[97]:


model_dense = Sequential([
    Flatten(input_shape=(32, 32, 3)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model_dense.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the Dense model
history_dense = model_dense.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# Save the model's architecture and weights
model_dense.save("dense_model.h5")


# In[98]:


print(boot_lists[0:10])
train_labels[list[0:10]]


# In[89]:


import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
# Define input layer. This specifies that model will receive input data of shape (784,)
input_tensor = Input(shape=(784,))
# Build the rest of the model using the functional API
x = Dense(128, activation='relu')(input_tensor)
output_tensor = Dense(10, activation='softmax')(x)
# Instantiate the model
model = Model(inputs=input_tensor, outputs=output_tensor)
model.summary()


# In[90]:


import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
# Define a simple model
input_tensor = Input(shape=(32,))
x = Dense(64, activation='relu')(input_tensor)
output_tensor = Dense(10, activation='softmax')(x)
model = Model(inputs=input_tensor, outputs=output_tensor)
# Accessing model.output
print(model.output)


# In[91]:


intermediate_model = Model(inputs=model.input,
outputs=model.layers[1].output)
print(intermediate_model.output)


# In[92]:


# Defining a simple model
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model


# In[93]:


# Define a simple model
input_tensor = Input(shape=(32,))
x = Dense(64, activation='relu')(input_tensor)
output_tensor = Dense(10, activation='softmax')(x)
model = Model(inputs=input_tensor, outputs=output_tensor)


# In[94]:


weights = model.get_weights()


# In[95]:


for w in weights:
 print(w.shape)


# In[96]:


# Get weights of the first Dense layer
first_layer_weights = model.layers[1].get_weights()
# Print the shapes of the weight matrices of the first layer
for w in first_layer_weights:
 print(w.shape)


# In[54]:


import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
# Define a simple model
input_tensor = Input(shape=(32,))
x = Dense(64, activation='relu')(input_tensor)
output_tensor = Dense(10, activation='softmax')(x)
model = Model(inputs=input_tensor, outputs=output_tensor)


# In[55]:


# Get weights of the first Dense layer
first_layer = model.layers[1]
first_layer_weights, first_layer_biases = first_layer.get_weights()
# Modify weights and biases
first_layer_weights[:] = 1.0
first_layer_biases[:] = 0.0
# Set weights back to the first Dense layer
first_layer.set_weights([first_layer_weights, first_layer_biases])



# In[101]:


# Get all model weights
all_weights = model.get_weights()
# if we want to set all weights to ones and biases to zeros
for i in range(len(all_weights)):
 if len(all_weights[i].shape) == 1: # this is a bias vector
   all_weights[i][:] = 0.0
 else: # this is a weight matrix
    all_weights[i][:] = 1.0
# Set modified weights back to the model
model.set_weights(all_weights)


# In[102]:


#modify
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
# Define a simple model
input_tensor = Input(shape=(32,))
x = Dense(64, activation='relu')(input_tensor)
x = Dropout(0.5)(x)
output_tensor = Dense(10, activation='softmax')(x)
model = Model(inputs=input_tensor, outputs=output_tensor)


# In[103]:


# Print names of all layers in the model
for layer in model.layers:
 print(layer.name)


# In[104]:


dense_layer = model.layers[1]
print(dense_layer.name)


# In[105]:


dense_layer.trainable = False
dense_layer_weights = dense_layer.get_weights()
print(dense_layer_weights[0].shape) # Weight matrix shape
print(dense_layer_weights[1].shape) # Bias vector shape


# In[106]:


dense_layer_output = dense_layer.output


# In[107]:


# Assume input shape for images of size 28x28
input_tensor = Input(shape=(32, 32))
x = Flatten()(input_tensor)
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)
output_tensor = Dense(10, activation='softmax')(x)
model = Model(inputs=input_tensor, outputs=output_tensor)


# In[109]:


model_cnn_modified = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),  # Add an additional convolutional layer
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model_cnn_modified.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the modified CNN model
history_cnn_modified = model_cnn_modified.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# Save the model's architecture and weights
model_cnn_modified.save("cnn_model_modified.h5")


# In[110]:


test_loss_cnn_modified, test_accuracy_cnn_modified = model_cnn_modified.evaluate(test_images, test_labels)
print(f"Modified CNN Model Test accuracy: {test_accuracy_cnn_modified}")


# In[ ]:




