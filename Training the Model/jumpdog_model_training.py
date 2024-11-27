# ************ pip install ******************
# install following library:

# pip install numpy tensorflow matplotlib tensorflowjs
# (might take some time , you can start from now)



# Import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, regularizers
import matplotlib.pyplot as plt
import os
import random

# Set a seed for reproducibility
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)

# Assume the training data is in 'data.npy' (replace with your actual file)
# Each sample is a tuple: (board_state, label)
# board_state: 5x5 numpy array
# label: scalar value between -100 and 100

# Load the data
data = np.load('data.npy', allow_pickle=True)

# Shuffle the data
np.random.shuffle(data)

# Split features and labels
X = np.array([sample[0] for sample in data])
y = np.array([sample[1] for sample in data])

# Normalize labels to -1 to 1 for numerical stability during training
y = y / 100.0

# Reshape X to add channel dimension
X = X.reshape(-1, 5, 5, 1)

# Split into training and validation sets (80% training, 20% validation)
split_index = int(0.8 * len(X))
X_train, X_val = X[:split_index], X[split_index:]
y_train, y_val = y[:split_index], y[split_index:]

# Save the training and validation data for later reuse
np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)
np.save('X_val.npy', X_val)
np.save('y_val.npy', y_val)

# Build the CNN model
model = keras.Sequential()

# Input Layer and Convolutional Layer 1
model.add(layers.Conv2D(
    filters=32,                # Use 32 filters
    kernel_size=(3, 3),
    padding='same',
    activation='relu',
    input_shape=(5, 5, 1)
))

# Convolutional Layer 2
model.add(layers.Conv2D(
    filters=64,                # Use 64 filters
    kernel_size=(3, 3),
    padding='same',
    activation='relu'
))

# Flatten Layer
model.add(layers.Flatten())

# Dense Layer 1
model.add(layers.Dense(
    units=128,
    activation='relu',
    # kernel_regularizer=regularizers.l2(0.001)  # Optional L2 Regularization
))

# Optional Dropout Layer for Regularization
# model.add(layers.Dropout(0.5))

# Dense Layer 2
model.add(layers.Dense(
    units=64,
    activation='relu',
    # kernel_regularizer=regularizers.l2(0.001)  # Optional L2 Regularization
))

# Optional Dropout Layer for Regularization
# model.add(layers.Dropout(0.5))

# Output Layer
model.add(layers.Dense(
    units=1,
    activation='linear'
))

# Compile the model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),  # Default learning rate
    loss='mean_squared_error',
    metrics=['mae']
)

# Optional Learning Rate Scheduler
# lr_scheduler = keras.callbacks.ReduceLROnPlateau(
#     monitor='val_loss',
#     factor=0.1,
#     patience=5,
#     verbose=1
# )

# Optional Early Stopping
# early_stopping = keras.callbacks.EarlyStopping(
#     monitor='val_loss',
#     patience=5,
#     restore_best_weights=True,
#     verbose=1
# )

# Prepare for live plotting of loss function
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.line1, = self.ax.plot([], [], 'b-', label='Training Loss')
        self.line2, = self.ax.plot([], [], 'r-', label='Validation Loss')
        self.ax.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.line1.set_data(range(len(self.losses)), self.losses)
        self.line2.set_data(range(len(self.val_losses)), self.val_losses)
        self.ax.relim()
        self.ax.autoscale_view()
        plt.draw()
        plt.pause(0.01)

loss_history = LossHistory()

# Training the model
history = model.fit(
    X_train,
    y_train,
    epochs=50,                         # Train for 50 epochs
    batch_size=32,                     # Use batch size of 32
    validation_data=(X_val, y_val),
    callbacks=[
        loss_history,
        # lr_scheduler,                # Optional Learning Rate Scheduler
        # early_stopping               # Optional Early Stopping
    ]
)

# After training, validate the model
val_loss, val_mae = model.evaluate(X_val, y_val, verbose=1)
print(f'Validation Loss: {val_loss}')
print(f'Validation MAE: {val_mae}')

# Save the model in TensorFlow.js format
# First, save the model in the TensorFlow SavedModel format
model.save('saved_model/my_model')

# Convert the model to TensorFlow.js format
# Requires the TensorFlow.js converter to be installed: pip install tensorflowjs
os.system('tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model --signature_name=serving_default --saved_model_tags=serve saved_model/my_model tfjs_model')

# Save the training history plot
plt.ioff()
plt.savefig('training_history.png')

# Save evaluation data
with open('evaluation.txt', 'w') as f:
    f.write(f'Validation Loss: {val_loss}\n')
    f.write(f'Validation MAE: {val_mae}\n')

# Comments for Computational Efficiency
# To reduce computational load, you can:
# - Reduce the number of filters in Conv2D layers from 32 and 64 to 16 and 32 respectively.
# - Reduce the number of units in dense layers from 128 and 64 to 64 and 32 respectively.
# - Remove one of the convolutional or dense layers if acceptable.

# Example modifications for computational efficiency:
# model.add(layers.Conv2D(
#     filters=16,  # Reduced from 32 to 16
#     kernel_size=(3, 3),
#     padding='same',
#     activation='relu',
#     input_shape=(5, 5, 1)
# ))
# ...
# model.add(layers.Dense(
#     units=64,   # Reduced from 128 to 64
#     activation='relu',
# ))

# Note: Reducing model complexity may impact accuracy.

