# The model for the skin cancer classifier

# accuracy - 97
# top2 accuracy - 99
# top3 accuracy - 99

# validation acc - 83
# validation top2 acc - 94
# validation top3 acc - 98

# so fucking slow to train that I had to sleep, very accurate

import logging
logging.basicConfig(level=logging.ERROR, # show only error msgs,
                    format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# Import the libraries
from keras.metrics import top_k_categorical_accuracy
import numpy as np
#import keras
from tensorflow import keras
#from keras import backend as K
from keras.layers.core import Dense, Dropout
from keras.optimizers import adam_v2
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
#from keras.applications.mobilenet import MobileNet
import onnx
from onnx2keras import onnx_to_keras

from cnn import plot_history, plot_confusion_matrix, plot_fractional_incorrect_misclassifications

# Check if GPU is available
# K.tensorflow_backend._get_available_gpus()

# model path
model_path = 'tp_models/zfnet512/model.onnx'

# Load ONNX model
onnx_model = onnx.load(model_path)

# Call the converter (input - is the main model input name, can be different for your model)
zfnet_model = onnx_to_keras(onnx_model, ['gpu_0/data_0'], verbose=True, change_ordering=True)

# The paths for the training and validation images
train_path = 'base_dir/train_dir'
valid_path = 'base_dir/val_dir'

# Declare a few useful values
num_train_samples = 9013
num_val_samples = 1002
train_batch_size = 16
val_batch_size = 16
image_size = 224

# Declare how many steps are needed in an iteration
train_steps = np.ceil(num_train_samples / train_batch_size)
val_steps = np.ceil(num_val_samples / val_batch_size)

# Set up generators
train_batches = ImageDataGenerator(
    preprocessing_function=keras.applications.mobilenet.preprocess_input,data_format="channels_last").flow_from_directory(
    train_path,
    target_size=(image_size, image_size),
    batch_size=train_batch_size)

valid_batches = ImageDataGenerator(
    preprocessing_function=keras.applications.mobilenet.preprocess_input,data_format="channels_last").flow_from_directory(
    valid_path,
    target_size=(image_size, image_size),
    batch_size=val_batch_size)

test_batches = ImageDataGenerator(
    preprocessing_function=keras.applications.mobilenet.preprocess_input,data_format="channels_last").flow_from_directory(
    valid_path,
    target_size=(image_size, image_size),
    batch_size=val_batch_size,
    shuffle=False)

# See a summary of the layers in the model
zfnet_model.summary()

#exit(0)

# Modify the model
# Exclude the last 5 layers of the model
x = zfnet_model.layers[-6].output
# Add a dropout and dense layer for predictions
x = Dropout(0.25)(x)

# flatten output and feed into dense layer
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(32)(x)

predictions = Dense(7, activation='softmax')(x)

# Create a new model with the new outputs
model = Model(inputs=zfnet_model.input, outputs=predictions)

# See a summary of the new layers in the model
model.summary()

# Freeze the weights of the layers that we aren't training (training the last 10)
for layer in model.layers[:-10]:
    layer.trainable = False

# Train the model
# Define Top2 and Top3 Accuracy


def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)


def top_2_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)


# Compile the model

model.compile(adam_v2.Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=[
    top_2_accuracy, top_3_accuracy, "accuracy"])

keras.utils.plot_model(
    model, to_file='./plots/zfnet_architecture.png', show_shapes=True)

# Add weights to make the model more sensitive to melanoma
class_weights = {
    0: 1.0,  # akiec
    1: 1.0,  # bcc
    2: 1.0,  # bkl
    3: 1.0,  # df
    4: 3.0,  # mel
    5: 1.0,  # nv
    6: 1.0,  # vasc
}

# Declare the filepath for the saved model
filepath = "zfnet"

# Declare a checkpoint to save the best version of the model
checkpoint = ModelCheckpoint(filepath, monitor='val_top_3_accuracy', verbose=1,
                             save_best_only=True, mode='max')

# Reduce the learning rate as the learning stagnates
reduce_lr = ReduceLROnPlateau(monitor='val_top_3_accuracy', factor=0.5, patience=2,
                              verbose=1, mode='max', min_lr=0.00001)

callbacks_list = [checkpoint, reduce_lr]

# train batches test
x, y = next(train_batches)

print(x.shape)
print(y.shape)

# Fit the model
history = model.fit(train_batches,
                    steps_per_epoch=train_steps,
                    class_weight=class_weights,
                    validation_data=valid_batches,
                    validation_steps=val_steps,
                    epochs=50,
                    verbose=1,
                    callbacks=callbacks_list)

plot_history(history,name='zfnet')

# Evaluate the model
# Evaluation of the last epoch
val_loss, val_cat_acc, val_top_2_acc, val_top_3_acc = \
    model.evaluate_generator(test_batches, steps=val_steps)

print('val_loss:', val_loss)
print('val_cat_acc:', val_cat_acc)
print('val_top_2_acc:', val_top_2_acc)
print('val_top_3_acc:', val_top_3_acc)

# Evaluation of the best epoch
model.load_weights(filepath)

val_loss, val_cat_acc, val_top_2_acc, val_top_3_acc = \
    model.evaluate(test_batches, steps=val_steps)

print('val_loss:', val_loss)
print('val_cat_acc:', val_cat_acc)
print('val_top_2_acc:', val_top_2_acc)
print('val_top_3_acc:', val_top_3_acc)

# Create a confusion matrix of the test images
test_labels = test_batches.classes

# Make predictions
predictions = model.predict(test_batches, steps=val_steps, verbose=1)

# Declare a function for plotting the confusion matrix

#cm = confusion_matrix(test_labels, predictions.argmax(axis=1))

cm_plot_labels = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

#plot_confusion_matrix(cm, cm_plot_labels)

matrix = plot_confusion_matrix(test_labels, np.argmax(predictions, axis=1),name='zfnet')

plot_fractional_incorrect_misclassifications(matrix,name='zfnet')
