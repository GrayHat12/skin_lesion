from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
from load_data import load_data
from keras.models import load_model


def prepare_datasets(test_size, validation_size):
    # load data
    X, y = load_data()

    X = X / 255

    y = keras.utils.to_categorical(y, num_classes=7)

    # create train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size)

    # create train/validation split
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train, y_train, test_size=validation_size)

    return X_train, X_validation, X_test, y_train, y_validation, y_test


def plot_confusion_matrix(y_true, y_pred, name='cnn'):
    matrix = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots()

    sns.set(font_scale=1.6)
    sns.heatmap(matrix, annot=True, ax=ax, linewidths=5)

    fig.savefig(f'./plots/{name}_confusion_matrix.png')
    fig.show()
    return matrix


def plot_fractional_incorrect_misclassifications(confusion_matrix, name='cnn'):
    fig, ax = plt.subplots()
    incorr_fraction = 1 - np.diag(confusion_matrix) / \
        np.sum(confusion_matrix, axis=1)
    ax.bar(np.arange(7), incorr_fraction)
    ax.set_xlabel('True Label')
    ax.set_ylabel('Fraction of incorrect predictions')

    fig.savefig(f'./plots/{name}_fractional_incorrect_misclassifications.png')
    fig.show()


X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(
    0.25, 0.2)

model = load_model('cnn_model')

model.summary()

# plot confusion matrix
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)
matrix = plot_confusion_matrix(y_true, y_pred)

# plot fractional incorrect misclassifications
plot_fractional_incorrect_misclassifications(matrix)
