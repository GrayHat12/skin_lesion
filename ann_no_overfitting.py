from sklearn.model_selection import train_test_split
from load_data import load_data, load_data_through_image_data_generator
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

# faster than cnn to train, 41% accuracy

def plot_history(history):
    
    fig, axs = plt.subplots(2)
    
    # create accuracy sublpot
    axs[0].plot(history.history['accuracy'], label="train accuracy")
    axs[0].plot(history.history['val_accuracy'], label="test accuracy")
    axs[0].set_ylabel('Accuracy')
    axs[0].legend(loc="lower right")
    axs[0].set_title('Accuracy evaluation')
    
    # create loss sublpot
    axs[1].plot(history.history['loss'], label="train loss")
    axs[1].plot(history.history['val_loss'], label="test loss")
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].legend(loc="upper right")
    axs[1].set_title('Loss evaluation')
    
    fig.savefig('./plots/ann_no_overfitting_history.png')
    plt.show()

def prepare_datasets(test_size,validation_size):
    # load data
    X, y = load_data()

    X = X / 255

    y = keras.utils.to_categorical(y, num_classes=7)
    
    # create train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    
    # create train/validation split
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)
    
    return X_train, X_validation, X_test, y_train, y_validation, y_test

def plot_confusion_matrix(y_true,y_pred):
    matrix = confusion_matrix(y_true,y_pred)

    fig, ax = plt.subplots()

    sns.set(font_scale=1.6)
    sns.heatmap(matrix, annot=True, ax=ax, linewidths=5)

    fig.savefig('./plots/ann_no_overfitting_confusion_matrix.png')
    fig.show()
    return matrix

def plot_fractional_incorrect_misclassifications(confusion_matrix):
    fig, ax = plt.subplots()
    incorr_fraction = 1 - np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)
    ax.bar(np.arange(7), incorr_fraction)
    ax.set_xlabel('True Label')
    ax.set_ylabel('Fraction of incorrect predictions')

    fig.savefig('./plots/ann_no_overfitting_fractional_incorrect_misclassifications.png')
    fig.show()

if __name__ == "__main__":

    # create train, validation and test sets
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.25, 0.2)

    # build network topology
    model = keras.Sequential([

        # input layer
        keras.layers.Flatten(input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3])),

        # 1st dense layer
        keras.layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),

        # 2nd dense layer
        keras.layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),

        # 3rd dense layer
        keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),

        # output layer
        keras.layers.Dense(7, activation='softmax')
    ])

    # compile model
    optimiser = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimiser,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()
    keras.utils.plot_model(model, to_file='./plots/ann_no_overfitting_architecture.png', show_shapes=True)

    # train model
    #history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=16, epochs=50)

    del X_train
    del y_train
    history = model.fit(load_data_through_image_data_generator(),
                                  validation_data=(X_validation, y_validation),
                                  epochs=50)

    # plot accuracy and error as a function of the epochs
    plot_history(history)

    # plot confusion matrix
    y_pred = model.predict(X_validation)
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_validation, axis=1)
    matrix = plot_confusion_matrix(y_true, y_pred)

    # plot fractional incorrect misclassifications
    plot_fractional_incorrect_misclassifications(matrix)

    model.save('ann_no_overfitting_model')