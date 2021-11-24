import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from load_data import load_data, load_data_through_image_data_generator, load_val_data_through_image_data_generator
import seaborn as sns

# 73 % accuracy, slow as fuck


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

def build_model(input_shape):

    # create a model
    model = keras.Sequential()

    # 1st conv layer
    model.add(keras.layers.Conv2D(
        256, (3, 3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.3))

    # 2nd conv layer
    model.add(keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.3))

    # 3rd conv layer
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.3))

    # flatten output and feed into dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(32))

    # output layer
    model.add(keras.layers.Dense(7, activation='softmax'))

    return model

def predict(model, X, y):

    X = X[np.newaxis, ...]

    # prediction = [[0,1, 0.2, ...]]
    prediction = model.predict(X)  # X -> (1, 130 ,13 ,1)

    predicted_index = np.argmax(prediction, axis=1)  # [3]

    print("Predicted Index : {}, Actual Index : {}".format(predicted_index, y))

def plot_history(history, name='cnn'):

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

    fig.savefig(f'./plots/{name}_history.png')
    plt.show()

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

if __name__ == "__main__":
    # create train, validation and test sets
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(
        0.25, 0.2)
    
    del X_train
    del X_test
    del y_train
    del y_test

    train_data = load_data_through_image_data_generator()
    #val_data = load_val_data_through_image_data_generator()
    test_data = load_val_data_through_image_data_generator()

    # build the CNN net
    input_shape = (32, 32, 3)
    model = build_model(input_shape)
    model.summary()

    # compile network
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    keras.utils.plot_model(
        model, to_file='./plots/cnn_architecture.png', show_shapes=True)

    # print(X_train.shape,y_train.shape)
    # train CNN
    #history = model.fit(X_train,y_train,validation_data=(X_validation,y_validation), batch_size=16, epochs=50)
    #del X_train
    #del y_train
    history = model.fit(train_data,
                        validation_data=(X_validation,y_validation),
                        epochs=50)

    plot_history(history)

    # evaluate CNN on test set
    test_err, test_acc = model.evaluate(test_data, verbose=1)
    print('Test accuracy:', test_acc)
    print('Test error:', test_err)

    # make predictions on sample
    #X = X_test[100]
    #y = y_test[100]

    #predict(model, X, y)

    # plot confusion matrix
    y_pred = model.predict(test_data)
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(test_data.classes, axis=1)
    matrix = plot_confusion_matrix(y_true, y_pred)

    # plot fractional incorrect misclassifications
    plot_fractional_incorrect_misclassifications(matrix)

    model.save('cnn_model')
