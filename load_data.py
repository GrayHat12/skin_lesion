import numpy as np
import json
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet import preprocess_input

def load_json(path):
    with open(path) as f:
        data = json.load(f)
    return data

def load_data_through_image_data_generator():
    print("Loading data...")
    root_data_dir = './HAM10000'
    datagen = ImageDataGenerator(
        rotation_range=20,
        rescale=1./255,
        #preprocessing_function=preprocess_input,
        zoom_range=0.5
    )

    data = datagen.flow_from_directory(
        directory=f'{root_data_dir}/reorganized',
        class_mode='categorical',
        batch_size=16,
        target_size=(32, 32),
        shuffle=True
    )

    return data

def load_val_data_through_image_data_generator():
    print("Loading data...")
    root_data_dir = './base_dir'
    datagen = ImageDataGenerator(
        rotation_range=20,
        rescale=1./255,
        #preprocessing_function=preprocess_input,
        zoom_range=0.5
    )

    data = datagen.flow_from_directory(
        directory=f'{root_data_dir}/val_dir',
        class_mode='categorical',
        batch_size=16,
        target_size=(32, 32)
    )

    return data

def load_data():
    print("Loading data...")
    root_data_dir = './HAM10000'
    skin_data = load_json(f'{root_data_dir}/balanced_data.json')
    X = np.asarray([data['image'] for data in skin_data])
    Y = np.asarray([data['label'] for data in skin_data])
    print("Data loaded.")
    return X,Y

if __name__ == "__main__":
    gen = load_data_through_image_data_generator()

    x,y = next(gen)
    print(x.shape)
    print(y.shape)
    print(x[0])