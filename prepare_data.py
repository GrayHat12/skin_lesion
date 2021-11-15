import os
import pandas as pd
import shutil

import skimage as sk
import random

root_data_folder = './HAM10000'

data_dir = f'{root_data_folder}/all_images'

dest_dir = f'{root_data_folder}/reorganized'

skin_df = pd.read_csv(f'{root_data_folder}/HAM10000_metadata.csv')

print(skin_df['dx'].value_counts())

labels = skin_df['dx'].unique().tolist()

label_images = []

def random_rotation(image_array):
    # pick a random degree of rotation between 25% on the left and 25% on the right
    random_degree = random.uniform(-25, 25)
    
    return sk.transform.rotate(image_array, random_degree)

def random_noise(image_array):
    # add random noise to the image
    return sk.util.random_noise(image_array)

def horizontal_flip(image_array):
    # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !
    return image_array[:, ::-1]

for label in labels:
    os.makedirs(f'{dest_dir}/{label}/')

    sample = skin_df[skin_df['dx'] == label]['image_id']

    label_images.extend(sample)

    random.shuffle(label_images)

    difference = 1000 - len(label_images)

    while difference > 0:
        label_images.extend(random.sample(label_images, min(len(label_images),difference)))
        difference = 1000 - len(label_images)
    
    label_images = label_images[:1000]

    for i,id in enumerate(label_images):
        shutil.copyfile(f'{data_dir}/{id}.jpg', f'{dest_dir}/{label}/{str(i)}_{id}.jpg')
    
    label_images = []