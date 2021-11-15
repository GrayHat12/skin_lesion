import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
import os
from glob import glob
import numpy as np
from PIL import Image
import json

root_data_folder = './HAM10000'

image_dir = f'{root_data_folder}/reorganized'

skin_df = pd.read_csv(f'{root_data_folder}/HAM10000_metadata.csv')

IMAGE_SIZE = 32

label_encoder = LabelEncoder()
label_encoder.fit(skin_df['dx'])

print(list(label_encoder.classes_))

skin_df['label'] = label_encoder.transform(skin_df['dx'])
print(skin_df.sample(10))


print(skin_df['label'].value_counts())

df_0 = skin_df[skin_df['label'] == 0]
df_1 = skin_df[skin_df['label'] == 1]
df_2 = skin_df[skin_df['label'] == 2]
df_3 = skin_df[skin_df['label'] == 3]
df_4 = skin_df[skin_df['label'] == 4]
df_5 = skin_df[skin_df['label'] == 5]
df_6 = skin_df[skin_df['label'] == 6]

n_samples = 700

df_0_balanced = resample(
    df_0, replace=True, n_samples=n_samples, random_state=42)
df_1_balanced = resample(
    df_1, replace=True, n_samples=n_samples, random_state=42)
df_2_balanced = resample(
    df_2, replace=True, n_samples=n_samples, random_state=42)
df_3_balanced = resample(
    df_3, replace=True, n_samples=n_samples, random_state=42)
df_4_balanced = resample(
    df_4, replace=True, n_samples=n_samples, random_state=42)
df_5_balanced = resample(
    df_5, replace=True, n_samples=n_samples, random_state=42)
df_6_balanced = resample(
    df_6, replace=True, n_samples=n_samples, random_state=42)

skin_df_balanced = pd.concat([df_0_balanced,
                              df_1_balanced,
                              df_2_balanced,
                              df_3_balanced,
                              df_4_balanced,
                              df_5_balanced,
                              df_6_balanced])

print(skin_df_balanced['label'].value_counts())


image_path = {
    os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(root_data_folder, '*', '*.jpg'))
}

skin_df_balanced['path'] = skin_df['image_id'].map(image_path.get)

skin_json_balanced = json.loads(skin_df_balanced.to_json(orient='records'))

print(skin_json_balanced[0])

for i in range(len(skin_json_balanced)):
    skin_json_balanced[i]['image'] = np.asarray(Image.open(skin_json_balanced[i]['path']).resize((IMAGE_SIZE, IMAGE_SIZE))).tolist()

with open(f'{root_data_folder}/balanced_data.json', 'w') as f:
    json.dump(skin_json_balanced, f)

#skin_df_balanced['image'] = skin_df_balanced['path'].map(lambda x: np.asarray(Image.open(x).resize((IMAGE_SIZE, IMAGE_SIZE))))

#print(skin_df_balanced.sample(10))

#skin_df_balanced.to_csv(f'{root_data_folder}/balanced_data.csv', index=False)