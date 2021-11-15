import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy import stats

root_data_folder = './HAM10000'

image_dir = f'{root_data_folder}/reorganized'

skin_df = pd.read_csv(f'{root_data_folder}/balanced_data.csv')

IMAGE_SIZE = 32

label_encoder = LabelEncoder()
label_encoder.fit(skin_df['dx'])

print(list(label_encoder.classes_))

skin_df['label'] = label_encoder.transform(skin_df['dx'])
print(skin_df.sample(10))

fig,axs = plt.subplots(1)
skin_df['dx'].value_counts().plot(kind='bar',ax=axs)
axs.set_ylabel('Count')
axs.set_title('Cell Type')

fig.savefig('./plots/cell_type.png')
fig.show()

fig,axs = plt.subplots(1)
skin_df['sex'].value_counts().plot(kind='bar',ax=axs)
axs.set_ylabel('Count')
axs.set_title('Sex')

fig.savefig('./plots/sex.png')
fig.show()

fig,axs = plt.subplots(1)
skin_df['localization'].value_counts().plot(kind='bar',ax=axs)
axs.set_ylabel('Count')
axs.set_title('Localization')

fig.savefig('./plots/localization.png')
fig.show()

fig,axs = plt.subplots(1)
sample_age = skin_df[pd.notnull(skin_df['age'])]
sns.distplot(sample_age['age'],ax=axs,fit=stats.norm,color='red')
axs.set_title('Age')

fig.savefig('./plots/age.png')
fig.show()

