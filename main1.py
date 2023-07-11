import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
train_fname = 'pop_vs_classical_train.csv'
df_train = pd.read_csv(train_fname)

# Store data in dictionary
data = {'pop': df_train[df_train['label'] == 'pop'],
        'classical': df_train[df_train['label'] == 'classical']}

# Define features
features = ['spectral_centroid_mean', 'harmony_mean', 'tempo']

# Apply Seaborn style
sns.set()

# Create subplots
fig, axes = plt.subplots(nrows=len(features), ncols=1, figsize=(10, 15))

# Loop over features and create plots
for i, feature in enumerate(features):
    for label, df in data.items():
        sns.kdeplot(df[feature], label=label, ax=axes[i], linewidth=2)
    axes[i].set_xlabel('')
    axes[i].set_ylabel('Density')
    axes[i].set_title('Probability Density Function - {}'.format(feature))
    axes[i].legend()
    sns.despine(ax=axes[i], top=True, right=True)

# Adjust spacing between subplots
plt.subplots_adjust(hspace=0.5)

plt.show()
