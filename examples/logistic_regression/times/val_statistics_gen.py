#!/usr/bin/env python3
import os

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Read all the files.
files = [f for f in os.listdir() if os.path.isfile(f)]
files = [f for f in files if f.endswith(".csv")]

# Divide by dataset.
files_mnist =  [f for f in files if "mnist" in f]

# Divide train and validation files.
train_files = [f for f in files_mnist if "validation" in f]

train_data = []
for f in train_files:
    data = pd.read_csv(f, header=None)
    data['platform'] = f.split('_')[0]
    data['iterations_per_boot'] = int(f.split('_')[2].split('boot')[1])
    train_data.append(data)

train = pd.concat(train_data).reset_index(drop=True)
train = train.drop(columns=[2, 3])
train = train.rename(columns={0: 'iterations', 1:'total_time', 4:'nag', 5:'precision'})
train['nag'] = train['nag'].map({0: 'No', 1: 'Si'})
train['total_time'] = train['total_time'] / 1000 # From us to ms
train['t_amortizado'] = train['total_time'] / 128

unique_combinations = train[['platform', 'iterations_per_boot']].drop_duplicates().values

for platform, ipb in unique_combinations:
    subset_train = train[(train['platform'] == platform) & (train['iterations_per_boot'] == ipb)]

    if not subset_train.empty:

        
        matplotlib.rcParams.update({'font.size': 16})
        plt.figure(figsize=(8, 6)) 

        sns.lineplot(
            data=subset_train,
            x='iterations',
            y='precision',
            hue='nag',
            palette='viridis', 
            marker='o',       
            errorbar=None     
        )

        plt.title(f'Plataforma: {platform}, Iteraciones por Boot: {ipb}')
        plt.xlabel('Iteraciones de Entrenamiento')
        plt.ylabel('Precisión (%)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(title='NAG')
        plt.tight_layout() 
        plt.savefig(str(platform) + str(ipb) + ".jpg")

average_time_per_platform = train.groupby('platform')[['total_time', 't_amortizado']].mean()
print(average_time_per_platform)