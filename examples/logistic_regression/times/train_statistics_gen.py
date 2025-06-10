#!/usr/bin/env python3
import os

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
train_files = [f for f in files_mnist if "train" in f]

# Create data frames.
train_data = []
for f in train_files:
    data = pd.read_csv(f, header=None)
    data['type'] = f.split('_')[0]
    data['iterations_per_boot'] = int(f.split('_')[2].split('boot')[1])
    train_data.append(data)

# Fuse dataframes.
train = pd.concat(train_data).reset_index(drop=True)

# Name columns.
train = train.rename(columns={0: 'iterations', 1:'total_time', 2:'bootstrapping_time', 3:'activation_function', 4:'NAG'})

# Modify columns data.
lambda_nag = lambda x: 'NAG' if x == 1 else 'No NAG'
train['NAG'] = train['NAG'].apply(lambda_nag)
train['total_time'] = train['total_time'] / 1000
train['bootstrapping_time'] = train['bootstrapping_time'] / 1000

# Compute relevant data.

samples_per_ciphertexts = 128
train['amortized_time_per_data_sample'] = train['total_time'] / samples_per_ciphertexts

averages_training = train.groupby(['type', 'NAG', 'iterations', 'iterations_per_boot']).agg('median')[['total_time', 'bootstrapping_time', 'amortized_time_per_data_sample']]
averages_training = averages_training.groupby(['type', 'NAG', 'iterations_per_boot']).agg('median')[['total_time', 'bootstrapping_time', 'amortized_time_per_data_sample']]
averages_training['speedup'] = averages_training.loc[averages_training['total_time'].idxmax()]['total_time'] / averages_training['total_time']

print(averages_training)
