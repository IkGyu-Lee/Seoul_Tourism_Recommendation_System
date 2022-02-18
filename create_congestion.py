import os
import sys
import torch
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from model_congestion.MF import MatrixFactorization

# Dataset for creating congestion
class Sample_Dataset(Dataset):
    def __init__(self, df):
        super(Sample_Dataset, self).__init__()
        self.df = df
        self.destination, self.time, self.dayofweek, self.month, self.day = self.change_tensor()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.month[idx], self.day[idx], self.dayofweek[idx], self.time[idx], self.destination[idx]

    def change_tensor(self):
        destination = torch.tensor(list(self.df['destination']))
        time = torch.tensor(list(self.df['time']))
        dayofweek = torch.tensor(list(self.df['dayofweek']))
        month = torch.tensor(list(self.df['month']))
        day = torch.tensor(list(self.df['day']))
        return destination, time, dayofweek, month, day

############################################################################################################
use_pretrain = 'False'
model_name = 'MF'
epochs = '20'
sample_df = pd.read_csv('../../Preprocessing/Datasets_v3.1/sample_for_congestion.csv')
# sample_df = pd.read_csv('../dataset/sample_for_congestion.csv')

# device select
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# PATH of Pretained Model about congestion1, congestion2
FOLDER_PATH = 'pretrain_model'
MODEL_PATH_CONGESTION1 = os.path.join(FOLDER_PATH, f'0_{use_pretrain}_{model_name}_{epochs}_512_48_congestion_1.pth')
MODEL_PATH_CONGESTION2 = os.path.join(FOLDER_PATH, f'0_{use_pretrain}_{model_name}_{epochs}_512_48_congestion_2.pth')

# the number of feature
num_destination = sample_df['destination'].max() + 1
num_time = sample_df['time'].max() + 1
num_dayofweek = sample_df['dayofweek'].max() + 1
num_month = sample_df['month'].max() + 1
num_day = sample_df['day'].max() + 1

# Creating Model
model_c1 = MatrixFactorization(num_dayofweek=num_dayofweek,
                               num_time=num_time,
                               num_month=num_month,
                               num_day=num_day,
                               num_destination=num_destination,
                               num_factor=48)
model_c2 = MatrixFactorization(num_dayofweek=num_dayofweek,
                               num_time=num_time,
                               num_month=num_month,
                               num_day=num_day,
                               num_destination=num_destination,
                               num_factor=48)
model_c1.to(device)
model_c2.to(device)

# Load Pretrained Model
model_c1.load_state_dict(torch.load(MODEL_PATH_CONGESTION1, map_location=device))
model_c2.load_state_dict(torch.load(MODEL_PATH_CONGESTION2, map_location=device))

# Dataset
sample_dataset = Sample_Dataset(sample_df)

# Dataloader
sample_dataloader = DataLoader(dataset=sample_dataset, batch_size=32, shuffle=False, drop_last=False)

# tensor for collecting predicted congestion1 & 2
c1_tensor, c2_tensor = torch.tensor([]), torch.tensor([])
c1_tensor, c2_tensor = c1_tensor.to(device), c2_tensor.to(device)

# Predict c1, c2
for month, day, dayofweek, time, destination in sample_dataloader:
    month, day, dayofweek, time, destination = month.to(device), day.to(device), dayofweek.to(device),\
                                               time.to(device), destination.to(device)
    # Predict congestion 1 & 2
    pred_c1 = model_c1(dayofweek, time, month, day, destination)
    pred_c2 = model_c2(dayofweek, time, month, day, destination)

    pred_c1 = pred_c1.view(-1)
    pred_c2 = pred_c2.view(-1)

    c1_tensor = torch.cat([c1_tensor, pred_c1], dim=0)
    c2_tensor = torch.cat([c2_tensor, pred_c2], dim=0)

c1_list = c1_tensor.view(-1).tolist()
c2_list = c2_tensor.view(-1).tolist()
#print(c1_list[0:100])

sample_df['congestion_1'] = c1_list
sample_df['congestion_2'] = c2_list
#print(sample_df.iloc[0:100,5])

pretrain_dir = 'dataset'
if not os.path.exists(pretrain_dir):
    os.mkdir(pretrain_dir)


#print(sample_df.describe())
sample_df.to_csv('dataset/congestion_1_2.csv', index=False)
