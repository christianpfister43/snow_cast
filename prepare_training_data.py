"""

@author: Christian Pfister
"""


import numpy as np
import pandas as pd
import random
import json
import cv2
import os

import pickle

from cell_asingment_training_data import asing_training_cells


#%% set global variables for training data preparation
data_dir = './data/dev'
_region = 'central rockies'         # 'sierras' or 'central rockies'

#%% reading data files


ground_features = pd.read_csv(f'{data_dir}/ground_measures_train_features.csv')
grid_size = 600
df_train_labels = pd.read_csv(f'{data_dir}/train_labels.csv')
df_train_labels.replace(np.nan, 0, inplace=True)
target = pd.read_csv(f'{data_dir}/submission_format.csv')

with open (f'{data_dir}/grid_cells.geojson') as f:
    geo = json.load(f)

asing_training_cells(_region,data_dir,target,df_train_labels,ground_features,geo, generate_new=False)

#%%

grid = np.zeros((grid_size,grid_size))

if os.path.isfile(f'{data_dir}/metatdata_newly_generated/cell_asignment_{grid_size}_{_region.replace(" ","_")}_train.pickle'):
    grid_to_cell_asignment = pickle.load(open(f'{data_dir}/metatdata_newly_generated/cell_asignment_{grid_size}_{_region.replace(" ","_")}_train.pickle','rb'))
else:
    grid_to_cell_asignment = pickle.load(open(f'{data_dir}/metadata_provided/cell_asignment_{grid_size}_{_region.replace(" ","_")}_train.pickle','rb'))
if os.path.isfile(f'{data_dir}/metatdata_newly_generated/station_asignment_{grid_size}_{_region.replace(" ","_")}.pickle'):
    grid_to_station_asignment = pickle.load(open(f'{data_dir}/metatdata_newly_generated/station_asignment_{grid_size}_{_region.replace(" ","_")}.pickle','rb'))
else:
        grid_to_station_asignment = pickle.load(open(f'{data_dir}/metadata_provided/station_asignment_{grid_size}_{_region.replace(" ","_")}.pickle','rb'))

for t in range(1,214):
    grid = np.zeros((grid_size,grid_size))
    for gi in range(grid_size):
        for gj in range(grid_size):
            # stat_id = grid_to_cell_asignment_g[gi,gj]
            station = grid_to_station_asignment[gi,gj]
            cellid_list = grid_to_cell_asignment[gi,gj]
            if not cellid_list == '':
                number_cells = int(len(cellid_list) / 36)
                for n in range(number_cells):
                    cellid = cellid_list[n*36:n*36+36]
                    if len(df_train_labels.loc[df_train_labels['cell_id']==cellid]) > 0:
                        grid[gi,gj] += np.array(df_train_labels.loc[df_train_labels['cell_id']==cellid])[0][t]
            if not station == '':
                grid[gi,gj] += np.array(ground_features[ground_features['Unnamed: 0'] == station])[0][t]
    cv2.imwrite(f'{data_dir}/train/{_region.replace(" ","_")}/swe_ground/image_{t}.png',grid)


#%%
grid = np.zeros((grid_size,grid_size))

if os.path.isfile(f'{data_dir}/metatdata_newly_generated/station_asignment_{grid_size}_{_region.replace(" ","_")}.pickle'):
    grid_to_station_asignment = pickle.load(open(f'{data_dir}/metatdata_newly_generated/station_asignment_{grid_size}_{_region.replace(" ","_")}.pickle','rb'))
else:
    grid_to_station_asignment = pickle.load(open(f'{data_dir}/metadata_provided/station_asignment_{grid_size}_{_region.replace(" ","_")}.pickle','rb'))

for t in range(1,214):
    grid = np.zeros((grid_size,grid_size))
    for gi in range(grid_size):
        for gj in range(grid_size):
            # stat_id = grid_to_cell_asignment_g[gi,gj]
            station = grid_to_station_asignment[gi,gj]
            if not station == '':
                grid[gi,gj] = np.array(ground_features[ground_features['Unnamed: 0'] == station])[0][t]
    cv2.imwrite(f'{data_dir}/train/{_region.replace(" ","_")}/only_ground/image_{t}.png',grid)

#######################
