# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 14:05:57 2022

@author: chpf1
"""



import numpy as np
import pandas as pd
import random
import json
import cv2
import glob
import tensorflow as tf
import pickle
from tensorflow.keras.models import  load_model
import os

from cell_asingment_eval_data import asing_evaluation_cells
from prepare_eval_data import prepare_evaluation_data


#%% setting global variables, reading in data
SEQ_LEN = 5
FUTURE_STEPS = 1
# _region = 'central rockies'         # 'sierras' or 'central rockies'
regions =['sierras','central rockies']
grid_size = 600
data_dir = './data/eval'


ground_features = pd.read_csv(f'{data_dir}/ground_measures_features.csv')
target = pd.read_csv(f'{data_dir}/submission_format.csv')
with open (f'{data_dir}/grid_cells.geojson') as f:
    geo = json.load(f)

target.rename(columns={'Unnamed: 0':'cell_id'}, inplace=True)   # cell_id name missing

prediction_delivery = pd.read_csv(f'{data_dir}/eval/prediction_snow_cast.csv')
prediction_delivery.rename(columns={'Unnamed: 0':'cell_id'}, inplace=True)

for _region in regions:
    print(f'running inference for region {_region}...')
    asing_evaluation_cells(_region,data_dir,target,ground_features,geo, generate_new = False)
    
    prepare_evaluation_data(_region, data_dir, target,ground_features,geo,grid_size = 600,SEQ_LEN=5)
    
    #%% 
    data_path=f'{data_dir}/eval/{_region.replace(" ","_")}'
    
    image_list = glob.glob(f'{data_path}/recent_5_eval/*.png')
    
    
    cols = ground_features.columns
    image_col_to_predict = 1
    for col in ground_features.columns[1:]:
        sum_col = np.nansum(ground_features[col])
        if sum_col == 0:
            break
        image_col_to_predict += 1
    
    target_col = target.columns[image_col_to_predict]
    
    x_list = []
    # reads images in the right order and adds to x_list
    for i in range(image_col_to_predict-SEQ_LEN,image_col_to_predict):  
        im = cv2.imread(f'{data_path}/recent_5_eval/image_{i}.png',cv2.IMREAD_GRAYSCALE)
        x_list.append(im)
    
    
    x_eval = np.array(x_list)
    x_eval = x_eval.reshape(1,SEQ_LEN,600,600,1)
    
    
    if os.path.isfile(f'{data_dir}/metatdata_newly_generated/cell_asignment_{grid_size}_{_region.replace(" ","_")}_eval.pickle'):
        grid_to_cell_asignment = pickle.load(open(f'{data_dir}/metatdata_newly_generated/cell_asignment_{grid_size}_{_region.replace(" ","_")}_eval.pickle', 'rb'))
    else:
        grid_to_cell_asignment = pickle.load(open(f'{data_dir}/metadata_provided/cell_asignment_{grid_size}_{_region.replace(" ","_")}_eval.pickle', 'rb'))
    
    if os.path.isfile(f'{data_dir}/metatdata_newly_generated/cell_asignment_{grid_size}_{_region.replace(" ","_")}_eval_grid_mask.pickle'):
        test_grid_mask = pickle.load(open(f'{data_dir}/metatdata_newly_generated/cell_asignment_{grid_size}_{_region.replace(" ","_")}_eval_grid_mask.pickle', 'rb'))
    else:
        test_grid_mask = pickle.load(open(f'{data_dir}/metadata_provided/cell_asignment_{grid_size}_{_region.replace(" ","_")}_eval_grid_mask.pickle', 'rb'))
    
    
    model = load_model(f'./models/snow_cast_{_region}_model_v2.hdf5')
    pred = model.predict(x_eval)
    
    image_predicted = np.array(pred[0,0,:,:,0])
    image_predicted = image_predicted.clip(0,2000) # negative snow levels dont make sense :)
    cv2.imwrite(f'{data_path}/predictions/image_{image_col_to_predict}.png', image_predicted)
    pred_values_test = []
    cell_test_list = []
    print(f'matching predictions from network to evaluation grid cells...')
    for gi in range(grid_size):
            for gj in range(grid_size):
                asignment = test_grid_mask[gi,gj]
                if asignment:
                    pred_values_test.append(image_predicted[gi,gj])
                    cell_test_list.append(grid_to_cell_asignment[gi,gj])
    for te in range(len(cell_test_list)):
        prediction_delivery.loc[prediction_delivery['cell_id']==cell_test_list[te],[target_col]] = pred_values_test[te]
    print('done')

prediction_delivery.replace(np.nan, 0, inplace=True)
prediction_delivery.rename(columns={'cell_id':''}, inplace=True)
prediction_delivery.to_csv(f'{data_dir}/eval/prediction_snow_cast.csv', index=False)















