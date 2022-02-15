"""

@author: Christian Pfister
"""


import numpy as np
import pandas as pd
import json
import cv2
import os
import glob
import pickle




def prepare_evaluation_data(_region, data_dir, target,ground_features,geo,grid_size = 600,SEQ_LEN=5):
    print(f'preparing evaluation data for region: {_region} ...')
    #%% find "index to predict" --> next weekly column that has no data
    cols = ground_features.columns
    col_index = 1
    for col in ground_features.columns[1:]:
        sum_col = np.nansum(ground_features[col])
        if sum_col == 0:
            break
        col_index += 1
    print(f'preparing evaluation data for:  {cols[col_index]}')
    #%% remove old inference data
    print('removing old evaluation data for clean inference...')
    image_list = glob.glob(f'{data_dir}/eval/{_region.replace(" ","_")}/recent_5_eval/*.png')
    for old_image in image_list:
        try:
            os.remove(old_image)
        except:
            pass
    
    #%% generate new / most recent inference data
    grid = np.zeros((grid_size,grid_size))
    # if new generated asingment files are available, use that, otherwise use the provided ones
    if os.path.isfile(f'{data_dir}/metatdata_newly_generated/station_asignment_{grid_size}_{_region.replace(" ","_")}_eval.pickle'):
        grid_to_station_asignment = pickle.load(open(f'{data_dir}/metatdata_newly_generated/station_asignment_{grid_size}_{_region.replace(" ","_")}_eval.pickle','rb'))
    else:
        grid_to_station_asignment = pickle.load(open(f'{data_dir}/metadata_provided/station_asignment_{grid_size}_{_region.replace(" ","_")}_eval.pickle','rb'))
    
    for t in range(col_index-SEQ_LEN,col_index):
        grid = np.zeros((grid_size,grid_size))
        for gi in range(grid_size):
            for gj in range(grid_size):
                station = grid_to_station_asignment[gi,gj]
                if not station == '':
                    grid[gi,gj] = np.array(ground_features[ground_features['Unnamed: 0'] == station])[0][t]
        cv2.imwrite(f'{data_dir}/eval/{_region.replace(" ","_")}/recent_5_eval/image_{t}.png',grid)
        cv2.imwrite(f'{data_dir}/eval/{_region.replace(" ","_")}/all_eval_data/image_{t}.png',grid)
    print(f'finished preparing evaluation data for region {_region} and date: {cols[col_index]}')
    print('#########################################')
