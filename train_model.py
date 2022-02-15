# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 17:36:48 2022

@author: chpf1
"""


import numpy as np
import pandas as pd
import random
import cv2
import glob
import tensorflow as tf
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization,Conv3D, MaxPool3D, Conv3DTranspose
import time
import os
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split



SEQ_LEN = 5
FUTURE_STEPS = 1
batch_size = 2
_region = 'sierras'         # 'sierras' or 'central rockies'
data_path=f'./data/dev/train/{_region.replace(" ","_")}'

season_sep = [32,62,93,134,166,207,262] ## values is first of new winter season 
#makes sure that on season change (June --> Dec to split the data)

#%% rough training with swe and ground station data
image_list = glob.glob(f'{data_path}/swe_ground/*.png')
data_array = []

for i in range(1,len(image_list)+1):  # reads images in the right order and adds to data_array
    im = cv2.imread(f'{data_path}/swe_ground/image_{i}.png',cv2.IMREAD_GRAYSCALE)
    data_array.append(im)

x_list = []
y_list = []
counter_test = []
image_counter = 0
while image_counter+SEQ_LEN < len(image_list):
    if image_counter+SEQ_LEN in season_sep: #makes sure that on season change (June --> Dec to split the data)
        image_counter += SEQ_LEN
    x_temp = []
    counter_test.append(image_counter)
    for s in range(SEQ_LEN):
        x_temp.append(data_array[image_counter+s])
    x_list.append(x_temp)
    y_list.append(data_array[image_counter+SEQ_LEN])
    image_counter += 1



######## fine tuning with only ground station data:

image_list_fine = glob.glob(f'{data_path}/only_ground/image_{i}.png')
data_array_fine = []

for i in range(1,len(image_list_fine)+1):
    im = cv2.imread(f'{data_path}/only_ground/image_{i}.png',cv2.IMREAD_GRAYSCALE)
    data_array_fine.append(im)


x_list_fine = []
y_list_fine = []
counter_test = []
image_counter = 0
while image_counter+SEQ_LEN < len(image_list):
    if image_counter+SEQ_LEN in season_sep: #makes sure that on season change (June --> Dec to split the data)
        image_counter += SEQ_LEN
    x_temp = []
    counter_test.append(image_counter)
    for s in range(SEQ_LEN):
        x_temp.append(data_array[image_counter+s])
    x_list_fine.append(x_temp)
    y_list_fine.append(data_array[image_counter+SEQ_LEN])
    image_counter += 1



#%% create x_train and y_train feature - labels



X_train, X_test, y_train, y_test = train_test_split(x_list,y_list,test_size=0.1, random_state=321984)
X_train = np.array(X_train)
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],X_train.shape[3],1)

X_test = np.array(X_test)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],X_test.shape[3],1)

y_train = np.array(y_train)
y_train = y_train.reshape(y_train.shape[0],1,y_train.shape[1],y_train.shape[2],1)

y_test = np.array(y_test)
y_test = y_test.reshape(y_test.shape[0],1,y_test.shape[2],y_test.shape[2],1)


######## fine tuning with only ground station data:

X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(x_list_fine,y_list_fine,test_size=0.1, random_state=321984)
X_train_f = np.array(X_train_f)
X_train_f = X_train_f.reshape(X_train_f.shape[0],X_train_f.shape[1],X_train_f.shape[2],X_train_f.shape[3],1)

X_test_f = np.array(X_test_f)
X_test_f = X_test_f.reshape(X_test_f.shape[0],X_test_f.shape[1],X_test_f.shape[2],X_test_f.shape[3],1)

y_train_f = np.array(y_train_f)
y_train_f = y_train_f.reshape(y_train_f.shape[0],1,y_train_f.shape[1],y_train_f.shape[2],1)

y_test_f = np.array(y_test_f)
y_test_f = y_test_f.reshape(y_test_f.shape[0],1,y_test_f.shape[2],y_test_f.shape[2],1)


#%%
opt = tf.keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)

nb_epochs = 10
nb_epochs_f = 15

model = Sequential()
model.add(Conv3D(filters=100, kernel_size=5, strides=(1,3,3), padding="same", activation='relu', 
                  input_shape =(SEQ_LEN,  X_train.shape[2], X_train.shape[3],1),data_format='channels_last' ))

model.add(MaxPool3D(pool_size=(1,2,2), padding='same'))
model.add(Conv3D(filters=150, kernel_size=5, strides=(1,2,2), padding="same", activation='relu'))
model.add(MaxPool3D(pool_size=(1,2,2), padding='same'))
model.add(Conv3D(filters=150, kernel_size=5, strides=(5,1,1), padding="same", activation='relu'))
model.add(Dense(600, activation='relu'))
model.add(Dense(600, activation='relu'))
model.add(Conv3DTranspose(filters=150, kernel_size=5, strides=(1,2,2), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv3DTranspose(filters=100, kernel_size=5, strides=(1,2,2), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv3DTranspose(filters=100, kernel_size=5, strides=(1,2,2), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv3DTranspose(filters=100, kernel_size=5, strides=(1,3,3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Dense(800, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer=opt)
model.summary()
    
start_time = time.time()


#rough training with swe + ground measurements as input data
model.fit(X_train,y_train,batch_size=batch_size,epochs=nb_epochs, verbose=1, shuffle=False)

#fine tuning with only ground measurements as input data --> like in inference
model.fit(X_train_f,y_train_f,batch_size=batch_size,epochs=nb_epochs_f, verbose=1, shuffle=False)

print(f'--------time passed:   {round((time.time() - start_time)/60,1)}  minutes------')
if not os.path.isfile(f'./models/snow_cast_{_region}_model_v2.hdf5'):
    model.save(f'./models/snow_cast_{_region}_model_v2.hdf5')

"""
with open('modelsummary.txt', 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))
"""
#%% calculate rsme for all test cells, pred values vs gt
# loading only a few examples at a time, due to limited GPU-RAM
pred_f = model.predict(X_test_f[0:1,:,:,:])
grid_size = 600
data_dir = './data/dev'

test_grid_mask = pickle.load(open(f'{data_dir}/metadata_provided/cell_asignment_{grid_size}_{_region.replace(" ","_")}_test_grid_mask.pickle', 'rb'))
rsme_list = []
j=0
for p in pred_f:
    pred_values = []
    gt_values = []
    im_p = np.array(p[0,:,:,0])
    im_gt = np.array(y_test_f[j,0,:,:,0])
    for gi in range(grid_size):
        for gj in range(grid_size):
            asignment = test_grid_mask[gi,gj]
            if asignment:
                pred_values.append(im_p[gi,gj])
                gt_values.append(im_gt[gi,gj])
    rsme = mean_squared_error(pred_values, gt_values, squared=False)
    rsme_list.append(rsme)
    print(f'RSME for test image nr  {j}:   {rsme}')
    j+=1
print('###########################')
print(f'Mean RSME over all test data:  {np.mean(rsme_list)}')
print('###########################')

# print(f'Test RMSE :   {mean_squared_error(pred[0,:,:,0],y_test[0,:,:,0],squared=False)}') 


#%% write predictions and ground truths as images for manual evaluation
"""
j=0
for p in pred_f:
    # for j in range(SEQ_LEN):
    
    im_p = np.array(p[0,:,:,0])
    im_gt = np.array(y_test_f[j,0,:,:,0])
    cv2.imwrite(f'./data/debug/image_pred_{j}.png', im_p)
    cv2.imwrite(f'./data/debug/image_gt_{j}.png',im_gt)
    j+=1
    
cv2.imwrite(f'./data/debug/X_train_0.png', np.array(X_train_f[0,0,:,:,0]))
"""
