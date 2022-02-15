"""

@author: Christian Pfister
"""




import numpy as np
import pandas as pd
import json
import os
import pickle


#%% helper functions

def point_is_in_cell(lo_min,la_min,lo_max,la_max, lon_test,lat_test):
    if (lon_test > lo_min) & (lon_test < lo_max) & (lat_test > la_min) & (lat_test < la_max): #> < reverted for lon, because neg nummbers in this range
        return True
    else:
        return False

def cell_center_is_in_cell(lo_min,la_min,lo_max,la_max, lon_min_test,lat_min_test,lon_max_test,lat_max_test):
    lo_center_test = np.mean([lon_min_test,lon_max_test])
    la_center_test = np.mean([lat_min_test,lat_max_test])
    if (lo_center_test > lo_min) & (lo_center_test < lo_max) & (la_center_test > la_min) & (la_center_test < la_max):
        return True
    else:
        return False

class ContinueI(Exception):
    pass


def asing_evaluation_cells(_region,data_dir,target,ground_features,geo, generate_new = False):
    print(f'Asigning evaluation cells for region {_region}')
    
    #%% creating a list of training / testing cells for the specified region
    cell_ids_eval = np.array(target['cell_id'])
    
    cell_ids_eval_region = []
    
    for cell_nr in range (len(geo['features'])):
        cell_id = geo['features'][cell_nr]['properties']['cell_id']
        if geo['features'][cell_nr]['properties']['region'] == _region:
            if cell_id in cell_ids_eval:
                cell_ids_eval_region.append(cell_id)
    
    
    #%% defining the regions maximum extend in lon and lat
    lons_all = []
    lats_all = []
    for cell_nr in range (len(geo['features'])):
        cell_id = geo['features'][cell_nr]['properties']['cell_id']
        if (cell_id in cell_ids_eval_region):
            for i in range (5):
                lons_all.append(geo['features'][cell_nr]['geometry']['coordinates'][0][i][0])
                lats_all.append(geo['features'][cell_nr]['geometry']['coordinates'][0][i][1])
    min_lat = np.min(lats_all)
    min_lon = np.min(lons_all)
    max_lat = np.max(lats_all)
    max_lon = np.max(lons_all)
    
    
    
    #%%
    """
    Evaluation data:
    define a grid of grid_size x grid_size pixels that span over the area where all the cells lie
    asign each cell (center of cell) to a pixel in the grid
    """
    grid_size = 600
    lat_grid_points = np.linspace(min_lat,max_lat,grid_size)
    lon_grid_points = np.linspace(min_lon,max_lon,grid_size)
    grid = np.zeros((grid_size,grid_size))
    grid_mask = np.array(grid, dtype=bool)
    grid_to_cell_asignment = np.empty([grid_size, grid_size], dtype="<U45")
    
    continue_i = ContinueI()
    # skips this step if metadata is available or not wanted to generate new
    for run in range(1):
        print('checking for existing meta data files...')
        if os.path.isfile(f'{data_dir}/metatdata_newly_generated/cell_asignment_{grid_size}_{_region.replace(" ","_")}_eval.pickle'):
            print(f'found existing newly generated file: {data_dir}/metatdata_newly_generated/cell_asignment_{grid_size}_{_region.replace(" ","_")}_eval.pickle')
            print('skipping step')
            continue
        elif (not generate_new) & (os.path.isfile(f'{data_dir}/metadata_provided/cell_asignment_{grid_size}_{_region.replace(" ","_")}_eval.pickle')):
            print(f'found existing provided file: {data_dir}/metadata_provided/cell_asignment_{grid_size}_{_region.replace(" ","_")}_eval.pickle')
            print(f'generating new meta data files is set to  {generate_new} , skipping step')
            continue
    
        for cell_nr in range (len(geo['features'])):
            cell_id = geo['features'][cell_nr]['properties']['cell_id']
            if not cell_id in cell_ids_eval_region:
                print(f'Cell with id {cell_id} is not in eval region {_region}, moving on')
                continue
            lons = []
            lats = []
            for i in range (5):
                lons.append(geo['features'][cell_nr]['geometry']['coordinates'][0][i][0])
                lats.append(geo['features'][cell_nr]['geometry']['coordinates'][0][i][1])
            try:
                for gi in range(grid_size-1):
                    for gj in range(grid_size-1):
                        asignment = cell_center_is_in_cell(lon_grid_points[gi],lat_grid_points[gj],lon_grid_points[gi+1],lat_grid_points[gj+1], lons[0],lats[0],lons[2],lats[2])
                        if asignment:
                            grid_to_cell_asignment[gi,gj] = cell_id
                            print(f'found place for cell {cell_id}')
                            grid_mask[gi,gj] = asignment
                            print(f'finished with cell number  {cell_nr}  our of  {len(geo["features"])}')
                            raise continue_i
            except:
                continue
            print(f'finished look up for cell {cell_id} with number  {cell_nr}  without success')
        
        
        pickle.dump(grid_to_cell_asignment, open(f'{data_dir}/metatdata_newly_generated/cell_asignment_{grid_size}_{_region.replace(" ","_")}_eval.pickle','wb'))
        pickle.dump(grid_mask, open(f'{data_dir}/metatdata_newly_generated/cell_asignment_{grid_size}_{_region.replace(" ","_")}_eval_grid_mask.pickle','wb'))
        
        print('#########################################')
        print('Finished with asining train cells to grid')
    #%%
    """
    Ground Stations:
    define a grid of grid_size x grid_size pixels that span over the area where all the cells lie
    asign each ground station to a pixel in the grid
    """
    print(f'Asigning ground stations for region {_region}')
    grid_size = 600
    grid = np.zeros((grid_size,grid_size))
    grid_mask_g = np.array(grid, dtype=bool)
    grid_to_cell_asignment_g = np.empty([grid_size, grid_size], dtype="<U45")
    lat_grid_points = np.linspace(min_lat,max_lat,grid_size)
    lon_grid_points = np.linspace(min_lon,max_lon,grid_size)
    locations = np.array(ground_features['Unnamed: 0'])
    # skips this step if metadata is available or not wanted to generate new
    for run in range(1):
        if os.path.isfile(f'{data_dir}/metatdata_newly_generated/station_asignment_{grid_size}_{_region.replace(" ","_")}_eval.pickle'):
            print(f'found existing newly generated file: {data_dir}/metatdata_newly_generated/station_asignment_{grid_size}_{_region.replace(" ","_")}_eval.pickle')
            print('skipping step')
            continue
        elif (not generate_new) & (os.path.isfile(f'{data_dir}/metadata_provided/station_asignment_{grid_size}_{_region.replace(" ","_")}_eval.pickle')):
            print(f'found existing provided file: {data_dir}/metadata_provided/station_asignment_{grid_size}_{_region.replace(" ","_")}_eval.pickle')
            print(f'generating new meta data files is set to  {generate_new} , skipping step')
            continue
    
        for loc in locations:
            long_g = ground_meta[ground_meta['station_id'] == loc]['longitude'].values[0]
            lat_g = ground_meta[ground_meta['station_id'] == loc]['latitude'].values[0]
            try:
                for gi in range(grid_size-1):
                    for gj in range(grid_size-1):
                        asignment = point_is_in_cell(lon_grid_points[gi],lat_grid_points[gj],lon_grid_points[gi+1],lat_grid_points[gj+1], long_g,lat_g)
                        if asignment:
                            grid_to_cell_asignment_g[gi,gj] = loc
                            print(f'found place for station {loc}')
                            grid_mask_g[gi,gj] = asignment
                            raise continue_i
            except:
                continue
            print(f'finished look up for station {loc} without success')
        
        
        pickle.dump(grid_to_cell_asignment_g, open(f'{data_dir}/metatdata_newly_generated/station_asignment_{grid_size}_{_region.replace(" ","_")}_eval.pickle','wb'))
        pickle.dump(grid_mask_g, open(f'{data_dir}/metatdata_newly_generated/station_asignment_{grid_size}_{_region.replace(" ","_")}_eval_grid_mask.pickle','wb'))
        
        print('#########################################')
        print('Finished with asining ground stations to grid')
    print('######################################################')