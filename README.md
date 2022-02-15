# Code Repository for Snowcast Showdown
Link to the competition:
Development stage:

https://www.drivendata.org/competitions/86/competition-reclamation-snow-water-dev/page/414/

Evaluation stage:

https://www.drivendata.org/competitions/90/competition-reclamation-snow-water-eval/page/430/

## General remarks:
For model training I am using only the ground measurements (ground_measures_train_features.csv) and the swe information for the training grid cells (train_labels.csv) from the provided data downloads.

This solution only adresses the "sierras" and "central rockies" sub-regions for this contest!!!
However it could be extended to include the entire western US. To include other data sources, a feature fusion could be achived  in the fully connected layers in the middle of the network. I did not have the hardware (GPUs) to realize these approches.

To run the inference, there are .pickle files needed that I generated (s. description later). If you do not want to use the provided file, you can run the scripts `cell_asingment_training_data.py` `and cell_asingment_eval_data.py` with the argument `generate_new=True`.
All scripts always look if there are newly generated metadata-files available, and if possible use thesee. The provided ones are only used if there are no other files available.




## Prerequisits
The development was done in Python 3.7 (the only tested version)

The minimum required Python version for this repository to run is Python 3.6 

This repository uses Tensorflow 2.1.0, other 2.x versions will likely work too, but are not testet.

Important: if you use the -gpu version of Tensorflow, you need a compatible CUDA Version (e.g. 10.1) and appropriate NVIDIA driver!

Install required Python packages. 
`pip install -r requirements.txt`

If you have a fresh installation of Tensorflow 2.1.0, you need to do the following:

in your environment (e.g. "snow-env" ) open this file: `snow-env\lib\site-packages\tensorflow_core\python\keras\saving\hdf5_format.py` and remove all `.decode('utf-8')` and `.decode('utf8')` tp prevent errors.

Make sure the following data files are in these folders:
`./data/dev`: `grid_cells.geojson`, `ground_measures_metadata.csv`, `ground_measures_test_features.csv`, `ground_measures_train_features`, `submission_format.csv`, `train_labels.csv`;

`./data/eval`:  `grid_cells.geojson`, `ground_measures_features.csv`, `ground_measures_metadata.csv`, `labels_2020_2021.csv`, `submission_format.csv`;

`./data/eval/eval`: `prediction_snow_cast.csv`(the initial prediction file)

## General Idea of my Aproach
I created a Grid of 600 x 600 for the sub-reagions "sierras" and "central rockies". Their min and max are automatically determined by the GPS locations of the Grid cells with the labels of the respective region.

To each of these grid-points, a ground measurement station or a provided training/evaluation cell (-center) is matched. This happens in the scipts `cell_asingment_training_data.py` and `cell_asingment_eval_data.py`. These are controlled by the `prepare_training_data.py` and `inference.py` scripts respectively.

For each week in the training data, there is a picture (600x600 pixel) created with the swe level in the matched pixels.

I am using a sequence length of 5, i.e. the network uses the past 5 weeks of ground measurement data to predict 1 week ahead.

These images are then used for training a CNN-type architecture which can be seen in the modelsummary.txt

It uses 3D convolutions, to adress spacial as well as temporal aspects of the data.



## Usage Inference
Copy the most recent! version of `ground_measures_features.csv` in the folder `./data/eval/`. make sure it contains all data (columns) up to the point that you want to predict. The week to predict needs to be the first column with ONLY NANs!
then run `python inference.py` or `python3 inference.py` (depending on your installation).

This will update the prediction set: `/data/eval/eval/prediction_snow_cast.csv` with the predictions for the new week.

## Usage Training
If you want to retrain the model(s):

Run `prepare_training_data.py` once with the `_region` variable set to "sierras" and once set to "central rockies" to generate all training data
Run `python train_model.py` once with the `_region` variable set to "sierras" and once set to "central rockies"The models will be saved to `./models` with a clear name for the respective region.

## Provided Data:
Make sure to follow the instructins under "Prerequesites" to place the provided data files in the respective folders

A copy of this repositrory will be hosted on my  [Goodgle Drive](https://drive.google.com/drive/folders/19SeDjPlYD4t7BQqFc8DIS33DstWUUCyj?usp=sharing)

The meta data files as well as pre-trained models are provided in this link in seperate sub-folders

There is a sub folder "meta_data" which needs to be placed in the folder `./data` 

Another subfolder "models" contains the pre-trained models for the regions "sierras" and "central rockies", these need to placed in the folder `./models`

