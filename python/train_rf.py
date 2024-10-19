# Tune the hyperparameters of the random forest model using
# 5 fold cross validation

import pandas as pd
import xarray as xr
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
from random import seed
from multiprocessing import Pool
import joblib
import itertools
seed(1234)

import warnings
#suppress warnings
warnings.filterwarnings('ignore')

# Directories for input and output
inputdir='data/train'
inputdir='data'
outputdir='data/results/train'

# Window size is 2n+1
n=1

center=[360-75,31.75]
lats=np.arange(center[1]-n*.25,center[1]+(n+1)*.25,.25)
lons=np.arange(center[0]-n*.25,center[0]+(n+1)*.25,.25)

lat_bounds=[min(lats),max(lats)]
lon_bounds=[min(lons),max(lons)]

# Read in training/testing data
buoy=pd.read_csv(f'{inputdir}/buoy41002.csv',index_col=0)
# Only use train and validation datasets
buoy=buoy.loc[buoy['fold']!=5] # 5 is independent test set

# Read in NBS
nbs=xr.open_dataset(f'{inputdir}/nbs_train.nc')
# Only get box size 2n+1 and convert to dataframe
nbs=nbs.sel(lat=slice(min(lats),max(lats)),lon=slice(min(lons),max(lons)))
nbs=nbs.to_dataframe().reset_index()
nbs.drop('zlev',axis=1,inplace=True)

nbs=nbs.pivot(columns=['lat','lon'],index='time')
nbs.columns=[f'L{i}' for i in range(nbs.shape[1])]

# Hyperparameter values to tune
n_estimators_values=[20,40,60,80,100,200,300,400,500,600,700]
min_samples_leaf_values=[5,10,15,20,25,30,35]
max_features_values=[int(nbs.shape[1]/3)]
# Fold values
folds=[0,1,2,3,4]
tasks=list(itertools.product(n_estimators_values, \
                             min_samples_leaf_values, \
                             max_features_values, \
                             folds))

def run_model(ntrees,min_samples,max_features,fold):
    print(f'Starting: fold: {fold}, {ntrees}, {min_samples}')

    # buoy data is target Y
    # NBS data is features X

    # Train using all other folds
    train_Y=buoy.loc[buoy['fold']!=fold,['w']]
    train_X=nbs.loc[nbs.index.isin(train_Y.index)]
    # Validate using given fold
    val_Y=buoy.loc[buoy['fold']==fold,['w']]
    val_X=nbs.loc[nbs.index.isin(val_Y.index)]


    # Initialize and train the model
    regressor = RandomForestRegressor(n_estimators=ntrees, \
                                      min_samples_leaf=min_samples, \
                                      max_features=max_features, \
                                      oob_score=True, \
                                      random_state=0,n_jobs=1)
    start=datetime.now()
    regressor.fit(train_X,train_Y)
    print(f'Time to train fold {fold}',datetime.now() - start) # See how long the model took to train

    oob=pd.DataFrame()
    oob['fold']=[fold]
    oob['ntree']=[ntrees]
    oob['min_samples']=[min_samples]
    oob['OOB_Error'] = [1 - regressor.oob_score_]
    oob.to_csv(f'{outputdir}/rf_oob.csv',mode='a',header=False,index=False)

processes_start=datetime.now()
# Read in data to test models on each location separately
if __name__ == '__main__':
    with Pool(processes=10) as pool:
        # Run all the combinations of hyperparameters and save the output dataframes
        pool.starmap(run_model,tasks)
print('Total time')
print(datetime.now()-processes_start)

