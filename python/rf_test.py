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

# Directory for input and output
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
# Get train and Test buoy datasets
train_Y=buoy.loc[buoy['fold']!=5] # 5 is independent test set
test_Y=buoy.loc[buoy['fold']==5]

# Get rid of fold values as they aren't necessary for final model
train_Y.drop('fold',axis=1,inplace=True)
test_Y.drop('fold',axis=1,inplace=True)

# Read in NBS
nbs=xr.open_dataset(f'{inputdir}/nbs_train.nc')
# Only get box size 2n+1 and convert to dataframe
nbs=nbs.sel(lat=slice(min(lats),max(lats)),lon=slice(min(lons),max(lons)))
nbs=nbs.to_dataframe().reset_index()
nbs.drop('zlev',axis=1,inplace=True)

# Make each grid point its own column
nbs=nbs.pivot(columns=['lat','lon'],index='time')
nbs.columns=[f'{i[1]}_{i[2]}' for i in nbs.columns]

# Get train and test NBS data
train_X=nbs.loc[nbs.index.isin(train_Y.index)]
test_X=nbs.loc[nbs.index.isin(test_Y.index)]

# Hyperparameter values
n_estimators_values=[500]
min_samples_leaf_values=[30]
max_features_values=[int(nbs.shape[1]/3)]
tasks=list(itertools.product(n_estimators_values, \
                             min_samples_leaf_values, \
                             max_features_values))

def run_model(ntrees,min_samples,max_features):
    print(f'Starting: {ntrees}, {min_samples}, {max_features} ')

    # Initialize and train the model
    regressor = RandomForestRegressor(n_estimators=ntrees, \
                                      min_samples_leaf=min_samples, \
                                      max_features=max_features, \
                                      oob_score=True, \
                                      random_state=0,n_jobs=1)
    start=datetime.now()
    regressor.fit(train_X,train_Y)
    print(f'Time to train',datetime.now() - start) # See how long the model took to train

    # Get the feature importances
    FI=pd.DataFrame(np.array(regressor.feature_importances_))
    FI=FI.T
    FI.columns=train_X.columns
    FI=FI.T
    FI.columns=['FI']
    FI.sort_values('FI',inplace=True,ascending=False)
    print(FI)

    # Predict the wind speeds
    pred=regressor.predict(test_X)
    test_Y['pred']=pred

    # Print errors
    print((test_Y['pred']-test_Y['w']).describe(),'\n',abs(test_Y['pred']-test_Y['w']).describe())
    test_Y.to_csv(f'{outputdir}/rf_results.csv')

processes_start=datetime.now()
# Read in data to test models on each location separately
if __name__ == '__main__':
    with Pool(processes=5) as pool:
        # Run all the combinations of hyperparameters and save the output dataframes
        pool.starmap(run_model,tasks)
print('Total time')
print(datetime.now()-processes_start)

