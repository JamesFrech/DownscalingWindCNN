import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Read in clean buoy data
buoy=pd.read_csv('data/41002_clean.csv',index_col=0)
# Turn index into datetime values
buoy.index=pd.DatetimeIndex(buoy.index)

# Independent test set
test=buoy.loc[buoy.index.year>=2017]
# train and validate data
train=buoy.loc[buoy.index.year<2017]


# Get the size of a fold
n=int(train.shape[0]/5)
# Initialize all fold values to 6 (no fold)
train['fold']=[6]*train.shape[0]

# Create folds and have 15 day splits (60 obs) 
# to avoid information leakage
train.iloc[:n,1]=0
train.iloc[n+60:2*n,1]=1
train.iloc[2*n+60:3*n,1]=2
train.iloc[3*n+60:4*n,1]=3
train.iloc[4*n+60:,1]=4

# Remove points in gaps that weren't assigned to a fold
train=train.loc[train[f'fold']!=6]

# Merge test and train data
train=train[['w','fold']]
# Test data will have fold value 5
test['fold']=[5]*test.shape[0]
# Make a 15 day split
test=test.iloc[60:]
buoy=pd.concat([train,test])

# Plot out the time series
buoy['colors']='C'+buoy['fold'].astype(str) 
buoy['time']=buoy.index
# Get each fold to plot individually
f0=buoy.loc[buoy['fold']==0]
f1=buoy.loc[buoy['fold']==1]
f2=buoy.loc[buoy['fold']==2]
f3=buoy.loc[buoy['fold']==3]
f4=buoy.loc[buoy['fold']==4]
test=buoy.loc[buoy['fold']==5]

fig=plt.figure(figsize=(12,4))
ax=plt.axes()
f0.plot(x='time',y='w',kind='scatter',s=.1,xlabel='Time', \
        ylabel='Wind Speed (m/s)',title='Train/Validate/Test Split',label='fold 1', \
        ax=ax,c='C0')
f1.plot(x='time',y='w',kind='scatter',s=.1,xlabel='Time', \
        ylabel='Wind Speed (m/s)',label='fold 2',ax=ax,c='C1')
f2.plot(x='time',y='w',kind='scatter',s=.1,xlabel='Time', \
        ylabel='Wind Speed (m/s)',label='fold 3',ax=ax,c='C2')
f3.plot(x='time',y='w',kind='scatter',s=.1,xlabel='Time', \
        ylabel='Wind Speed (m/s)',label='fold 4',ax=ax,c='C3')
f4.plot(x='time',y='w',kind='scatter',s=.1,xlabel='Time', \
        ylabel='Wind Speed (m/s)',label='fold 5',ax=ax,c='C4')
test.plot(x='time',y='w',kind='scatter',s=.1,xlabel='Time', \
          ylabel='Wind Speed (m/s)',label='Test',ax=ax,c='C5')
plt.legend(markerscale=7,bbox_to_anchor=(0.825,-.15),ncol=7)
plt.savefig('images/buoy_time_series_split.png',bbox_inches='tight',dpi=300)
plt.close()

# Print size of train, validate, and test sets.
print(f0.shape[0]+f1.shape[0]+f2.shape[0]+f3.shape[0])
print(f4.shape[0])
print(test.shape[0])

# Output training data with labeled folds to csv
buoy[['w','fold']].to_csv('data/train/buoy41002.csv')
