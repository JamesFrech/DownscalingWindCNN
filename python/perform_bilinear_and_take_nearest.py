
##############################################################################
# Take nearest value as proxy and compute bilinear interpolation on test set #
##############################################################################

import xarray as xr
import pandas as pd
from bilinear import bilinear

# Get coordinats of the buoy and the lat/lons of the nearest observations
# to perform bilinear interpolation
buoy_coords=(360-74.936,31.759)
nearest=(360-75,31.75,360-74.75,32)

inputdir='data/train'
inputdir='data'

# Read in buoy and only grab independent test set
buoy=pd.read_csv(f'{inputdir}/buoy41002.csv')
buoy=buoy.loc[buoy['fold']==5]
# Get datetime index
buoy['time']=pd.to_datetime(buoy['time'])
buoy.index=buoy['time']
buoy.drop(['fold','time'],axis=1,inplace=True)

# Read in NBS and only grad nearest 4 points
nbs=xr.open_dataset(f'{inputdir}/nbs_train.nc')
nbs=nbs.sel(lat=slice(nearest[1],nearest[3]), \
            lon=slice(nearest[0],nearest[2]), \
            time=(nbs['time'].isin(buoy.index)))
nbs=nbs.to_dataframe().reset_index()
nbs.drop('zlev',axis=1,inplace=True)

# Compute bilinear interpolation to downscale winds to buoy
buoy['bilinear']=nbs.groupby('time').apply(lambda x: bilinear(x,location=buoy_coords,nearest_xy=nearest))

# Get the values from the nearest location to use as a baseline
nbs.set_index('time',inplace=True)
buoy['nearest']=nbs.loc[(nbs['lon']==360-75)&(nbs['lat']==31.75),['w']]

# Print out error metrics
print((buoy['nearest']-buoy['w']).describe())
print((buoy['bilinear']-buoy['w']).describe())

print(abs(buoy['nearest']-buoy['w']).describe())
print(abs(buoy['bilinear']-buoy['w']).describe())

# Output results to csv
buoy.to_csv('data/results/test/nearest_and_bilinear_results.csv')
