import itertools
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

buoy_coords=[360-74.936,31.759]
center=[360-75,31.75]

n=10
lats=np.arange(center[1]-n*.25,center[1]+(n+1)*.25,.25)
lons=np.arange(center[0]-n*.25,center[0]+(n+1)*.25,.25)

locations=list(itertools.product(lons, lats))

location_lons=[i[0] for i in locations]
location_lats=[i[1] for i in locations]


png='images/BuoyGrid.png'
# Find the values needed to plot the correct bounds on the maps
max_lon = 360-70
min_lon = 360-80
mid_lon = (max_lon+min_lon)/2
lon_dist= max_lon-mid_lon
max_lat = 35
min_lat = 25
mid_lat = (max_lat+min_lat)/2
lat_dist= max_lat-mid_lat

# Create shaded contour map for EOF i
crs0 = ccrs.PlateCarree(central_longitude=0)
# Have to do this for anything centered over Pacific Ocean. If not Pacific, works anyways
crs180 = ccrs.PlateCarree(central_longitude=mid_lon)
fig = plt.figure() # initialize figure
ax = plt.axes(projection=crs180)
# Extent of graph. Need to + or - 1 for lat to get labeled end points for some reason
extent = [min_lon, max_lon, min_lat, max_lat]
ax.set_extent(extent, crs=crs0)
# Draws a grid and puts x,y labels for coordinates
gl=ax.gridlines(crs=crs0,draw_labels=True,
                dms=True,x_inline=False,y_inline=False)
# Removes the x and y grid lines if you would like. Can comment these two out and keep grid.
gl.xlines=False
gl.ylines=False
gl.top_labels=False
gl.right_labels=False
# Add land and coastline features to the map. Keeping coast/removing land keeps outlines of countries.
ax.add_feature(cfeature.LAND,edgecolor='green',zorder=2)
ax.add_feature(cfeature.OCEAN,edgecolor='green',zorder=1)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.STATES,zorder=3)
#Sets coordinates at which latitudes are labeled. Not perfect...
#ax.yaxis.set_ticks([0,15,30],crs=crs0)

plt.title(f'Buoy 41002 South Hatteras') # Title the plot. Uses the column name of the EOF in your data (EOFi) likely.
plt.scatter(location_lons,location_lats,s=1,transform=crs0,c='black')#,c=dat[0])
plt.scatter(buoy_coords[0],buoy_coords[1],s=1,transform=crs0,c='red')#,c=dat[0])
plt.savefig(png,bbox_inches='tight',dpi=300)
plt.close()
