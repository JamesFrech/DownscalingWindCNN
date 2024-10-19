# Function to perform bilinear interpolation

def bilinear(df,location,nearest_xy):
    '''
    Perform bilinear interpolation.


    Inputs:
    df (dataframe): Dataframe from which the neighboring values of the interpolation point are.

    location (tuple): (longitude,latitude) of grid point to interpolate.

    nearest_xy (tuple): x and y values to use for interpolation at location.
    Give in format (x1,y1,x2,y2).


    Output:
    interp_val (float): Interpolated value at given location.
    '''

    x,y=location
    x1,y1,x2,y2=nearest_xy

    # Compute the weights
    denom=(x2-x1)*(y2-y1)
    w11=(x2-x)*(y2-y)/denom
    w12=(x2-x)*(y-y1)/denom
    w21=(x-x1)*(y2-y)/denom
    w22=(x-x1)*(y-y1)/denom

    # Get the values at the near locations
    f11=df.loc[(df['lon']==x1)&(df['lat']==y1),'w'].values[0]
    f12=df.loc[(df['lon']==x1)&(df['lat']==y2),'w'].values[0]
    f21=df.loc[(df['lon']==x2)&(df['lat']==y1),'w'].values[0]
    f22=df.loc[(df['lon']==x2)&(df['lat']==y2),'w'].values[0]
    
    # Perform bilinear interpolation
    interpolated=w11*f11+w12*f12+w21*f21+w22*f22
    return(interpolated)
