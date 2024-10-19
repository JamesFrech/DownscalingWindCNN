# Plot errors for each model on test set
# Plot both time series and error bar charts

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

inputdir='data/results/test'
inputdir='data'

# Read in predictions on test set for each model and merge into one dataframe
df1=pd.read_csv(f'{inputdir}/nearest_and_bilinear_results.csv')
df2=pd.read_csv(f'{inputdir}/rf_results.csv')
df3=pd.read_csv(f'{inputdir}/CNN_normalized01_testresults.csv',index_col=0)
df4=pd.read_csv(f'{inputdir}/CNN_standardizedboxcox_testresults.csv',index_col=0)
df5=pd.read_csv(f'{inputdir}/CNN_standardizedboxcox_testresults_customloss_leakyrelu.csv',index_col=0)

df2.rename({'pred':'rf'},axis=1,inplace=True)
df3.rename({'true':'w','pred':'CNN_Norm'},axis=1,inplace=True)
df4.rename({'true':'w','pred':'CNN_Stand'},axis=1,inplace=True)
df5.rename({'true':'w','pred':'CNN_Loss_Leaky'},axis=1,inplace=True)

# drop duplicates of true value
df2.drop('w',axis=1,inplace=True)
df3.drop('w',axis=1,inplace=True)
df4.drop('w',axis=1,inplace=True)
df5.drop('w',axis=1,inplace=True)

df=df1.merge(df2,on='time')
df=df.merge(df3,on='time')
df=df.merge(df4,on='time')
df=df.merge(df5,on='time')
df['time']=pd.to_datetime(df['time'])

####################
# Calculate Errors #
####################

df['nearest_diff']=df['nearest']-df['w']
df['bilinear_diff']=df['bilinear']-df['w']
df['rf_diff']=df['rf']-df['w']
df['CNN_Norm_diff']=df['CNN_Norm']-df['w']
df['CNN_Stand_diff']=df['CNN_Stand']-df['w']
df['CNN_Loss_Leaky_diff']=df['CNN_Loss_Leaky']-df['w']

df['abs_nearest_diff']=abs(df['nearest']-df['w'])
df['abs_bilinear_diff']=abs(df['bilinear']-df['w'])
df['abs_rf_diff']=abs(df['rf']-df['w'])
df['abs_CNN_Norm_diff']=abs(df['CNN_Norm']-df['w'])
df['abs_CNN_Stand_diff']=abs(df['CNN_Stand']-df['w'])
df['abs_CNN_Loss_Leaky_diff']=abs(df['CNN_Loss_Leaky']-df['w'])

print(df[['nearest_diff','bilinear_diff','rf_diff','CNN_Norm_diff', \
          'CNN_Stand_diff','CNN_Loss_Leaky_diff']].describe())
print(df[['abs_nearest_diff','abs_bilinear_diff', 'abs_rf_diff', \
          'abs_CNN_Norm_diff','abs_CNN_Stand_diff','abs_CNN_Loss_Leaky_diff']].describe())

# Calculate Biases
bias_nearest=np.mean(df['nearest']-df['w'])
bias_bilinear=np.mean(df['bilinear']-df['w'])
bias_rf=np.mean(df['rf']-df['w'])
bias_CNN_Norm=np.mean(df['CNN_Norm']-df['w'])
bias_CNN_Stand=np.mean(df['CNN_Stand']-df['w'])
bias_CNN_Loss_Leaky=np.mean(df['CNN_Loss_Leaky']-df['w'])

# Calculate MAE
mae_nearest=np.mean(abs(df['nearest']-df['w']))
mae_bilinear=np.mean(abs(df['bilinear']-df['w']))
mae_rf=np.mean(abs(df['rf']-df['w']))
mae_CNN_Norm=np.mean(abs(df['CNN_Norm']-df['w']))
mae_CNN_Stand=np.mean(abs(df['CNN_Stand']-df['w']))
mae_CNN_Loss_Leaky=np.mean(abs(df['CNN_Loss_Leaky']-df['w']))

# Calculate RMSEs
rmse_nearest=np.mean((df['nearest']-df['w'])**2)**0.5
rmse_bilinear=np.mean((df['bilinear']-df['w'])**2)**0.5
rmse_rf=np.mean((df['rf']-df['w'])**2)**0.5
rmse_CNN_Norm=np.mean((df['CNN_Norm']-df['w'])**2)**0.5
rmse_CNN_Stand=np.mean((df['CNN_Stand']-df['w'])**2)**0.5
rmse_CNN_Loss_Leaky=np.mean((df['CNN_Loss_Leaky']-df['w'])**2)**0.5

errors=np.array([[bias_nearest,bias_bilinear,bias_rf,bias_CNN_Norm,bias_CNN_Stand,bias_CNN_Loss_Leaky],
                 [rmse_nearest,rmse_bilinear,rmse_rf,rmse_CNN_Norm,rmse_CNN_Stand,rmse_CNN_Loss_Leaky],
                 [mae_nearest,mae_bilinear,mae_rf,mae_CNN_Norm,mae_CNN_Stand,mae_CNN_Loss_Leaky]])
errors=pd.DataFrame(errors, \
                    columns=['Nearest','Bilinear','RF','CNN_Norm','CNN_Stand','CNN_Loss_Leaky'], \
                    index=['Bias','RMSE','MAE'])
errors['RF_decr']=(abs(errors['RF'])-abs(errors['Nearest']))*-1
errors['Norm_decr']=(abs(errors['CNN_Norm'])-abs(errors['Nearest']))*-1
errors['Stand_decr']=(abs(errors['CNN_Stand'])-abs(errors['Nearest']))*-1
errors['LossLeaky_decr']=(abs(errors['CNN_Loss_Leaky'])-abs(errors['Nearest']))*-1
print(errors)

#########################
# Plot error bar charts #
#########################

width = 0.15
x=[0,1,2]
#xs=[[i-2*width,i-width,i,i+width,i+2*width] for i in x]

png='images/error_bars.png'
fig,ax=plt.subplots()
ax.bar([i-2*width for i in x], errors['Nearest'], width, label='Nearest')#, edgecolor='black')
ax.bar([i-width for i in x], errors['Bilinear'], width, label='Bilinear')#, edgecolor='black')
ax.bar(x, errors['RF'], width, label='RF')#, edgecolor='black')
ax.bar([i+width for i in x], errors['CNN_Norm'], width, label='CNN 1')#, edgecolor='black')
ax.bar([i+2*width for i in x], errors['CNN_Stand'], width, label='CNN 2')#, edgecolor='black')
ax.bar([i+3*width for i in x], errors['CNN_Loss_Leaky'], width, label='CNN 3')#, edgecolor='black')
ax.set_xticks(x)
ax.set_xticklabels(['Bias', 'RMSE', 'MAE'])
plt.title('Model Error Comparison')
plt.ylabel('Wind Speed (m/s)')
plt.ylim(-.5,2.5)
plt.axhline(0,c='black',linewidth=0.5)
plt.legend(ncol=6,bbox_to_anchor=(1,-.1),fontsize='small')
plt.savefig(png,dpi=300,bbox_inches='tight')
plt.close()

png='images/error_bars_nonorm.png'
fig,ax=plt.subplots()
ax.bar([i-2*width for i in x], errors['Nearest'], width, label='Nearest')#, edgecolor='black')
ax.bar([i-width for i in x], errors['Bilinear'], width, label='Bilinear')#, edgecolor='black')
ax.bar(x, errors['RF'], width, label='RF')#, edgecolor='black')
ax.bar([i+width for i in x], errors['CNN_Stand'], width, label='CNN_MSE_ReLu')#, edgecolor='black')
ax.bar([i+2*width for i in x], errors['CNN_Loss_Leaky'], width, label='CNN_Kernel_Leaky')#, edgecolor='black')
ax.set_xticks(x)
ax.set_xticklabels(['Bias', 'RMSE', 'MAE'])
plt.title('Model Error Comparison')
plt.ylabel('Wind Speed (m/s)')
plt.ylim(-.5,2.5)
plt.axhline(0,c='black',linewidth=0.5)
plt.legend()
plt.savefig(png,dpi=80)
plt.close()


################################################
# Plot Time series for each set of predictions #
################################################


print(df[['CNN_Stand','CNN_Loss_Leaky']].describe())
png='images/timeseries/CNN_comparison_time_series_test.png'
fig,axs=plt.subplots(nrows=2,figsize=(12,8))
df.plot(x='time',y=['w','CNN_Stand'],ax=axs[0],legend=False)
axs[0].set_title('CNN (MSE and ReLU) Test Predictions')
axs[0].set_xlabel('')
axs[0].set_ylabel('Wind Speed (m/s)')
axs[0].set_xticklabels([])
df.plot(x='time',y=['w','CNN_Loss_Leaky'],ax=axs[1],legend=False)
axs[1].set_title('CNN (Kernel MSE and Leaky ReLU)')
axs[1].set_xlabel('Time')
axs[1].set_ylabel('Wind Speed (m/s)')
axs[1].set_xticklabels(['Jan\n2017','March','May','July','Sept','Nov','Jan\n2018','March'])
plt.savefig(png,dpi=300,bbox_inches='tight')
plt.close()

png='images/timeseries/nearest_time_series_test.png'
fig,ax=plt.subplots(figsize=(12,4))
plt.title('Nearest NBS vs Buoy')
plt.xlabel('Time')
plt.ylabel('Wind Speed (m/s)')
df.plot(x='time',y=['w','nearest'],ax=ax)
plt.legend(fontsize='x-small')
plt.savefig(png,dpi=80)
plt.close()

png='images/timeseries/bilinear_time_series_test.png'
fig,ax=plt.subplots(figsize=(12,4))
plt.title('Bilinear Prediction vs Buoy')
plt.xlabel('Time')
plt.ylabel('Wind Speed (m/s)')
df.plot(x='time',y=['w','bilinear'],ax=ax)
plt.legend(fontsize='x-small')
plt.savefig(png,dpi=80)
plt.close()

png='images/timeseries/rf_time_series_test.png'
fig,ax=plt.subplots(figsize=(12,4))
plt.title('RF Prediction vs Buoy')
plt.xlabel('Time')
plt.ylabel('Wind Speed (m/s)')
df.plot(x='time',y=['w','rf'],ax=ax)
plt.legend(fontsize='x-small')
plt.savefig(png,dpi=80)
plt.close()

png='images/timeseries/CNN_Stand_time_series_test.png'
fig,ax=plt.subplots(figsize=(12,4))
plt.title('CNN Standardized Prediction vs Buoy')
plt.xlabel('Time')
plt.ylabel('Wind Speed (m/s)')
df.plot(x='time',y=['w','CNN_Stand'],ax=ax)
plt.legend(fontsize='x-small')
plt.savefig(png,dpi=80)
plt.close()

png='images/timeseries/CNN_Norm_time_series_test.png'
fig,ax=plt.subplots(figsize=(12,4))
plt.title('CNN Normalized Prediction vs Buoy')
plt.xlabel('Time')
plt.ylabel('Wind Speed (m/s)')
df.plot(x='time',y=['w','CNN_Norm'],ax=ax)
plt.legend(fontsize='x-small')
plt.savefig(png,dpi=80)
plt.close()

png='images/timeseries/CNN_Loss_Leaky_time_series_test.png'
fig,ax=plt.subplots(figsize=(12,4))
plt.title('CNN Kernel MSE and Leaky ReLu Prediction vs Buoy')
plt.xlabel('Time')
plt.ylabel('Wind Speed (m/s)')
df.plot(x='time',y=['w','CNN_Loss_Leaky'],ax=ax)
plt.legend(fontsize='x-small')
plt.savefig(png,dpi=80)
plt.close()



