import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

#inputdir='data/results/train'
inputdir='data'
oob=pd.read_csv(f'{inputdir}/rf_oob.csv')
print(oob)

min_samples_leaf=[5,10,15,20,25,30,35]
ntree=[20,40,60,80,100,200,300,400,500,600,700]
c=['C0','C1','C2','C4','C5','C7','C8']#,'C9']
png=f'images/hyperparameters_OOB_score.png'
fig,axs=plt.subplots(nrows=1,ncols=5,figsize=(12,3))
for i in range(5):
    fold=oob.loc[oob['fold']==i]
    for j, leaves in enumerate(min_samples_leaf):
        dat_lf=fold.loc[fold['min_samples']==leaves].sort_values('ntree')
        axs[i].plot(dat_lf['ntree'],dat_lf['oob_error'],label=leaves,c=c[j]) 
    if i==0:
        axs[i].set_ylabel('OOB Error')
    axs[i].set_title(f'Fold {i}')
    axs[i].set_xlabel('Number of Trees')
    axs[i].xaxis.set_ticks(np.arange(100,800,200))
    axs[i].xaxis.set_tick_params(labelsize='small')
    #axs[i].xaxis.set_major_locator(MultipleLocator(500)) # create major x ticks at every 5 years
    #axs[i].xaxis.set_minor_locator(MultipleLocator(100)) # create minor x ticks at every 1 year
    #axs[i].xaxis.set_ticklabels([0,5,10,15,20])
    axs[i].grid()
    #axs[4].legend(bbox_to_anchor=(1.15,.85),title='MinLeafSize',title_fontsize='small',ncol=6)
    axs[2].legend(bbox_to_anchor=(2.5,-.2),title='MinLeafSize',title_fontsize='small',ncol=7)
    #axs[4].legend(bbox_to_anchor=(1.25,.85))
plt.subplots_adjust(wspace=.5)
#plt.tight_layout()
plt.savefig(png,bbox_inches='tight',dpi=300)
plt.close()
