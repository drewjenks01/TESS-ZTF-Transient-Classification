"""
This file is used for reading in data and making initial plots.
"""
#%%

import pandas as pd
import numpy as np
from os import walk
import matplotlib.pyplot as plt
import pickle as pkl
import os

filepath='/Users/drewj/Documents/Urops/Muthukrishna/Tess-transient-classification/data/'
lc_files=next(walk('/Users/drewj/Documents//Urops/Muthukrishna/Tess-transient-classification/data/light_curves_fausnaugh'), (None, None, []))[2]


def read_data(save=False):
    """
    Loads all data files into Panda Dataframes with option to save as pickle fikes.
    """

    #extracts data from all transients file
    all_transients=pd.read_csv(filepath+'all_transients.txt',delim_whitespace=True)
    all_transients.columns=['sector','ra','dec','mag at discovery','time of discovery','type of transient',
    'classification','IAU name','discovery survey','cam','ccd','column','row']

    #extracts data from confirmed supernovae file
    supernovae=pd.read_csv(filepath+'supernovae.txt',delim_whitespace=True)
    supernovae.columns=['sector','ra','dec','mag at discovery','time of discovery','type of transient',
    'classification','IAU name','discovery survey','cam','ccd','column','row']
   
    #reads through all light curve files and appends each df to a list
    #TODO: make into a set for O(1) accessing?
    light_curves=[]
    for f in lc_files:
        df=pd.read_csv(filepath+'light_curves_fausnaugh/'+f,delim_whitespace=True)
        df['filename'] = f
        light_curves.append(df)


    light_curves=np.array(light_curves)
    #print(light_curves)
   #save as pickle files
    if save:
        all_transients.to_pickle(filepath+'all_transients.pkl')
        supernovae.to_pickle(filepath+'supernovae.pkl')
        
        with open(filepath+'light_curves.pkl',"wb") as f:
            pkl.dump(light_curves,f)

    return [all_transients,supernovae,light_curves]



def load_data():
    """
    Loads data from saved pickle files
    """

    all_transients=pd.read_pickle(filepath+'all_transients.pkl')
    supernovae=pd.read_pickle(filepath+'supernovae.pkl')

    with open(filepath+'light_curves.pkl',"rb") as f:
        light_curves=pkl.load(f)

    return [all_transients,supernovae,light_curves]






def plot_data(data):
    x=data.loc[:,"BTJD"]
    y=data.loc[:,"cts"]


    plt.plot(x,y)
    plt.title(data.loc[:,"filename"][0])

    plt.show()




def main():


    data=load_data()[2]
    ex=data[0]
    print(ex.loc[:,'BTJD'])





if __name__=='__main__':
    main()

# %%





