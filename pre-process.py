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
from collections import Counter

filepath='/Users/drewj/Documents/Urops/Muthukrishna/data/'
lc_files=next(walk('/Users/drewj/Documents//Urops/Muthukrishna/data/light_curves_fausnaugh'), (None, None, []))[2]


def read_data(save=True):
    """
    Loads all data files into Panda Dataframes.
    Light curves are a numpy array full of dataframes.
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

    if save:

        light_curves=np.array(light_curves)

        all_transients.to_pickle(filepath+'all_transients.pkl')
        supernovae.to_pickle(filepath+'supernovae.pkl')
        
        with open(filepath+'light_curves.pkl',"wb") as f:
            pkl.dump(light_curves,f)



def save_data(data):
    """
    Saves data in pkl files. Should come in as [all_transients, supernovae, light curves]
    """
    data[0].to_pickle(filepath+'all_transients.pkl')
    data[1].to_pickle(filepath+'supernovae.pkl')
    
    with open(filepath+'light_curves.pkl',"wb") as f:
        pkl.dump(data[2],f)



def load_data():
    """
    Loads data from saved pickle files
    """

    all_transients=pd.read_pickle(filepath+'all_transients.pkl')
    supernovae=pd.read_pickle(filepath+'supernovae.pkl')

    with open(filepath+'light_curves.pkl',"rb") as f:
        light_curves=pkl.load(f)


    return {'all_transients':all_transients,'supernovae':supernovae,'light_curves':light_curves}


def load_binned():
    """
    Load binned data from pkl file
    """

    with open(filepath+'binned_light_curves.pkl',"rb") as f:
        binned_light_curves=pkl.load(f)

    return binned_light_curves



#TODO: binned_data as np.array or set
def bin_data(data,save=True):
    """
    Bins light curves to 1 day intervals. Light curves come in as either 10 or 30 minute intervals
    """

    binned_data=[]

    thirty_minute=30.0/1440
    ten_minute=10.0/1440

    for lc in data:

        #extract time and brightness vals
        time=lc.loc[:,'BTJD'].to_numpy()
        flux=lc.loc[:,'cts'].to_numpy()



        #starting time
        curr_time=round(time[0])

        #new binned arrays
        binned_time=[curr_time]
        binned_flux=[]

        #sub-interval sums
        flux_sum=0.0
        count=1

        for t in range(len(time)):

            #if time goes to next day then start new bin interval
            if round(time[t])!=curr_time:

                #set new current day
                curr_time=round(time[t])
                binned_time.append(curr_time)

                #append binned flux data
                binned_flux.append(flux_sum/count)
                flux_sum=0
                count=0

            flux_sum += flux[t]
            count+=1


        #append the leftover
        binned_flux.append(flux_sum)

        binned_data.append( np.array([binned_time,binned_flux]))

   # binned_data=np.array(binned_data)


    if save:

        with open(filepath+'Binned_light_curves.pkl',"wb") as f:
            pkl.dump(binned_data,f)



    return binned_data




def plot_data(data):
    x=data.loc[:,"BTJD"]
    y=data.loc[:,"cts"]


    plt.plot(x,y)
    plt.title(data.loc[:,"filename"][0])

    plt.show()




def main():

    #read and save data (only has to be done once)
  #  read_data()

    #load saved data
    data=load_data()
    data=data['light_curves']

    binned_data=load_binned()

    lengths= [d[0].shape[0] for d in binned_data]
    print(Counter(lengths))


   # print(binned_data)







if __name__=='__main__':
    main()

# %%





