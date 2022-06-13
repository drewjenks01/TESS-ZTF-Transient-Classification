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
import math
from tqdm import tqdm

filepath='/Users/drewj/Documents/Urops/Muthukrishna/data/'
lc_files=next(walk('/Users/drewj/Documents//Urops/Muthukrishna/data/light_curves_fausnaugh'), (None, None, []))[2]


def read_data(save=True):
    """
    Loads all data files into Panda Dataframes.
    Light curves are a numpy array full of dataframes.
    """

    print("reading raw data...")

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
    for f in tqdm(lc_files):
        df=pd.read_csv(filepath+'light_curves_fausnaugh/'+f,delim_whitespace=True)
        df['filename'] = f
        light_curves.append(df)

    if save:
        print("saving raw data...")

        #light_curves=np.array(light_curves)

        all_transients.to_pickle(filepath+'all_transients.pkl')
        supernovae.to_pickle(filepath+'supernovae.pkl')
        
        with open(filepath+'light_curves.pkl',"wb") as f:
            pkl.dump(light_curves,f)



def load_data():
    """
    Loads data from saved pickle files
    """
    print("loading raw data..")

    all_transients=pd.read_pickle(filepath+'all_transients.pkl')
    supernovae=pd.read_pickle(filepath+'supernovae.pkl')

    with open(filepath+'light_curves.pkl',"rb") as f:
        light_curves=pkl.load(f)


    return {'all_transients':all_transients,'supernovae':supernovae,'light_curves':light_curves}



#TODO: binned_data as np.array or set
def bin_data(data,save=True):
    """
    Bins light curves to 1 day intervals. Light curves come in as either 10 or 30 minute intervals

    binned_data.shape = (3857,2, 13-30)
    """
    print("binning data...")

    binned_data=[]

    for lc in tqdm(data):

        #extract time and brightness vals
        time=lc.loc[:,'BTJD'].to_numpy()
        flux=lc.loc[:,'cts'].to_numpy()



        #starting time
        curr_time=math.floor(time[0])

        #new binned arrays
        binned_time=[curr_time]
        binned_flux=[]

        #sub-interval sums
        flux_sum=0.0
        count=0

        for t in range(len(time)):

            #if time goes to next day then start new bin interval
            if math.floor(time[t])!=curr_time:

                #set new current day
                curr_time=math.floor(time[t])
                binned_time.append(curr_time)

                #append binned flux data
                binned_flux.append(flux_sum/count)
                flux_sum=0
                count=0

            flux_sum += flux[t]
            count+=1


        #append the leftover
        binned_flux.append(flux_sum/count)

        binned_data.append( np.array([binned_time,binned_flux]))

   
    # binned_data=np.array(binned_data)
    print(len(binned_data),binned_data[1].shape)


    if save:
        print('saving binned...')

        with open(filepath+'Binned_light_curves.pkl',"wb") as f:
            pkl.dump(binned_data,f)


    return binned_data


def load_binned():
    """
    Load binned data from pkl file
    """
    print("loading binned data...")

    with open(filepath+'binned_light_curves.pkl',"rb") as f:
        binned_light_curves=pkl.load(f)

    return binned_light_curves



def augment_binned_data(data,save=True):

    """
    Takes binned data and augments it so that each light curve
    has length of 29 days. Padded numbers assigned val = -1

    aug_data.shape = (3857,2,30)
    """
    print("augmenting data...")
    data=data.copy()
    pad=np.array([[-1,-1]])
    for i in tqdm(range(len(data))):
        while len(data[i][0])!=30:
            data[i]=np.concatenate((data[i],pad.T),axis=1)

         #check new shape of tensor
        #print(data[i].shape)

    aug_data=np.array(data)
    
    if save:
        print('saving augmented data...')

        with open(filepath+'aug_binned_light_curves.pkl',"wb") as f:
            pkl.dump(aug_data,f)

    return aug_data

def load_augmented():
    """
    Load augmented, binned data file form pkl
    """
    print("loading augmented data...")

    with open(filepath+'aug_binned_light_curves.pkl',"rb") as f:
        aug_binned_light_curves=pkl.load(f)

    return aug_binned_light_curves


def plot_data(data):
    x=data[0][:28]
    y=data[1][:28]


    plt.plot(x,y)
    #plt.title(data.loc[:,"filename"][0])

    plt.show()




def main():

    #load saved data
    data=load_data()
    data=data['light_curves']

    # x=data[92].loc[:,'BTJD']
    # y=data[92].loc[:,'cts']
    # print(y)
    # name=data[92].loc[:,'filename'][0]
    # print(name)

    plt.plot(x,y)
    plt.show()

    #bin_data(data)

    binned_data=load_binned()
    
    aug_data= load_augmented()


    lengths= [d[0].shape for d in binned_data]
    aug_lengths= [d[0].shape for d in aug_data]

    #print(Counter(data))
    print(Counter(lengths))
    print(Counter(aug_lengths))

    # for lc in binned_data[92]:

    plot_data(binned_data[92])

    print(binned_data[92])


   # print(binned_data)







if __name__=='__main__':
    main()

# %%





