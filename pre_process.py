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

filepath='/Users/drewj/Documents/Urops/Muthukrishna/'
lc_files=next(walk('/Users/drewj/Documents//Urops/Muthukrishna/data/processed_curves'), (None, None, []))[2]


def read_data(save=True):
    """
    Loads all data files into Panda Dataframes.
    Light curves are a numpy array full of dataframes.
    """

    print("reading raw data...")

    #extracts data from all transients file
    all_transients=pd.read_csv(filepath+'data/all_transients.txt',delim_whitespace=True)
    all_transients.columns=['sector','ra','dec','mag at discovery','time of discovery','type of transient',
    'classification','IAU name','discovery survey','cam','ccd','column','row']

    #extracts data from confirmed supernovae file
    supernovae=pd.read_csv(filepath+'data/supernovae.txt',delim_whitespace=True)
    supernovae.columns=['sector','ra','dec','mag at discovery','time of discovery','type of transient',
    'classification','IAU name','discovery survey','cam','ccd','column','row']
   
    #reads through all light curve files and appends each df to a list
    #TODO: make into a set for O(1) accessing?
    light_curves=[]
    for f in tqdm(lc_files):
        df=pd.read_csv(filepath+'data/processed_curves/'+f)
        df['filename'] = f
        light_curves.append(df)

    if save:
        print("saving raw data...")

        #light_curves=np.array(light_curves)

        all_transients.to_pickle(filepath+'data/all_transients.pkl')
        supernovae.to_pickle(filepath+'data/supernovae.pkl')
        
        with open(filepath+'data/light_curves.pkl',"wb") as f:
            pkl.dump(light_curves,f)



def load_data():
    """
    Loads data from saved pickle files
    """
    print("loading raw data..")

    all_transients=pd.read_pickle(filepath+'data/all_transients.pkl')
    supernovae=pd.read_pickle(filepath+'data/supernovae.pkl')

    with open(filepath+'data/light_curves.pkl',"rb") as f:
        light_curves=pkl.load(f)


    return {'all_transients':all_transients,'supernovae':supernovae,'light_curves':light_curves}



#TODO: binned_data as np.array or set
def bin_data(data,save=True):
    """
    Bins light curves to 1 day intervals. Light curves come in as either 10 or 30 minute intervals

    binned_data.shape = (3857,3, 13-30)
    """
    print("binning data...")

    binned_data=[]

    for lc in tqdm(data):

        #extract time and brightness vals
        time=lc.loc[:,'relative_time']
        flux=lc.loc[:,'cts']
        error=lc.loc[:,'e_cts']


        #starting time
        curr_time=math.floor(time[0])

        #new binned arrays
        binned_time=[curr_time]
        binned_flux=[]
        binned_error=[]

        #sub-interval sums
        flux_sum=0.0
        count=0
        sub_error=np.array([])

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

                #append binned error (root mean square)
                binned_error.append(np.sqrt(np.mean(sub_error**2)))
                sub_error=[]

            flux_sum += flux[t]
            count+=1
            sub_error=np.append(sub_error,error[t])



        #append the leftover
        binned_flux.append(flux_sum/count)
        binned_error.append(np.sqrt(np.mean(sub_error**2)))


       # print('shapes (data,time,flux,err): ',binned_data.shape,len(binned_time),len(binned_flux),len(binned_error))
        binned_data.append(np.array([binned_time,binned_flux,binned_error]))
       # print(binned_data)

   
    # binned_data=np.array(binned_data)
    print('binned data shape[1]: ',binned_data[1].shape)


    if save:
        print('saving binned...')

        with open(filepath+'data/Binned_light_curves.pkl',"wb") as f:
            pkl.dump(binned_data,f)


    return binned_data


def load_binned():
    """
    Load binned data from pkl file
    """
    print("loading binned data...")

    with open(filepath+'data/binned_light_curves.pkl',"rb") as f:
        binned_light_curves=pkl.load(f)

    return binned_light_curves



def augment_binned_data(data,save=True):

    """
    Takes binned data and augments it so that each light curve
    has length of 29 days. Padded numbers assigned val = -1

    aug_data.shape = (3857,3,30)
    """
    print("augmenting data...")
    print('old data shape[1]: ',data[1].shape)
    data=data.copy()
    pad=np.array([[-1,-1,-1]])

    print(len(data[1]))
    for i in tqdm(range(len(data))):
        while len(data[i][0])!=30:
            data[i]=np.concatenate((data[i],pad.T),axis=1)

         #check new shape of tensor
        #print(data[i].shape)

    aug_data=np.array(data)
    print('aug data shape: ',aug_data.shape)
    
    if save:
        print('saving augmented data...')

        with open(filepath+'data/aug_binned_light_curves.pkl',"wb") as f:
            pkl.dump(aug_data,f)

    return aug_data

def load_augmented():
    """
    Load augmented, binned data file form pkl
    """
    print("loading augmented data...")

    with open(filepath+'data/aug_binned_light_curves.pkl',"rb") as f:
        aug_binned_light_curves=pkl.load(f)

    return aug_binned_light_curves


def plot_binned_data(raw_data, binned_data):

    plt.figure()

    title=raw_data.loc[:,'filename'][0]

    #raw data
    plt.errorbar(raw_data.loc[:,'relative_time'],raw_data.loc[:,'cts'],fmt='.',color='r',alpha=0.5)

    #binned data
    plt.errorbar(binned_data[0],binned_data[1],binned_data[2],color='b')

    plt.title(title)

    plt.legend(['raw','binned'])

    #save image
    plt.savefig(filepath+'plots/raw_err_'+title+'.png',facecolor='white')

    #show image
    plt.show()





def main():

    #read raw data
   # read_data()

    #load saved data
    data=load_data()
    data=data['light_curves']

    #bin data
   # bin_data(data)

    #load binned data
    binned_data=load_binned()

    #augment_binned_data(binned_data)
    
    aug_data= load_augmented()

    #print(aug_data[1][0])


    lengths= [d[0].shape for d in binned_data]
    aug_lengths= [d[0].shape for d in aug_data]

    #print(Counter(data))
    print(Counter(lengths))
    print(Counter(aug_lengths))

    plot_binned_data(data[1], binned_data[1])








if __name__=='__main__':
    main()

# %%





