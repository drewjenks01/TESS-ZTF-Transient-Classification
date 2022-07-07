"""
This file is used for reading in data and making initial plots.
"""
# %%

import pandas as pd
import numpy as np
from os import walk
import matplotlib.pyplot as plt
import pickle as pkl
import os
from collections import Counter
import math
from sklearn.model_selection import train_test_split
from tqdm import tqdm

filepath = '/Users/drewj/Documents/Urops/Muthukrishna/'
lc_files = next(walk(
    '/Users/drewj/Documents//Urops/Muthukrishna/data/processed_curves'), (None, None, []))[2]


def read_data(save=True):
    """
    Loads all data files into Panda Dataframes.
    Light curves are a numpy array full of dataframes.
    """

    print("reading raw data...")

    # extracts data from all transients file
    all_transients = pd.read_csv(
        filepath+'data/all_transients.txt', delim_whitespace=True)
    all_transients.columns = ['sector', 'ra', 'dec', 'mag at discovery', 'time of discovery', 'type of transient',
                              'classification', 'IAU name', 'discovery survey', 'cam', 'ccd', 'column', 'row']

    print(len(all_transients.loc[:, 'mag at discovery']))

    # extracts data from confirmed supernovae file
    supernovae = pd.read_csv(
        filepath+'data/supernovae.txt', delim_whitespace=True)
    supernovae.columns = ['sector', 'ra', 'dec', 'mag at discovery', 'time of discovery', 'type of transient',
                          'classification', 'IAU name', 'discovery survey', 'cam', 'ccd', 'column', 'row']

    # reads through all light curve files and appends each df to a list
    # TODO: make into a set for O(1) accessing?
    light_curves = []
    for f in tqdm(lc_files):
        df = pd.read_csv(filepath+'data/processed_curves/'+f)
        df['filename'] = f
        light_curves.append(df)

    if save:
        print("saving raw data...")

        # light_curves=np.array(light_curves)

        all_transients.to_pickle(filepath+'data/all_transients.pkl')
        supernovae.to_pickle(filepath+'data/supernovae.pkl')

        with open(filepath+'data/light_curves.pkl', "wb") as f:
            pkl.dump(light_curves, f)


def load_data():
    """
    Loads data from saved pickle files
    """
    print("loading raw data..")

    all_transients = pd.read_pickle(filepath+'data/all_transients.pkl')
    supernovae = pd.read_pickle(filepath+'data/supernovae.pkl')

    with open(filepath+'data/light_curves.pkl', "rb") as f:
        light_curves = pkl.load(f)

    return {'all_transients': all_transients, 'supernovae': supernovae, 'light_curves': light_curves}


def bin_data(data, save=True, normalize=False):
    """
    Bins light curves to 1 day intervals. Light curves come in as either 10 or 30 minute intervals

    binned_data.shape = (3857,13-30,3) -> time, flux, error
    """
    print("binning data...")

    binned_data = []
    rmin = -1.0
    rmax = 1.0

    for lc in tqdm(data):

        # extract time and brightness vals
        time = np.array(lc.loc[:, 'relative_time'])
        flux = np.array(lc.loc[:, 'cts'])
        error = np.array(lc.loc[:, 'e_cts'])

       # print('f1:' ,flux)

        # normalize b/w [-1,1]
        if normalize:
            flux = 2 * (flux-np.min(flux))/(np.max(flux)-np.min(flux)) - 1
            error = 2 * (error-np.min(error))/(np.max(error)-np.min(error)) - 1
           # print('f2: ', flux)

        # starting time
        curr_time = math.floor(time[0])

        # new binned arrays
        binned_time_flux_err = []

        # sub-interval sums
        flux_sum = 0.0
        count = 0
        sub_error = np.array([])

        for t in range(len(time)):

            # if time goes to next day then start new bin interval
            if math.floor(time[t]) != curr_time:

                binned_time_flux_err.append([curr_time, flux_sum/count,
                                             np.sqrt(np.mean(sub_error**2))])

                # set new current day
                curr_time = math.floor(time[t])

                # append binned flux data
                flux_sum = 0
                count = 0

                sub_error = np.array([])

            flux_sum += flux[t]
            count += 1
            sub_error = np.append(sub_error, error[t])

        # append the leftover
        binned_time_flux_err.append(
            [curr_time, flux_sum/count, np.sqrt(np.mean(sub_error**2))])

        binned_time_flux_err = np.array(binned_time_flux_err)

        binned_data.append(binned_time_flux_err)

    # binned_data=np.array(binned_data)
    print('binned data shape[1]: ', binned_data[1].shape)

    if save:
        print('saving binned...')

        with open(filepath+'data/Binned_light_curves.pkl', "wb") as f:
            pkl.dump(binned_data, f)

    return binned_data


def load_binned():
    """
    Load binned data from pkl file
    """
    print("loading binned data...")

    with open(filepath+'data/binned_light_curves.pkl', "rb") as f:
        binned_light_curves = pkl.load(f)

    return binned_light_curves


def augment_binned_data(data, save=True):
    """
    Takes binned data and augments it so that each light curve
    has length of 29 days. Padded numbers assigned val = -1

    aug_data.shape = (3857,30,3) -> time, flux, error
    """
    print("augmenting data...")
    print('old data shape[1]: ', data[1].shape)
    data = data.copy()
    pad = np.array([[0.0, 0.0, 0.0]])

    print('len of data[1] should be 29: ', len(data[1]))
    for i in tqdm(range(len(data))):
        while len(data[i]) != 30:
            data[i] = np.concatenate((data[i], pad), axis=0)

         # check new shape of tensor
        # print(data[i].shape)

    aug_data = np.array(data)
    print('aug data shape: ', aug_data.shape)

    if save:
        print('saving augmented data...')

        with open(filepath+'data/aug_binned_light_curves.pkl', "wb") as f:
            pkl.dump(aug_data, f)

    return aug_data


def load_augmented():
    """
    Load augmented, binned data file form pkl
    """
    print("loading augmented data...")

    with open(filepath+'data/aug_binned_light_curves.pkl', "rb") as f:
        aug_binned_light_curves = pkl.load(f)

    return aug_binned_light_curves


def plot_binned_data(raw_data, binned_data):

    plt.figure()

    title = raw_data.loc[:, 'filename'][0]

    # raw data
    plt.errorbar(raw_data.loc[:, 'relative_time'],
                 raw_data.loc[:, 'cts'], fmt='.', color='r', alpha=0.5)

    # binned data
    plt.errorbar(binned_data[:, 0], binned_data[:, 1],
                 binned_data[:, 2], color='b')

    plt.title(title)

    plt.legend(['raw', 'binned'])

    # save image
    plt.savefig(filepath+'plots/raw_err_'+title+'.png', facecolor='white')

    # show image
    plt.show()


def concatFeature(raw_data, feat_name):
    """
    Takes in binned data and concats a new feature from data to binned data.
    *Should be performed before augmenting*

    binned_inp: (3857,13-30,n)
    binned_out: (3857,13-30,n+1)
    """
    print("adding feature: ", feat_name, " to data...")


def main():

    # read raw data
    # read_data()

    # load saved data
    data = load_data()
    data = data['light_curves']

    # bin data
    bin_data(data, normalize=True)

    # load binned data
    binned_data = load_binned()

    augment_binned_data(binned_data)

    aug_data = load_augmented()


if __name__ == '__main__':
    main()

# %%
