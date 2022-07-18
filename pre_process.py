"""
This file is used for reading in data and making initial plots.

To do:
    - implement 1/2 day bins
    - normalize mag?

Qs:
    - normalize before or after binned 
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
import random

filepath = '/Users/drewj/Documents/Urops/Muthukrishna/'
lc_files = next(walk(
    '/Users/drewj/Documents//Urops/Muthukrishna/data/processed_curves'), (None, None, []))[2]


def read_raw_data(save=True):
    """
    Loads all data files into Panda Dataframes.
    Light curves are a numpy array full of dataframes.
    """

    print("reading raw data...")

    # extracts data from all transients file
    all_transients = pd.read_csv(
        filepath+'data/all_transients.txt', header=None, delim_whitespace=True)
    all_transients.columns = ['sector', 'ra', 'dec', 'mag at discovery', 'time of discovery', 'type of transient',
                              'classification', 'IAU name', 'discovery survey', 'cam', 'ccd', 'column', 'row']

    #print(all_transients.loc[:,'IAU name'])

    # reads through all light curve files and appends each df to a list
    light_curves = []
    for f in tqdm(lc_files):
        df = pd.read_csv(filepath+'data/processed_curves/'+f)
        df['filename'] = f
        light_curves.append(df)

    if save:
        print("saving raw data...")

        # light_curves=np.array(light_curves)

        all_transients.to_pickle(filepath+'data/all_transients.pkl')

        with open(filepath+'data/light_curves.pkl', "wb") as f:
            pkl.dump(light_curves, f)


def load_raw_data():
    """
    Loads data from saved pickle files
    """
    print("loading raw data..")

    all_transients = pd.read_pickle(filepath+'data/all_transients.pkl')
    supernovae = pd.read_pickle(filepath+'data/supernovae.pkl')

    with open(filepath+'data/light_curves.pkl', "rb") as f:
        light_curves = pkl.load(f)

    return {'all_transients': all_transients, 'light_curves': light_curves}


def create_raw_dataframe(save=True):
    """
    Create DataFrame used for NN+classifier

    DF: (Filename,Time, Flux, Error, Mag at Discovery, Class)
    """
    print('creating raw dataframe')

    light_curves = load_raw_data()['light_curves']
    all_transients = load_raw_data()['all_transients']

    df_dict = []

    # define columns
    columns = ['Filename', 'Time', 'Flux',
               'Error', 'Mag at Discovery', 'Class']

    # loop through AT and grab necessary data
    at_files = all_transients.loc[:, 'IAU name']
    at_class = all_transients.loc[:, 'classification']
    at_mags = all_transients.loc[:, 'mag at discovery']

    # change classificiations into numbers
    classes = {}
    count = 0

    # collect unique classifications, assign a number
    for c in at_class:
        if c not in classes:
            classes[c] = count
            count += 1

    print('unique classifiers: ', classes)

    at_dict = {}

    for i in range(len(at_files)):
        at_dict[at_files[i]] = (at_mags[i], at_class[i])

    # loop through light_curve file
    for lc in tqdm(light_curves):
        # extract filename, flux, error, time
        flux = np.array(lc.loc[:, 'cts'])
        error = np.array(lc.loc[:, 'e_cts'])
        filename = lc.loc[:, 'filename'][0]
        time = np.array(lc.loc[:, 'relative_time'])

        # process filename to be just abreviation ex: 2018evo
        name_split = filename.split('_')
        name_abr = name_split[1]

        # grab mag and class
        mag = at_dict[name_abr][0]
        classif = classes[at_dict[name_abr][1]]

        # add data to DataFrame dict list
        df_dict.append({'Filename': name_abr, 'Time': time, 'Flux': flux, 'Error': error,
                        'Mag at Discovery': mag, 'Class': classif})

    # create dataframe
    df = pd.DataFrame(df_dict, columns=columns)

    if save:
        print('saving raw DF')
        # save dataframe
        df.to_pickle(filepath+'data/raw_df.pkl')

    return df


def load_raw_dataframe():
    print('loading raw df...')
    return pd.read_pickle(filepath+'data/raw_df.pkl')


def create_binned_dataframe(save=True):
    """
    Creates Binned DataFrame.

    Bins light curves to 0.5 day intervals.

    binned_data.shape = (3857,13-30,6) -> filename, time, flux, error, mag, class
    """
    print("binning data...")

    # deep copy of old df w/out outliers
    old_df = remove_outliers().copy()

    # define columns
    columns = ['Filename', 'Time', 'Flux',
               'Error', 'Mag at Discovery', 'Class']

    binned_dicts = []

    # extract features to be binned
    lc_flux = old_df.loc[:, 'Flux']
    lc_error = old_df.loc[:, 'Error']
    lc_time = old_df.loc[:, 'Time']
    lc_name = old_df.loc[:, 'Filename']
    lc_class = old_df.loc[:, 'Class']
    lc_mag = old_df.loc[:, 'Mag at Discovery']

    # loop through each light curve to bin
    for i in tqdm(range(len(lc_flux))):

        # extract time, flux, error vals and singular name, classification, mag
        flux = lc_flux[i]
        error = lc_error[i]
        time = lc_time[i]
        name_abr = lc_name[i]
        classif = lc_class[i]
        mag = lc_mag[i]

        # normalize flux, error, and mag b/w [-1,1]
        flux = 2 * (flux-np.min(flux))/(np.max(flux)-np.min(flux)) - 1
        error = 2 * (error-np.min(error))/(np.max(error)-np.min(error)) - 1
        #mag = 2 * (mag-np.min(flux))/(np.max(flux)-np.min(flux)) - 1

        if i == 0:
            print('len of raw time,flux, error, mag: ',
                  time.size, flux.size, error.size)

        # starting time
        curr_time = math.floor(time[0])

        # new binned arrays
        binned_features = {'time': [], 'flux': [], 'error': []}

        # sub-interval sums
        flux_sum = 0.0
        count = 0
        sub_error = np.array([])

        for t in range(len(time)):

            # if time goes to next day then start new bin interval
            if math.floor(time[t]) != curr_time:

                flux_mean = flux_sum/count
                err_bin = np.sqrt(np.mean(sub_error**2))

                # append binned vals
                binned_features['time'].append(curr_time)
                binned_features['flux'].append(flux_mean)
                binned_features['error'].append(err_bin)

                # set new current day
                curr_time = math.floor(time[t])

                # reset bins
                flux_sum = 0.0
                count = 0
                sub_error = np.array([])

            flux_sum += flux[t]
            count += 1
            sub_error = np.append(sub_error, error[t])

        # append the leftover
        flux_mean = flux_sum/count
        err_bin = np.sqrt(np.mean(sub_error**2))

        # make numpy arrays
        binned_features['time'] = np.array(binned_features['time'])
        binned_features['flux'] = np.array(binned_features['flux'])
        binned_features['error'] = np.array(binned_features['error'])

        # add data to DataFrame dict list
        binned_dicts.append({'Filename': name_abr, 'Time': binned_features['time'],
                             'Flux': binned_features['flux'], 'Error': binned_features['error'],
                            'Mag at Discovery': mag, 'Class': classif})

        # check time test
        if i == 0:
            print('len of binned time,flux, error: ', len(binned_dicts[i]['Time']),
                  len(binned_dicts[i]['Flux']), len(binned_dicts[i]['Error']))

    # create dataframe
    binned_df = pd.DataFrame(binned_dicts, columns=columns)

    if save:
        print('saving binned df...')
        # save dataframe
        binned_df.to_pickle(filepath+'data/binned_df.pkl')

    return binned_df


def load_binned_dataframe():
    print('loading binned df...')
    return pd.read_pickle(filepath+'data/binned_df.pkl')


def create_aug_dataframe(save=True):
    """
    Creates a DataFrame for augmented data.

    Takes binned data and augments it so that each light curve
    has length of 29 days. Padded numbers assigned val = 0.0

    aug_data.shape = (3857,30,6) -> filename, time, flux, error, mag, class
    """
    print("augmenting data...")

    # deep copy of binned dataframe
    old_df = load_binned_dataframe().copy()

    # define columns
    columns = ['Filename', 'Time', 'Flux',
               'Error', 'Mag at Discovery', 'Class']

    aug_dicts = []

    # extract features to be binned
    lc_flux = old_df.loc[:, 'Flux']
    lc_error = old_df.loc[:, 'Error']
    lc_time = old_df.loc[:, 'Time']
    lc_name = old_df.loc[:, 'Filename']
    lc_class = old_df.loc[:, 'Class']
    lc_mag = old_df.loc[:, 'Mag at Discovery']

    # array to concat
    concat = np.array([0.0])

    # loop through each light curve to bin
    for i in tqdm(range(len(lc_flux))):

        # extract flux, error, and mag vals
        flux = np.array(lc_flux[i])
        error = np.array(lc_error[i])
        mag = np.array(lc_mag[i])
        name_abr = lc_name[i]
        classif = lc_class[i]
        time = lc_time[i]

        if i == 0:
            print('old len of flux, error, mag: ',
                  flux.size, error.size, mag.size)

        old_length = len(flux)

        # concat 0.0's until each feature has len = 30
        while (flux.size != 30):
            flux = np.concatenate((flux, concat))

        while (error.size != 30):
            error = np.concatenate((error, concat))

        mag = np.repeat(mag, old_length)

        while (mag.size != 30):
            mag = np.concatenate((mag, concat))

        # add data to DataFrame dict list
        aug_dicts.append({'Filename': name_abr, 'Time': time,
                          'Flux': flux, 'Error': error,
                          'Mag at Discovery': mag, 'Class': classif})

        # check lengths test
        if i == 0:
            check = aug_dicts[i]
            print('new len of flux, error, mag: ', len(check['Flux']), len(check['Error']),
                  len(check['Mag at Discovery']))

    # create dataframe
    aug_df = pd.DataFrame(aug_dicts, columns=columns)

    if save:
        print('saving augmented df...')
        # save dataframe
        aug_df.to_pickle(filepath+'data/aug_df.pkl')

    return aug_df


def load_aug_dataframe():
    print('loading aug df...')
    return pd.read_pickle(filepath+'data/aug_df.pkl')


def plot_random_binned_data():

    # load binned and raw df
    raw_df = load_raw_dataframe()
    binned_df = load_binned_dataframe()

    # number of lc's
    num_lc = raw_df.shape[0]
    print('number of lcs (should match): ', num_lc, binned_df.shape[0])

    # choose 5 random lc's
    for _ in range(5):

        indx = random.randint(0, num_lc-1)

        print('indx: ', indx)

        raw_lc = raw_df.iloc[indx]
        binned_lc = binned_df.iloc[indx]

       # print(raw_lc['Filename'])

        plt.figure()

        title = raw_lc['Filename']

        print(len(raw_lc['Time']), len(raw_lc['Flux']))

        # raw data
        plt.errorbar(list(raw_lc['Time']),
                     raw_lc['Flux'], fmt='.', color='r', alpha=0.5)

        # binned data
        plt.errorbar(binned_lc['Time'], binned_lc['Flux'],
                     binned_lc['Error'], color='b')

        plt.title(title)

        plt.legend(['raw', 'binned'])

        # save image
        plt.savefig(filepath+'plots/raw_err_'+title+'.png', facecolor='white')

        # show image
        plt.show()


def plot_specific(df_type, filename):
    """
    Plots the data of a specific light curve

    Args:
        df_type (str): type of dataframe to extract lc from (raw, binned, aug)
        filename (str): filename abreviation of the light curve to plot
    """
    if df_type == 'raw':
        df = load_raw_dataframe()

    elif df_type == 'binned':
        df = load_binned_dataframe()

    else:
        df = load_aug_dataframe()

    names = df.loc[:, 'Filename']

    for i in range(len(names)):

        if names[i] == filename:
            indx = i

    plt.figure()

    flux = df.loc[indx]['Flux']

    time = df.loc[indx]['Time']

    plt.plot(time, flux)

    plt.title(filename)

    plt.show()

    print('indx: ', indx)


def remove_outliers():
    """
    Outputs number of outliers caused by light scattering
    """
    df = load_raw_dataframe()

    print('old df len: ', len(df))

    indxs = []

    fluxs = df.loc[:, 'Flux']

    pd.DataFrame
    count = 0

    for i in tqdm(range(len(fluxs))):

        flux = fluxs[i]

        outlier = False

        for f in flux:
            if f > 1e6:
                outlier = True
                break

        if outlier:
            indxs.append(i)
            count += 1

    print('num of outliers: ', count)

    new_df = df.drop(labels=indxs)
    new_df = new_df.reset_index(drop=True)

    print('new df len: ', len(new_df))

    return new_df


def prepare_data(save=True):

    NN_features = ('Flux', 'Error', 'Mag at Discovery')

    df = load_aug_dataframe()

    columns = df.columns

    print('df length: ', len(df))

    prepared_data = []

    features = []

    for column in columns:

        if column in NN_features:

            print('adding feature: ', column)

            features.append(np.array(df.loc[:, column]))

    for i in range(len(df)):

        # extract lc feats
        lc_flux = np.vstack(features[0][i])
        lc_error = np.vstack(features[1][i])
        lc_mag = np.vstack(features[2][i])

        # combine
        lc_flux = np.concatenate((lc_flux, lc_error), axis=1)
        lc_flux = np.concatenate((lc_flux, lc_mag), axis=1)

        if i == 0:
            print('shape of combined features for one lc: ', lc_flux.shape)
        prepared_data.append(lc_flux)

    prepared_data = np.array(prepared_data)
    print('shape of prepared data: ', prepared_data.shape)
    if save:
        np.save(filepath+'data/prepared_data.npy', prepared_data)

    return prepared_data


def main():

    # read raw data
   # read_raw_data()

    # build df's
    create_raw_dataframe()
    # create_binned_dataframe()
    # create_aug_dataframe()

    # plot binned vs raw
   # plot_random_binned_data()

#    plot_specific('raw','2021yjr')
#    find_outliers()
    # prepare_data()


if __name__ == '__main__':
    main()

# %%
