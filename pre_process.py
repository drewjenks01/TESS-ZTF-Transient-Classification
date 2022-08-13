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
from sklearn import preprocessing


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
        filepath+'data/all_transients.csv')
    # all_transients.columns = ['sector', 'ra', 'dec', 'mag at discovery', 'time of discovery', 'type of transient',
    #                          'classification', 'IAU name', 'discovery survey', 'cam', 'ccd', 'column', 'row']

    # define filter id's
    tess_id = 8
    ztf_r_id = 6
    ztf_g_id = 5

    # define mask value
    mask_val = 0.0

    # define class encodings
    SNIa = {'SNIa', 'SNI', 'SNIa-91T-like',
            'SNIa-91bg-like', 'SNIa-pec', 'SNIa-SC'}
    SNIbc = {'SNIbn', 'SNIb/c', 'SNIb', 'SNIc', 'SNIc-BL'}
    SNIi = {'SNII', 'SNIIb', 'SNIIP', 'SNII-pec', 'SNIIn'}
    other = {'CV', 'SLSN-I', 'AGN', 'FRB', 'Mdwarf',
             'Nova', 'Other', 'Varstar'}

    def modify_class(name):
        if name in SNIa:
            return 0
        elif name in SNIbc:
            return 1
        elif name in SNIi:
            return 2
        elif name == 'Unclassified':
            return 3
        else:
            return 4

    at_dict = {}

    # loop through AT and grab necessary data
    at_files = all_transients.loc[:, 'IAU_name']
    at_class = all_transients.loc[:, 'classification'].copy()

    # change nan's to unclassified
    for i in range(len(at_class)):
        try:
            if np.isnan(at_class[i]):
                at_class[i] = 'Unclassified'
        except:
            at_class[i] = at_class[i].split('/')[1].replace(" ", '')

    for i in range(len(at_files)):
        at_dict[at_files[i]] = at_class[i]

    # dataframe for original light curves
    original_curves = pd.DataFrame(
        columns=['Filename', 'Time', 'Flux', 'Class'])

    # to find out the max light curve length
    max_lc_len = 0

    # reads through all light curve files
    light_curves = []
    for f in tqdm(lc_files):
        # filtered DF
        filtered_df = pd.DataFrame(
            columns=['Filename', 'Time', 'Filter ID', 'Flux', 'Error', 'Class'])

        # read light curve from file
        df = pd.read_csv(filepath+'data/processed_curves/'+f)

        # num of timesteps
        num_steps = len(df['relative_time'])

        # process filename to be just abreviation ex: 2018evo
        name_split = f.split('_')
        name_abr = name_split[0]

        # grab mag and class
        if name_abr == 'SN':
            continue
        classif = modify_class(at_dict[name_abr])

        # add info to original dataframe
        df_dict = {'Filename': name_abr,
                   'Time': df.loc[:, 'relative_time'], 'Flux': df.loc[:, 'tess_flux'], 'Class': classif}
        original_curves = original_curves.append(df_dict, ignore_index=True)

        # checks for Filter ID processing
        for i in range(num_steps):

            # if all missing data, skip this timestep
            if np.isnan(df.loc[i, 'tess_flux']) and np.isnan(df.loc[i, 'r_flux']) and np.isnan(df.loc[i, 'g_flux']):
                continue

            # if time not between -30 & +70, skip
            time = df.loc[i, 'relative_time']
            if time < -30 or time > 70:
                continue

            # TESS check
            if not np.isnan(df.loc[i, 'tess_flux']):
                # extract vals
                filt_id = tess_id
                flux = df.loc[i, 'tess_flux']
                error = df.loc[i, 'tess_uncert']

                # dict to add to DF
                df_dict = {'Filename': f, 'Time': time, 'Filter ID': filt_id,
                           'Flux': flux, 'Error': error, 'Class': classif}

                # add df_dict to dataframe
                filtered_df = filtered_df.append(df_dict, ignore_index=True)

            # Ztf-r check
            if not np.isnan(df.loc[i, 'r_flux']):
                # extract vals
                filt_id = ztf_r_id
                flux = df.loc[i, 'r_flux']
                error = df.loc[i, 'r_uncert']

                # dict to add to DF
                df_dict = {'Filename': f, 'Time': time, 'Filter ID': filt_id,
                           'Flux': flux, 'Error': error, 'Class': classif}

                # add df_dict to dataframe
                filtered_df = filtered_df.append(df_dict, ignore_index=True)

            # Ztf-g check
            if not np.isnan(df.loc[i, 'g_flux']):
                # extract vals
                filt_id = ztf_g_id
                flux = df.loc[i, 'g_flux']
                error = df.loc[i, 'g_uncert']

                # dict to add to DF
                df_dict = {'Filename': f, 'Time': time, 'Filter ID': filt_id,
                           'Flux': flux, 'Error': error, 'Class': classif}

                # add df_dict to dataframe
                filtered_df = filtered_df.append(df_dict, ignore_index=True)

        if filtered_df.shape[0] > max_lc_len:
            max_lc_len = filtered_df.shape[0]
        elif filtered_df.shape[0] == 0:
            continue

        light_curves.append(filtered_df)

    # augment data length if needed
    for i in tqdm(range(len(light_curves))):
        f = light_curves[i].copy()
        while(len(f.loc[:, 'Filename']) < max_lc_len):
            # dict to add to DF
            df_dict = {'Filename': f.loc[0, 'Filename'], 'Time': mask_val, 'Filter ID': mask_val,
                       'Flux': mask_val, 'Error': mask_val, 'Class': f.loc[0, 'Class']}
            f = f.append(df_dict, ignore_index=True)

        light_curves[i] = f

    # get classification counts
    counts = []
    for f in tqdm(light_curves):
        counts.append(f.loc[0, 'Class'])

    print('Class counts: ', Counter(counts))
    print('Number of light curves recorded: ', len(light_curves))
    print('Number of timesteps in first light curves',
          len(light_curves[0].loc[:, 'Time']))
    if save:
        print("saving raw data...")
        all_transients.to_pickle(filepath+'data/all_transients.pkl')
        original_curves.to_pickle(filepath+'data/original_curves.pkl')
        with open(filepath+'data/light_curves.pkl', "wb") as f:
            pkl.dump(light_curves, f)


def load_light_curve_dataframe():
    print('loading light curve df...')
    with open(filepath+'data/light_curves.pkl', "rb") as f:
        light_curves = pkl.load(f)
    return light_curves


def load_original_curves():
    print('loading original curves...')
    return pd.read_pickle(filepath+'data/original_curves.pkl')


def plot_specific(filename):
    """
    Plots the data of a specific light curve

    Args:
        df_type (str): type of dataframe to extract lc from (raw, binned, aug)
        filename (str): filename abreviation of the light curve to plot
    """
    df = load_original_curves()

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


def prepare_NN_data(save=True):

    # load light curve DF
    light_curves = load_light_curve_dataframe()

    # array for prepared data
    prepared_data = []

    # loop through each light curve
    for lc in light_curves:

        # extract the data for this specific data
        data = np.array(lc[['Time', 'Filter ID', 'Flux', 'Error']])

        # add to prepared data
        prepared_data.append(data)

    # make into numpy array
    prepared_data = np.array(prepared_data)

    print('shape of prepared data: ', prepared_data.shape)
    if save:
        print('saving prepared data...')
        np.save(filepath+'data/prepared_data.npy', prepared_data)
    return prepared_data


def main():

    # read raw data
    read_raw_data()

    # plot_specific('raw','2021aaeb')
    prepare_NN_data()
    loaded = np.load(filepath+'data/prepared_data.npy', allow_pickle=True)

    print(loaded[0])


if __name__ == '__main__':
    main()

sk  # %%

# %%
