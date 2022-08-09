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
        filepath+'data/all_transients.txt', header=None, delim_whitespace=True)
    all_transients.columns = ['sector', 'ra', 'dec', 'mag at discovery', 'time of discovery', 'type of transient',
                              'classification', 'IAU name', 'discovery survey', 'cam', 'ccd', 'column', 'row']

    # define filter id's
    tess_id = 700
    ztf_r_id = 630
    ztf_g_id = 550

    # define mask value
    mask_val = 3141592

    # define how long each light curve df should be
    lc_df_length = 180

    # define class encodings
    SNIa = {'SNIa','SNI','SNIa-91T-like','SNIa-91bg-like','SNIa-pec', 'SNIa-SC'}
    SNIbc = {'SNIbn','SNIb/c','SNIb','SNIc','SNIc-BL'}
    SNIi ={'SNII', 'SNIIb','SNIIP','SNII-pec','SNIIn'}
    other = {'CV', 'SLSN-I','AGN', 'FRB','Mdwarf', 
                        'Nova', 'Other', 'Varstar'}
    def modify_class(name):
        if name in SNIa:
            return 0
        elif name in SNIbc:
            return 1
        elif name in SNIi:
            return 2
        elif name in other:
            return 3
        else:
            return 4

    # reads through all light curve files
    light_curves = []
    for f in tqdm(lc_files):
        # filtered DF
        filtered_df = pd.DataFrame(columns=['Filename','Time', 'Filter ID', 'Flux', 'Error','Class'])

        #read light curve from file
        df = pd.read_csv(filepath+'data/processed_curves/'+f)
        
        # num of timesteps
        num_steps = len(df['relative_time'])

         # process filename to be just abreviation ex: 2018evo
        name_split = filename.split('_')
        name_abr = name_split[1]

        # grab mag and class
        if name_abr =='SN':
            continue
        classif = modify_class(name_abr)


        # checks for Filter ID processing
        for i in range(num_steps):
            
            # if all missing data, skip this timestep
            if df[i,'tess_flux'] ==np.NAN and df[i,'r_flux']==np.NAN and df[i,'g_flux'] == np.NAN:
                continue
            
            # TESS check
            if df[i,'tess_flux'] != np.NAN:
                # extract vals
                time=df.loc[i,'relative_time']
                filt_id = tess_id
                flux = df.loc[i,'tess_flux']
                error =df.loc[i,'tess_uncert']

                # add to DF
                filtered_df[-1]=[f,time,filt_id,flux,error,classif]

            # Ztf-r check
            if df[i,'r_flux'] != np.NAN:
                # extract vals
                filename = f
                time=df.loc[i,'relative_time']
                filt_id = ztf_r_id
                flux = df.loc[i,'r_flux']
                error =df.loc[i,'r_uncert']

                # add to DF
                filtered_df[-1]=[f,time,filt_id,flux,error,classif]

            # Ztf-g check
            if df[i,'g_flux'] != np.NAN:
                # extract vals
                filename = f
                time=df.loc[i,'relative_time']
                filt_id = ztf_g_id
                flux = df.loc[i,'g_flux']
                error =df.loc[i,'g_uncert']

                # add to DF
                filtered_df[-1]=[f,time,filt_id,flux,error,classif]

        # augment data length if needed
        while(len(filtered_df['filename'])<180):
            filtered_df[-1]=[f,mask_val,mask_val,mask_val,mask_val,classif]

        # append final df for this light curve
        light_curves.append(filtered_df)


    # concat all dataframes into one
    light_curve_df = pd.concat(light_curves)

    if save:
        print("saving raw data...")

        all_transients.to_pickle(filepath+'data/all_transients.pkl')

        light_curve_df.to_pickle(filepath+'data/light_curve_df.pkl')


def load_light_curve_dataframe():
    print('loading light curve df...')
    return pd.read_pickle(filepath+'data/light_curves_df.pkl')


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


def prepare_NN_data(save=True):

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
        print('saving prepared data...')
        np.save(filepath+'data/prepared_data.npy', prepared_data)

    return prepared_data


def main():

    # read raw data
    read_raw_data()


    # plot_specific('raw','2021aaeb')
    prepare_data()


if __name__ == '__main__':
    main()

# %%
