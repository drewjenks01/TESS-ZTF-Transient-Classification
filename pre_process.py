"""
This file is used for reading in data, processing.
"""
# %%
import pandas as pd
import numpy as np
from os import walk
import matplotlib.pyplot as plt
import pickle as pkl
from collections import Counter
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def read_raw_data(lc_path, transient_path):
    """
    Reads data from light curve and transient files, processes, 
    and loads into Panda's DataFrame.

    Args:
        lc_path (str): Filepath to the folder containing light curve files
        transient_path (str): Filepath to the file containing all transient info (IAU name, classification, etc.)

    Returns:
        light_curves (DataFrame): Light curves processed with class encodings and ZTF/TESS filter id's. ['Filename', 'Time', 'Filter ID', 'Flux', 'Error', 'Class'] colums
        original_curves (DataFrame): All light curves. ['Filename', 'Time', 'Flux', 'Class'] columns
        all_transients (DateFrame): Data from the transient file loaded into a DataFrame for easier access in other methods
    """
    print("reading raw data...")

    # get all light curve files
    lc_files = next(walk(lc_path), (None, None, []))[2]

    # extracts data from all transients file
    all_transients = pd.read_csv(transient_path)

    # define filter id's
    tess_id = 8
    ztf_r_id = 6
    ztf_g_id = 5

    # define mask value (update in NN_model if changed)
    mask_val = 0.0

    # define class encodings
    SNIa = {'SNIa', 'SNI', 'SNIa-91T-like',
            'SNIa-91bg-like', 'SNIa-pec', 'SNIa-SC'}
    SNIbc = {'SNIbn', 'SNIb/c', 'SNIb', 'SNIc', 'SNIc-BL'}
    SNIi = {'SNII', 'SNIIb', 'SNIIP', 'SNII-pec', 'SNIIn'}
    other = {'CV', 'SLSN-I', 'AGN', 'FRB', 'Mdwarf',
             'Nova', 'Other', 'Varstar'}

    def modify_class(name):
        """
        Numerically encodes class names
        """
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

    # used to hold filename -> class info
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
            # adjust string (based on the input file structure)
            at_class[i] = at_class[i].split('/')[1].replace(" ", '')

    # populate the at_dict
    for i in range(len(at_files)):
        at_dict[at_files[i]] = at_class[i]

    # dataframe for original light curves
    original_curves = pd.DataFrame(
        columns=['Filename', 'Time', 'Flux', 'Class'])

    # to find out the max light curve length (used for augmenting smaller light curves)
    max_lc_len = 0

    # reads through all light curve files
    light_curves = []
    for f in tqdm(lc_files):
        # filtered DF
        filtered_df = pd.DataFrame(
            columns=['Filename', 'Time', 'Filter ID', 'Flux', 'Error', 'Class'])

        # read light curve from file
        df = pd.read_csv(lc_path+f)

        # num of timesteps
        num_steps = len(df['relative_time'])

        # process filename to be just abreviation ex: 2018evo
        name_split = f.split('_')
        name_abr = name_split[1]

        # grab mag and class
        if name_abr == 'SN':
            continue
        classif = modify_class(at_dict[name_abr])

        # add info to original_curves dataframe
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

        # update max_lc_len if we find a longer lc
        if filtered_df.shape[0] > max_lc_len:
            max_lc_len = filtered_df.shape[0]

        # skip the light curve if there was no data
        elif filtered_df.shape[0] == 0:
            continue

        # add light curve to processed light curve DF
        light_curves.append(filtered_df)

    # augment data length if needed
    for i in tqdm(range(len(light_curves))):
        f = light_curves[i].copy()
        while(len(f.loc[:, 'Filename']) < max_lc_len):
            # dict to add to DF
            df_dict = {'Filename': f.loc[0, 'Filename'], 'Time': mask_val, 'Filter ID': mask_val,
                       'Flux': mask_val, 'Error': mask_val, 'Class': f.loc[0, 'Class']}
            f = f.append(df_dict, ignore_index=True)

        # update light curve
        light_curves[i] = f

    # get classification counts
    counts = []
    for f in tqdm(light_curves):
        counts.append(f.loc[0, 'Class'])

    print('Class counts: ', Counter(counts))
    print('Number of light curves recorded: ', len(light_curves))
    print('Number of timesteps in first light curves',
          len(light_curves[0].loc[:, 'Time']))

    return light_curves, original_curves, all_transients


def plot_specific(original_curves, filename):
    """
    Plots the data of a specific light curve

    Args:
        original_curves (DataFrame): DataFrame containing light curves before processing
        filename (str): filename abreviation of the light curve to plot (Ex: '2018fzi')
    """
    df = original_curves

    # get all names in the DataFrame
    names = df.loc[:, 'Filename']

    # handle case where filename doesnt exist
    if filename not in names:
        return 'Invalid filename: not found in DateFrame'

    # find the index of the file
    for i in range(len(names)):
        if names[i] == filename:
            indx = i

    # plot the light curve
    plt.figure()

    flux = df.loc[indx]['Flux']

    time = df.loc[indx]['Time']

    plt.plot(time, flux)

    plt.title(filename)

    plt.show()


def prepare_NN_data(lc):
    """
    Turns light curve data into format that is ready to be inputted into neural network model

    Args:
        lc (DataFrame): Processed light curve DataFrame

    Returns:
        prepared_data (Numpy matrix): Matrix of light curves. Shape = # light curves x # timesteps x # features
    """

    # load light curve DF
    light_curves = lc

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
    return prepared_data

# %%
