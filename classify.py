# %%
from NN_model import RVAE
from imblearn.ensemble import BalancedRandomForestClassifier
import pandas as pd
import numpy as np
import random
import joblib
from collections import Counter
import tensorflow as tf
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


class RandomForest:

    def __init__(self, light_curves, prepared_data, encoder):

        # initialize rvae
        self.rvae = RVAE(prepared_data)

        # encoded dimension from NN
        self.encoded_dim = self.rvae.latent_dim

        # get training encoder
        self.encoder = encoder

        # augmented data frame
        self.light_curves = light_curves

        # prepared data
        self.prepared_data = prepared_data

    def create_test_train(self):
        """
        Splits data into 85% training, 15% testing, and unlabeled
        """

        print("Splitting data for RF...")

        # extract all class outputs and inputs
        prep_out = np.array([c.loc[0, 'Class'] for c in self.light_curves])
        prep_inp = self.prepared_data

        # extract all training label indexes
        train_indx = [i for i in range(len(prep_out)) if prep_out[i] != 3]
        unclassified_indx = [i for i in range(
            len(prep_out)) if prep_out[i] == 3]
        num_indxs = len(train_indx)

        x_train = []
        y_train = []
        x_test = []
        y_test = []
        x_unclassified = []

        # extract training data
        while len(x_train) < int(num_indxs*0.85):
            ran = random.randint(0, len(train_indx)-1)
            ind = train_indx[ran]
            train_indx.remove(ind)
            x_train.append(prep_inp[ind])
            y_train.append(prep_out[ind])

        # append the rest of the data to testing
        for ind in train_indx:
            x_test.append(prep_inp[ind])
            y_test.append(prep_out[ind])

        # extract unclassified data
        for ind in unclassified_indx:
            x_unclassified.append(prep_inp[ind])

        # change to numpy arrays
        x_train = np.array(x_train)
        x_test = np.array(x_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        x_unclassified = np.array(x_unclassified)

        print('shape of x_train and x_test:', x_train.shape, x_test.shape)
        print('shape of y_train and y_test:', y_train.shape, y_test.shape)
        print('shape of x_unclassified:', x_unclassified.shape)

        return x_train, x_test, y_train, y_test, x_unclassified

    def make_encodings(self, x_train, x_test, x_unclassified):
        """
        Uses trained encoder to produce 1D encodings of light curves to be used for RF training
        """
        print('making encodings...')

        # encode training light curves
        x_train_enc = self.encoder.predict(
            x_train, workers=32, use_multiprocessing=True, batch_size=128, verbose=1)[2]
        x_test_enc = self.encoder.predict(
            x_test, workers=32, use_multiprocessing=True, batch_size=128, verbose=1)[2]
        x_unclassified_enc = self.encoder.predict(
            x_unclassified, workers=32, use_multiprocessing=True, batch_size=64, verbose=1)[2]

        # numpy arrays
        x_test_enc = np.array(x_test_enc)
        x_train_enc = np.array(x_train_enc)
        x_unclassified_enc = np.array(x_unclassified_enc)

        print('shape of encodings: ', x_train_enc.shape,
              x_test_enc.shape, x_unclassified_enc.shape)

        return x_train_enc, x_test_enc, x_unclassified_enc

    def build_classier(self, x_train, x_test, x_unclassified, y_train, y_test):
        """
        Trains a RF classifier and tests its prediction accuracy
        """
        print('building classifier...')

        # initialize random forest classifier
        rf = BalancedRandomForestClassifier(n_estimators=20)

        # reshape
        x_train = x_train.reshape(-1, self.encoded_dim)
        x_test = x_test.reshape(-1, self.encoded_dim)
        x_unclassified = x_unclassified.reshape(-1, self.encoded_dim)

        print('shape of encodings: ', x_train.shape,
              x_test.shape, x_unclassified.shape)

        # fit to data
        rf.fit(x_train, y_train)

        # performing predictions on the test dataset
        y_pred = rf.predict(x_test)
        print('y_train counts: ', Counter(y_train))
        print('y_test counts: ', Counter(y_test))
        print('y_pred counts: ', Counter(y_pred))

        # check accuracy
        print("ACCURACY OF THE MODEl: ", 100 *
              round(rf.score(x_test, y_test), 2), '%')

       # Create confusion matrix
        conf_mat = pd.crosstab(y_test, y_pred, rownames=[
                               'Actual Species'], colnames=['Predicted Species'])

        print('Confusion Matrix:')
        print(conf_mat.to_string())

        plot_confusion_matrix(rf, x_test, y_test)

        print('Unlabeled Classifications: ')
        unlabeled = rf.predict(x_unclassified)
        print(unlabeled)
        print(Counter(unlabeled))

        return rf

    def classify(self, rf,original_curves, filename):
        """
        Classifies a specific light curve.
        """
        print('classifying specific light curve data...')

        # load file data if file is passed
        if filename:
            raw_df = original_curves
            names = raw_df.loc[:, 'Filename']

            for i in range(len(names)):
                if names[i] == filename:
                    indx = i

            data = self.prepared_data[indx]
            correct = raw_df.loc[indx]['Class']

        # reshape
        data = data.reshape(1, self.rvae.num_timesteps, self.rvae.num_feats)

        # encode data
        data = self.encoder.predict(data)[2]

        # make class num -> classification dict
        classes = {0: 'SNIa', 1: 'SNIbc', 2: 'SNII',
                   3: 'Other', 4: 'Unclassified'}

        # make prediction from data
        pred = rf.predict(data)

        # print confidence
        probs = np.array(rf.predict_proba(data)[0])
        print('Number of different possible predictions: ', len(probs))
        highest_prob_ind = np.argmax(probs)
        highest_prob = max(probs)
        print('Prediction is ' + classes[highest_prob_ind]+' (',
              pred[0], ') with '+str(int(highest_prob*100))+'% confidence')
        print('Correct classification should be: ',
              classes[correct], ' (', correct, ')')

# %%
