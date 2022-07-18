# %%
from NN_model import RVAE
from pre_process import load_aug_dataframe, load_raw_dataframe
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
import joblib
from collections import Counter
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class RandomForest:

    def __init__(self):

        # initialize rvae
        self.rvae = RVAE()

        self.encoded_dim = 20

        # get training encoder
        self.encoder = self.rvae.get_encoder()

        # filepath
        self.filepath = '/Users/drewj/Documents/Urops/Muthukrishna/data/'

        # augmented data frame
        self.aug_df = load_aug_dataframe()

        # prepared data (Flux, Error, Mag)
        self.prepared_data = np.load(self.filepath+'prepared_data.npy')

        # test and train data
        self.train_test_data = self.create_test_train()

    def create_test_train(self):
        """
        Splits data into 3/4 training, 1/4 testing
        """

        print("Splitting data into train and test...")

        # prepared out (only class)
        prep_out = self.aug_df.loc[:, 'Class']

        prep_inp = self.prepared_data

        x_train = []
        y_train = []
        x_test = []
        y_test = []

        train_indx = set()
        test_indx = set()

        # calc the # of light curves for train vs test
        num_lcs = len(prep_inp)
        train_perc = round(0.75 * num_lcs)
        test_perc = num_lcs-train_perc

        # save random indices for training
        while len(train_indx) != train_perc:
            indx = random.randint(0, num_lcs-1)
            train_indx.add(indx)

        # save random indices for testint -> no duplicates from training
        while len(test_indx) != test_perc:
            indx = random.randint(0, num_lcs-1)
            if indx not in train_indx:
                test_indx.add(indx)

        # extract training data
        for ind in train_indx:
            x_train.append(prep_inp[ind])
            y_train.append(prep_out[ind])

        # extract testing data
        for ind in test_indx:
            x_test.append(prep_inp[ind])
            y_test.append(prep_out[ind])

        # change to numpy arrays
        x_train = np.array(x_train)
        x_test = np.array(x_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)

        print('shape of prep_inp and x_train:', prep_inp.shape, x_train.shape)
        print('shape of prep_out and y_train:', prep_out.shape, y_train.shape)

        return [x_train, x_test, y_train, y_test]

    def make_encodings(self, save=True):
        """
        Uses trained encoder to produce 1D encodings of light curves to be used for RF training
        """
        print('making encodings...')
        # extract x vals
        x_train = self.train_test_data[0]
        x_test = self.train_test_data[1]

        # arrays for encoded light curves
        x_train_enc = []
        x_test_enc = []

        # encode training light curves
        x_train_enc = self.encoder.predict(
            x_train, workers=32, use_multiprocessing=True, batch_size=64, verbose=1)[2]
        x_test_enc = self.encoder.predict(
            x_test, workers=32, use_multiprocessing=True, batch_size=64, verbose=1)[2]

        # numpy arrays
        x_test_enc = np.array(x_test_enc)
        x_train_enc = np.array(x_train_enc)

        print('shape of encodings: ', x_test_enc.shape, x_train_enc.shape)

        if save:
            np.save(self.filepath+'x_train_enc.npy', x_train_enc)
            np.save(self.filepath+'x_test_enc.npy', x_test_enc)

    def build_classier(self, save=True):
        """
        Trains a RF classifier and tests its prediction accuracy
        """
        print('building classifier...')

        # initialize random forest classifier
        rf = RandomForestClassifier()

        # train using train data
        x_train = np.load(self.filepath+'x_train_enc.npy')
        x_test = np.load(self.filepath+'x_test_enc.npy')
        y_train = self.train_test_data[2]
        y_test = self.train_test_data[3]

        # reshape
        x_train = x_train.reshape(-1, self.encoded_dim)
        x_test = x_test.reshape(-1, self.encoded_dim)

        print('shape of encodings: ', x_test.shape, x_train.shape)

        rf.fit(x_train, y_train)

        # performing predictions on the test dataset
        y_pred = rf.predict(x_test)

        print('y_test counts: ', Counter(y_test))
        print('y_pred counts: ', Counter(y_pred))

        # check accuracy
        print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))

        # Create confusion matrix
        conf_mat = pd.crosstab(y_test, y_pred, rownames=[
                               'Actual Species'], colnames=['Predicted Species'])

        print('Confusion Matrix: ', conf_mat)

        if save:
            # save
            print('saving rf...')
            joblib.dump(rf, self.filepath+"random_forest.joblib")

        return rf

    def load_rf(self):
        print('loading rf')
        rf = joblib.load(self.filepath+"random_forest.joblib")
        return rf

    def classify(self, data=None, correct=None, filename=None):
        """
        Classifies a specific light curve.

        Data should not be encoded yet.

        Pulls data from file if filename abbreviation passed.
        """
        print('classifying specific light curve data...')

        # load file data if file is passed
        if filename:
            raw_df = load_raw_dataframe()
            names = raw_df.loc[:, 'Filename']

            for i in range(len(names)):
                if names[i] == filename:
                    indx = i

            data = self.prepared_data[indx]
            correct = raw_df.loc[indx]['Class']

        # load trained RF
        rf = self.load_rf()

        # reshape
        data = data.reshape(1, 30, 3)

        # encode data
        data = self.encoder.predict(data)[2]

        # make class num -> classification dict
        classes = {0: 'SNIa', 1: 'SNII', 2: 'Unclassified', 3: 'CV', 4: 'SLSN-I', 5: 'AGN',
                   6: 'SNIc-BL', 7: 'SNIbn', 8: 'SN', 9: 'FRB', 10: 'SNIc', 11: 'SNIb/c',
                   12: 'SNIIn', 13: 'SNIb', 14: 'SNIa-91T-like', 15: 'Mdwarf', 16: 'SNIIb',
                   17: 'Nova', 18: 'SNIa-91bg-like', 19: 'Other', 20: 'SNIIP', 21: 'SNI',
                   22: 'Varstar', 23: 'SNII-pec', 24: 'SNIa-pec', 25: 'SNIa-SC'}

        # make prediction from data
        pred = rf.predict(data)

        # print confidence
        probs = np.array(rf.predict_proba(data)[0])
        print('Number of different possible predictions: ', len(probs))
        highest_prob_ind = np.argmax(probs)
        highest_prob = max(probs)
        print('Prediction is ' + classes[highest_prob_ind]+' (',
              pred[0], ') with '+str(highest_prob*100)+'% confidence')
        print('Correct classification should be: ',
              classes[correct], ' (', correct, ')')


def main():

    # initialize random forest
    rf = RandomForest()

    # make lc encodings
    rf.make_encodings()

    # train rf
    rf.build_classier()

    # predict a specific light curve
    rf.classify(filename='2018evo')


if __name__ == "__main__":
    main()
# %%
