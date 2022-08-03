# %%
from NN_model import RVAE
from pre_process import load_aug_dataframe, load_raw_dataframe
from imblearn.ensemble import EasyEnsembleClassifier,BalancedRandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn import metrics
import pandas as pd
import numpy as np
import random
import joblib
from collections import Counter
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class RandomForest:

    def __init__(self):

        # initialize rvae
        self.rvae = RVAE()

        self.encoded_dim = 35

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
        Splits data into 85% training, 15% testing, and unlabeled
        """

        print("Splitting data for RF...")

        # extract all class outputs and inputs
        prep_out = self.aug_df.loc[:, 'Class']
        prep_inp = self.prepared_data

        # extract all training label indexes
        train_indx = list(self.aug_df[self.aug_df.loc[:, 'Class'] != 4].index)
        unclassified_indx = list(
            self.aug_df[self.aug_df.loc[:, 'Class'] == 4].index)
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

        return [x_train, x_test, y_train, y_test, x_unclassified]

    def make_encodings(self, save=True):
        """
        Uses trained encoder to produce 1D encodings of light curves to be used for RF training
        """
        print('making encodings...')
        # extract x vals
        x_train = self.train_test_data[0]
        x_test = self.train_test_data[1]
        x_unclassified = self.train_test_data[4]

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

        if save:
            np.save(self.filepath+'x_train_enc.npy', x_train_enc)
            np.save(self.filepath+'x_test_enc.npy', x_test_enc)
            np.save(self.filepath+'x_unclassified_enc.npy', x_unclassified_enc)

    def build_classier(self, save=True):
        """
        Trains a RF classifier and tests its prediction accuracy
        """
        print('building classifier...')

        # initialize random forest classifier
        rf = BalancedRandomForestClassifier(n_estimators=20)

        # train using train data
        x_train = np.load(self.filepath+'x_train_enc.npy')
        x_test = np.load(self.filepath+'x_test_enc.npy')
        x_unclassified = np.load(self.filepath+'x_unclassified_enc.npy')
        y_train = self.train_test_data[2]
        y_test = self.train_test_data[3]

        # reshape
        x_train = x_train.reshape(-1, self.encoded_dim)
        x_test = x_test.reshape(-1, self.encoded_dim)
        x_unclassified = x_unclassified.reshape(-1, self.encoded_dim)

        print('shape of encodings: ', x_train.shape,
              x_test.shape, x_unclassified.shape)

        rf.fit(x_train, y_train)
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

        print('Unlabeled Classifications: ')
        unlabeled = rf.predict(x_unclassified)
        print(unlabeled)
        print(Counter(unlabeled))

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
