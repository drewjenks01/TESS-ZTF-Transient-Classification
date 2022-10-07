# %%
import random
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import Input, Model
from keras.callbacks import EarlyStopping
from keras.layers import (GRU, Dense, Lambda, Masking,
                          RepeatVector, TimeDistributed, concatenate)
from keras.optimizers import adam_v2
from sklearn.manifold import TSNE
from tensorflow.python.framework.ops import disable_eager_execution
from tqdm import tqdm

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
session = tf.compat.v1.InteractiveSession(
    config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

disable_eager_execution()
"""
unique classifiers:  SNIa (0) = {'SNIa', 'SNI', 'SNIa-91T-like',
                                'SNIa-91bg-like', 'SNIa-pec', 'SNIa-SC'}
                     SNIbc (1) = {'SNIbn', 'SNIb/c', 'SNIb', 'SNIc', 'SNIc-BL'}
                     SNIi (2) = {'SNII', 'SNIIb', 'SNIIP', 'SNII-pec', 'SNIIn'}
                     Unclassified (3)
                     other (4) = {'CV', 'SLSN-I', 'AGN', 'FRB', 'Mdwarf',
             '                  Nova', 'Other', 'Varstar'}

class counts:  {3: 529, 0: 231, 2: 24, 1: 12, 4: 9}
"""


class RVAE:
    """ Recurrent Variational Autoencoder """

    def __init__(self, prepped_data):

        # training epochs
        self.epochs = 5

        # batch size
        self.batch_size = 64

        # optimizer
        self.optimizer = adam_v2.Adam(learning_rate=1e-2)

        # set the dimension of the latent vector
        self.latent_dim = 30

        # input to first enc, second dec layer
        self.gru_one = 175

        # input to first dec, second enc layer
        self.gru_two = 150

        # load prepared data (acts as input)
        self.prepared_data = prepped_data

        # number of input features
        self.num_feats = self.prepared_data.shape[2]

        # number of timesteps
        self.num_timesteps = self.prepared_data.shape[1]

        # dimension of the input space for encoder
        self.enc_input_shape = (self.num_timesteps, self.num_feats)

        # number of light curves
        self.num_lcs = self.prepared_data.shape[0]

        # indxs for test and train
        self.train_indx = set()
        self.test_indx = set()

        # mask value
        self.mask_val = 0.0

    def build_disconnected_model(self):
        """
        Builds entire RVAE model with encoder and decoder as two seperate Models

        Returns:
            model : rvae model
        """

        # BUILD ENCODER
        print("building encoder...")

        # input layer
        enc_input = Input(shape=self.enc_input_shape)

        # masking layer to get rid of unwanted values
        mask = Masking(mask_value=0.0)(enc_input)

        # first recurrent layer
        x = GRU(self.gru_one, activation='tanh',
                recurrent_activation='hard_sigmoid', return_sequences=True, name='gru1')(mask)

        # second recurrent layer
        encoded = GRU(self.gru_two, activation='tanh',
                      recurrent_activation='hard_sigmoid', return_sequences=True, name='gru2')(x)

        # z mean output
        z_mean = GRU(
            self.latent_dim, return_sequences=False, activation='linear', name='gru3')(encoded)

        # z variance output
        z_log_var = GRU(
            self.latent_dim, return_sequences=False, activation='linear', name='gru4')(encoded)

        # sample output
        z = Lambda(self.sampling, output_shape=(
            self.latent_dim,), name='lam')([z_mean, z_log_var])

        # encoder
        encoder = Model(enc_input, [z_mean, z_log_var, z], name='encoder')
        encoder.summary()

        # BUILD decoder
        print("building decoder...")

        # repeat layer
        dec_inp = Input(shape=(self.latent_dim,))

        repeater = RepeatVector(30, name='rep')(dec_inp)

        # first recurrent layer
        x = GRU(self.gru_two, activation='tanh',
                recurrent_activation='hard_sigmoid', return_sequences=True, name='gru5')(repeater)

        # second recurrent layer
        x = GRU(self.gru_one, activation='tanh',
                recurrent_activation='hard_sigmoid', return_sequences=True, name='gru6')(x)

        # decoder output
        dec_output = TimeDistributed(
            Dense(1, activation='tanh', input_shape=(None, 1)), name='td')(x)

        decoder = Model(dec_inp, dec_output, name='decoder')
        decoder.summary()

        # BUILD MODEL
        rvae = Model(enc_input, decoder(encoder(enc_input)[2]))
        rvae.summary()

        # for i, l in enumerate(decoder.layers):
        #     print(f'layer {i}: {l}')
        #     print(f'has input mask: {l.input_mask}')
        #     print(f'has output mask: {l.output_mask}')

        # for i, l in enumerate(encoder.layers):
        #     print(f'layer {i}: {l}')
        #     print(f'has input mask: {l.input_mask}')
        #     print(f'has output mask: {l.output_mask}')

        # define loss
        custom_loss = self.vae_loss(z_mean, z_log_var)

        # compile model
        rvae.compile(optimizer=self.optimizer, loss=custom_loss)

        es = EarlyStopping(monitor='val_loss', min_delta=0, patience=10,
                           verbose=0, mode='min', baseline=None,
                           restore_best_weights=True)

        return rvae, encoder, es

    def build_connected_model(self):
        """
        Builds entire RVAE model connected as one model

        Returns:
            model: full rvae model
            encoder: just encoder from model (used for testing and classification)
        """

        # BUILD ENCODER
        print("building encoder...")

        # input layer
        enc_input = Input(shape=self.enc_input_shape)

        # masking layer to get rid of unwanted values
        mask = Masking(mask_value=self.mask_val)

        # get mask for future layer
        mask_compute = mask(enc_input)

        # first recurrent layer
        gru1 = GRU(self.gru_one, activation='tanh',
                   recurrent_activation='hard_sigmoid', return_sequences=True, name='gru1')(mask_compute)

        mask_output = mask.compute_mask(gru1)

        # second recurrent layer
        encoded = GRU(self.gru_two, activation='tanh',
                      recurrent_activation='hard_sigmoid', return_sequences=True, name='gru2')(gru1)

        # z mean output
        z_mean = GRU(
            self.latent_dim, return_sequences=False, activation='linear', name='gru3')(encoded)

        # z variance output
        z_log_var = GRU(
            self.latent_dim, return_sequences=False, activation='linear', name='gru4')(encoded)

        # sample output
        z = Lambda(self.sampling, output_shape=(
            self.latent_dim,), name='lam')([z_mean, z_log_var])

        # encoder
        encoder = Model(enc_input, [z_mean, z_log_var, z], name='encoder')
        encoder.summary()

        # BUILD decoder
        print("building decoder...")
        repeater = RepeatVector(self.num_timesteps, name='rep')(z)

        # time and filter id vals
        input_two = Input(shape=(self.num_timesteps, 2))

        # concat timestep back
        concat = concatenate((repeater, input_two), axis=-1)

        # add mask from original input
        concat._keras_mask = mask_output

        # first recurrent layer
        x = GRU(self.gru_two, activation='tanh',
                recurrent_activation='hard_sigmoid', return_sequences=True, name='gru5')(concat)

        # second recurrent layer
        x = GRU(self.gru_one, activation='tanh',
                recurrent_activation='hard_sigmoid', return_sequences=True, name='gru6')(x)

        # decoder output
        dec_output = TimeDistributed(
            Dense(1, activation='tanh', input_shape=(None, 1)), name='td')(x)

        # BUILD MODEL
        rvae = Model([enc_input, input_two], dec_output)
        rvae.summary()

        # used to see the masking of the layers within the model
        for i, l in enumerate(rvae.layers):
            print(f'layer {i}: {l}')
            print(f'has input mask: {l.input_mask}')
            print(f'has output mask: {l.output_mask}')

        # define loss
        vae_loss = self.vae_loss(z_mean, z_log_var)

        # compile model
        rvae.compile(optimizer=self.optimizer, loss=vae_loss)

        return rvae, encoder

    def sampling(self, samp_args):
        z_mean, z_log_sigma = samp_args

        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean = 0 and std = 1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_sigma) * epsilon

    def customLoss(self, yTrue, yPred):
        """
        Custom loss which doesn't use the errors.

        Used as custom object when loading saved models.
        """

        return K.mean(K.square(yTrue - yPred)/K.square(yTrue))

    def vae_loss(self, encoded_mean, encoded_log_sigma):
        """
        Defines the reconstruction + KL loss in a format acceptable by the Keras model
        """

        kl_loss = - 0.5 * K.mean(1 + K.flatten(encoded_log_sigma) -
                                 K.square(K.flatten(encoded_mean)) - K.exp(K.flatten(encoded_log_sigma)), axis=-1)

        def lossFunction(yTrue, yPred):
            reconstruction_loss = K.log(K.mean(K.square(yTrue - yPred)))
            return reconstruction_loss + kl_loss

        return lossFunction

    def split_prep_data(self):
        """
        Splits data into 3/4 training, 1/4 testing
        """

        print("Splitting data into train and test...")

        # prepared out (only flux)
        prep_out = self.prepared_data[:, :, 2].reshape(
            self.num_lcs, self.num_timesteps, 1)

        prep_inp = self.prepared_data

        x_train = []
        y_train = []
        x_test = []
        y_test = []

        # calc the # of light curves for train vs test
        num_lcs = len(prep_inp)
        train_perc = round(1.0 * num_lcs)
        test_perc = round(num_lcs*0.2)

        # save random indices for training
        while len(self.train_indx) != train_perc:
            indx = random.randint(0, num_lcs-1)
            self.train_indx.add(indx)

        # save random indices for testint -> no duplicates from training
        while len(self.test_indx) <= test_perc:
            indx = random.randint(0, num_lcs-1)
            # if indx not in self.train_indx:
            self.test_indx.add(indx)

        # extract training data
        for ind in self.train_indx:
            x_train.append(prep_inp[ind])
            y_train.append(prep_out[ind])

        # extract testing data
        for ind in self.test_indx:
            x_test.append(prep_inp[ind])
            y_test.append(prep_out[ind])

        # change to numpy arrays
        x_train = np.array(x_train)
        x_test = np.array(x_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)

        print('shape of prep_inp and x_train:', prep_inp.shape, x_train.shape)
        print('shape of prep_out and y_train:', prep_out.shape, y_train.shape)

        return x_train, x_test, y_train, y_test

    def train_model(self, model, x_train, x_test, y_train, y_test):
        """
        Trains the NN on training data

        Returns the trained model.
        """
        # fit model
        train_inp_two = x_train[:, :, :2]
        assert (train_inp_two.shape == (
            x_train.shape[0], x_train.shape[1], 2))

        test_inp_two = x_test[:, :, :2]
        assert (test_inp_two.shape == (
            x_test.shape[0], x_test.shape[1], 2))

        print('fitting model...')
        history = model.fit([x_train, train_inp_two], y_train, epochs=self.epochs, batch_size=self.batch_size,
                            validation_data=([x_test, test_inp_two], y_test), verbose=1, shuffle=False)

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()
        return model

    def test_model(self, rvae, x_test, y_test):
        """
        Uses test data to and NN to predict light curve decodings.

        Plots reconstructed light curved from the model prediction vs the orignal curve.
        """
        test_inp_two = x_test[:, :, :2]

        print('test_inp_one shape: ', x_test.shape)
        print('test_inp_two shape: ', test_inp_two.shape)

        rvae.summary()

        print('predicting...')
        for i in tqdm(range(10)):

            # predicted flux
            predicted = rvae.predict([x_test[i].reshape(-1, self.num_timesteps, 4),
                                     test_inp_two[i].reshape(-1, self.num_timesteps, 2)])[0]

            # if first prediction, print the prediction
            if i == 0:
                print('shape of predicted data: ', predicted.shape)

            # if one of first 10 predictions, plot prediction vs true
            self.plot_true_pred(y_test[i], predicted, i)

        print("done predicting")

    def plot_true_pred(self, raw, pred, num):
        """
        Plots true lightcurves vs their decodings by the NN
        """

        # make 1 x 2 figure
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('True vs Decoded Light Curves')

        raw_flux = raw[:, 0]
        raw_flux = raw_flux[raw_flux != 0.0]
        pred_flux = pred
        #pred_flux = pred_flux[:len(raw_flux)]

        pred_time = range(len(pred_flux))
        raw_time = range(len(raw_flux))

        # plot raw data
        ax1.plot(raw_time, raw_flux)
        ax1.set_title('true light curve')

        # plot predicted data
        ax2.plot(pred_time, pred_flux)
        ax2.set_title('predicted light curve')

        # save image
        fig.show()

    def t_SNE_plot(self, light_curves, encoder):
        """
        Constructs 2D plots of light curves in latent space.
        """
        print('using t-SNE...')

        # extract all class outputs and inputs
        labels = np.array([c.loc[0, 'Class']
                          for c in light_curves if c.loc[0, 'Class'] != 3])

        # extract all training label indexes
        indxs = [i for i in range(len(labels)) if labels[i] != 3]

        labeled_data = np.array([self.prepared_data[i] for i in indxs])

        _, _, z = tqdm(encoder.predict(labeled_data))

        t_sne = TSNE(n_components=2, learning_rate='auto',
                     init='random').fit_transform(z)
        print('t-sne shape: ', t_sne.shape)

        plt.figure(figsize=(12, 10))
        plt.scatter(t_sne[:, 0], t_sne[:, 1], c=labels)
        plt.colorbar()
        plt.title("t-SNE with only labeled data")
        plt.show()

        # include unlabeled data
        labels = np.array([c.loc[0, 'Class'] for c in light_curves])
        data = self.prepared_data

        _, _, z = tqdm(encoder.predict(data))

        t_sne = TSNE(n_components=2, learning_rate='auto',
                     init='random').fit_transform(z)
        print('t-sne shape: ', t_sne.shape)

        plt.figure(figsize=(12, 10))
        plt.scatter(t_sne[:, 0], t_sne[:, 1], c=labels)
        plt.colorbar()
        plt.title("t-SNE with labeled and unlabeled data")
        #plt.savefig(self.filepath+'plots/unlabeled-t-sne-latent-space.png', facecolor='white')
        plt.show()

# %%
