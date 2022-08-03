# %%

import random
import sys

import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import Input, Model
from keras.callbacks import EarlyStopping
from keras.layers import (GRU, BatchNormalization, Dense, Embedding, Lambda,
                          Layer, Masking, RepeatVector, TimeDistributed,
                          concatenate)
from keras.models import load_model
from keras.optimizers import adam_v2
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from tcn import TCN
from tensorflow.python.framework.ops import disable_eager_execution
from tqdm import tqdm

from pre_process import load_aug_dataframe

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


disable_eager_execution()
"""
To test:
    - LSTM, TCN

Notes:
    - old model structrue 2x faster

Best model so far:
    - 250 epochs
    - 128 batch size
    - latent dim =35
    - 120 units each
    - all normalized
    - mask func in Lambda and GRU
    - connected NN
    - 1e-3 lr
    - negative loss
    - half day bins
    - final loss: -4.4278



unique classifiers:  {('SNI','SNIa','SNIa-91T-like','SNIa-91bg-like','SNIa-pec', 'SNIa-SC'): 0,
                      ('SNIbn','SNIb/c','SNIb','SNIc','SNIc-BL'): 1,
                      ('SNII', 'SNIIb','SNIIP','SNII-pec','SNIIn'): 2, 
                       ('CV', 'SLSN-I','AGN', 'FRB','Mdwarf', 
                        'Nova', 'Other', 'Varstar'): 3,
                        'Unclassified': 4 }

class counts:  {0: 245, 1: 15, 2: 62, 3: 20, 4: 2026}

"""


class InputReduction(Layer):
    def __init__(self, mask):
        super(InputReduction, self).__init__()
        self.masker = mask

    def call(self, inputs):
        m = self.masker
        print(inputs.shape)
        tf.print(m, output_stream=sys.stdout)
        inputs = tf.boolean_mask(inputs, m)
        print(inputs.shape)

        return inputs


class RVAE:
    """ Recurrent Variational Autoencoder """

    def __init__(self):

        # training epochs
        self.epochs = 200

        # batch size
        self.batch_size = 128

        # save and load filepath
        self.filepath = '/Users/drewj/Documents/Urops/Muthukrishna/'

        # optimizer
        self.optimizer = adam_v2.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999,
                                      decay=0)

        # set the dimension of the latent vector
        self.latent_dim = 35

        # input to first enc, second dec layer
        self.gru_one = 120

        # input to first dec, second enc layer
        self.gru_two = 120

        # load prepared data (acts as input)
        self.prepared_data = np.load(self.filepath+'data/prepared_data.npy')

        # number of input features
        self.num_feats = self.prepared_data.shape[2]

        self.num_timesteps = 46

        # dimension of the input space for enc
        # X.shape = (..., 30, ..)
        self.enc_input_shape = (self.num_timesteps, self.num_feats)

        # number of light curves
        self.num_lcs = 3155

        # indxs for test and train
        self.train_indx = set()
        self.test_indx = set()

        # test and train data
        self.train_test_data = self.split_prep_data()

    def build_disconnected_model(self):
        """
        Builds entire RVAE model with encoder and decoder as two seperate Models

        Returns:
            model : v=rvae model
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
            model : v=rvae model
        """

        # BUILD ENCODER
        print("building encoder...")

        # input layer
        enc_input = Input(shape=self.enc_input_shape)

        # masking layer to get rid of unwanted values
        mask = Masking(mask_value=0.)

        mask_compute = mask(enc_input)

        mask_output = mask.compute_mask(enc_input)

        # first recurrent layer
        gru1 = GRU(self.gru_one, activation='tanh',
                   recurrent_activation='hard_sigmoid', return_sequences=True, name='gru1')(mask_compute)

        #bn1 = BatchNormalization()(gru1)

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

        repeater._keras_mask = mask_output

        # first recurrent layer
        x = GRU(self.gru_two, activation='tanh',
                recurrent_activation='hard_sigmoid', return_sequences=True, name='gru5')(repeater)

       # bn = BatchNormalization()(x)

        # second recurrent layer
        x = GRU(self.gru_one, activation='tanh',
                recurrent_activation='hard_sigmoid', return_sequences=True, name='gru6')(x)

        # decoder output
        dec_output = TimeDistributed(
            Dense(1, activation='tanh', input_shape=(None, 1)), name='td')(x)

        # BUILD MODEL
        rvae = Model(enc_input, dec_output)
        rvae.summary()

        # for i, l in enumerate(decoder.layers):
        #     print(f'layer {i}: {l}')
        #     print(f'has input mask: {l.input_mask}')
        #     print(f'has output mask: {l.output_mask}')

        for i, l in enumerate(rvae.layers):
            print(f'layer {i}: {l}')
            print(f'has input mask: {l.input_mask}')
            print(f'has output mask: {l.output_mask}')

        # define loss
        custom_loss = self.vae_loss(z_mean, z_log_var)

        # compile model
        rvae.compile(optimizer=self.optimizer, loss=custom_loss)

        es = EarlyStopping(monitor='val_loss', min_delta=0, patience=20,
                           verbose=1, mode='min', baseline=None,
                           restore_best_weights=True)

        return rvae, encoder, es

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

       # @tf.function

        def lossFunction(yTrue, yPred):
            reconstruction_loss = K.log(K.mean(K.square(yTrue - yPred)))
            #reconstruction_loss = 30*K.mean(K.square(yTrue - yPred))

            # tf.print('rec: ',reconstruction_loss,output_stream=sys.stdout)
            # tf.print('kl: ', kl_loss,output_stream=sys.stdout)

            return reconstruction_loss + kl_loss

        return lossFunction

    def split_prep_data(self):
        """
        Splits data into 3/4 training, 1/4 testing
        """

        print("Splitting data into train and test...")

        # prepared out (only flux)
        prep_out = self.prepared_data[:, :, 0].reshape(
            self.num_lcs, self.num_timesteps, 1)

        prep_inp = self.prepared_data

        x_train = []
        y_train = []
        x_test = []
        y_test = []

        # calc the # of light curves for train vs test
        num_lcs = len(prep_inp)
        train_perc = round(0.75 * num_lcs)
        test_perc = num_lcs-train_perc

        # save random indices for training
        while len(self.train_indx) != train_perc:
            indx = random.randint(0, num_lcs-1)
            self.train_indx.add(indx)

        # save random indices for testint -> no duplicates from training
        while len(self.test_indx) != test_perc:
            indx = random.randint(0, num_lcs-1)
            if indx not in self.train_indx:
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

        return [x_train, x_test, y_train, y_test]

    def save_model(self, model, name):
        """
        Saves the model
        """
        print('saving model: ', name)
        model.save(self.filepath+name)

    def get_encoder(self):
        """
        Loads the encoder model
        """
        enc = load_model(self.filepath+'model/encoder', custom_objects={'sampling': self.sampling,
                                                                        'lossFunction': self.customLoss}, compile=False)

        return enc

    def train_model(self, model, es):
        """
        Trains the NN on training data

        Returns the trained model
        """
        # fit model
        train_inp = self.train_test_data[0]
        train_out = self.train_test_data[2]
        test_inp = self.train_test_data[1]
        test_out = self.train_test_data[3]

        print('fitting model...')
        history = model.fit(train_inp, train_out, epochs=self.epochs, batch_size=self.batch_size,
                            validation_data=(test_inp, test_out), verbose=1, callbacks=[es], shuffle=False)

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(self.filepath+'plots/training_history.png',
                    facecolor='white')

        return model

    def test_model(self, rvae=None):
        """
        Uses test data to and NN to predict light curve decodings
        """

        test_inp = self.train_test_data[1][:100]
        test_inp = test_inp.reshape(-1, 1, self.num_timesteps, self.num_feats)
        test_out = self.train_test_data[3][:100]

        # load model if none passed
        if not rvae:
            print('loading model for testing...')
            rvae = load_model(self.filepath+'model/rvae', custom_objects={'sampling': self.sampling,
                                                                          'lossFunction': self.customLoss})
        rvae.summary()

        # for each light curve, use model to predict decoded output
        # and plot against raw data to compare
       # avg_mse=()
        inpt_length = len(test_inp)
        indxs = set()
        for i in range(9):
            indxs.add(random.randint(0, inpt_length-1))

        print('predicting...')
        for i in tqdm(range(len(test_inp))):

            # predicted flux
            predicted = rvae.predict(test_inp[i])[0]

            # if first prediction, print the prediction
            if i == 0:
                print('shape of predicted data: ', predicted.shape)

            # if one of first 10 predictions, plot prediction vs true
            if i in indxs:
                self.plot_true_pred(test_out[i], predicted, i)

            # calulcate mse
            # mse = np.mean(np.square(predicted-test_out[i]))
            # avg_mse.add(mse)
            #print('mse of test '+str(i)+': ',mse)

       # avg_mse=np.array(avg_mse)
       # print('avg mse of'+str(len(test_inp))+ 'tests: ',np.mean(avg_mse))

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
        fig.savefig(self.filepath+'plots/raw_vs_pred' +
                    str(num)+'.png', facecolor='white')

    def t_SNE_plot(self):
        # display a 2D plot of the light curves in the latent space
        print('using t-SNE...')
        aug_df = load_aug_dataframe()

        #labeled
        indxs = (aug_df.loc[aug_df.loc[:, 'Class'] != 4].index)
        labels = [aug_df.loc[i,'Class'] for i in indxs]
        labeled_data =np.array( [self.prepared_data[i] for i in indxs])

        encoder = self.get_encoder()
        _, _, z = tqdm(encoder.predict(labeled_data))

        t_sne = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(z)
        print('t-sne shape: ',t_sne.shape)

        plt.figure(figsize=(12, 10))
        plt.scatter(t_sne[:, 0], t_sne[:, 1], c=labels)
        plt.colorbar()
        plt.title("t-SNE with only labeled data")
        plt.savefig(self.filepath+'plots/labeled-t-sne-latent-space.png', facecolor='white')


        #include unlabeled
        labels = aug_df.loc[:,'Class']
        data = self.prepared_data

        encoder = self.get_encoder()
        _, _, z = tqdm(encoder.predict(data))

        t_sne = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(z)
        print('t-sne shape: ',t_sne.shape)

        plt.figure(figsize=(12, 10))
        plt.scatter(t_sne[:, 0], t_sne[:, 1], c=labels)
        plt.colorbar()
        plt.title("t-SNE with labeled and unlabeled data")
        plt.savefig(self.filepath+'plots/unlabeled-t-sne-latent-space.png', facecolor='white')


def main():

    # initialize rvae
    rvae = RVAE()

#     # build the model
#     model, encoder, es = rvae.build_connected_model()

#     # train model
#     trained_rvae = rvae.train_model(model, es)

#     # save model
#     rvae.save_model(encoder, 'model/encoder')
#    # rvae.save_model(trained_rvae, 'model/rvae')

#     # load model
#     rvae.test_model(trained_rvae)
    rvae.t_SNE_plot()


if __name__ == "__main__":
    main()

# %%
