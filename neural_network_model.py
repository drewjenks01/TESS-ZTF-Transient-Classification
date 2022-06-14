from pytest import skip
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers, Input, Model
from tensorflow.python.keras.layers import Dense, GRU, Masking, TimeDistributed, RepeatVector
from tensorflow.python.keras.metrics import Mean
from keras.optimizers import adam_v2


"""
Questions:
    -TimeDistributed not working?
"""


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class RVAE(keras.Model):
    """ Recurrent Variational Autoencoder """

    def __init__(self):
        super(RVAE, self).init()

        # dimension of the input space for enc
        self.enc_input_shape = (2, 30)  # X.shape = (..., 29, ..)

        # set the dimension of the latent vector
        self.latent_dim =

        # input to first enc, second dec layer
        self.gru_one =

        # input to first dec, second enc layer
        self.gru_two =

        # number of filters
        self.nfilts = 1

        # train_step args
        self.total_loss_tracker = Mean(name="total_loss")
        self.reconstruction_loss_tracker = Mean(name="reconstruction_loss")
        self.kl_loss_tracker = Mean(name="kl_loss")

        #optimizer
        self.optimizer = adam_v2.Adam(learning_rate=1e-4)

    def encoder(self):
        """
        Builds encoder model

        Returns:
            model : encoder model
        """

        # input layer
        input = Input(shape=self.enc_input_shape)

        # masking layer to get rid of unwanted values
        mask = Masking(mask_value=-1)

        # first recurrent layer
        x = GRU(self.gru_one, activation='tanh',
                recurrent_activation='hard_sigmoid', return_sequences=True)(mask)

        # second recurrent layer
        x = GRU(self.gru_two, activation='tanh',
                recurrent_activation='hard_sigmoid', return_sequences=True)(x)

        # z mean output
        z_mean = GRU(
            self.latent_dim, return_sequences=False, activation='linear')(x)

        # z variance output
        z_log_var = GRU(
            self.latent_dim, return_sequences=False, activation='linear')(x)

        # sample output
        z = Sampling()([z_mean, z_log_var])

        # define encoder
        encoder = Model(input, [z_mean, z_log_var, z], name='encoder')
        encoder.summary()

        return encoder

    def repeatLayer(self):
        skip

    def decoder(self):
        """
        Builds decoder model

        Returns:
            model : encoder model
        """

       # input layer
        input = Input(shape=(self.latent_dim,))

        # first recurrent layer
        x = GRU(self.gru_two, activation='tanh',
                recurrent_activation='hard_sigmoid', return_sequences=True)(input)

        # second recurrent layer
        x = GRU(self.gru_one, activation='tanh',
                recurrent_activation='hard_sigmoid', return_sequences=True)(x)

        # decoder output
        output = TimeDistributed(Dense(self.nfilts, activation='tanh'))(x)

        # define encoder
        decoder = Model([input, ], x, name='decoder')
        decoder.summary()

        return decoder

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    @tf.function
    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1,)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var -
                              tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
