import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers


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
        super(RVAE,self).init()

        #dimension of the input space
        self.input_shape = (29,)        #X.shape = (..., 29, ..)

        #set the dimension of the latent vector
        self.latent_dim = 2
        

        

    def encoder(self,):
