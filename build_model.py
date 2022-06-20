from tensorflow.python.keras import Model
from neural_network_model import RVAE

class BuildModel:

    def build_model():

        #initialize VAE
        rvae= RVAE()

        #build encoder
        encoder = rvae.build_encoder()

        #build repeater
        repeater = rvae.build_repeater()

        #build decoder
        decoder = rvae.build_decoder()

        #build model
        model = Model(encoder, repeater, decoder)

        return model
