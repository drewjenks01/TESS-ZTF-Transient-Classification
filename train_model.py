"""
imports:
    - NN model
    - processed data

functions:
    - train
        initialize VAE
        compile with optimizer
        fit to data
            hyperparams: epochs, batch_size
"""

#%%
from keras.optimizers import adam_v2
from build_model import BuildModel
from pre_process import load_augmented


class NNTrainer:

    def __init__(self):
        super(NNTrainer, self).init()

        # optimizer
        self.optimizer = adam_v2.Adam(learning_rate=1e-4)

        # training epochs
        self.epochs = 20

        # batch size
        self.batch_size = 64

        # save and load filepath
        self.filepath = '/Users/drewj/Documents/Urops/Muthukrishna/model/'

    def train(self):

        #import data
        light_curves = load_augmented()

        # initialize RVAE
        rvae = BuildModel

        # build model
        model = rvae.build_model()

        # compile model with optimizer
        model.compile(optimizer=self.optimizer)

        # fit model
        model.fit(light_curves, epochs=self.epochs, batch_size=self.batch_size)

        return model

    def save_model(self, model):
        model.save(self.filepath+'rvae')


def main():
    trainer = NNTrainer
    model = trainer.train()
    trainer.save_model(model)


if __name__ == "__main__":
    main()

# %%
