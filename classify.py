#%%
from NN_model import RVAE
from pre_process import load_augmented,load_data
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics 
import pandas as pd
import numpy as np
from tqdm import tqdm

class RandomForest:

    def __init__(self):
        #initialize rvae
        self.rvae = RVAE()

        # get training encoder
        self.encoder = self.rvae.get_encoder()

        #filepath
        self.filepath = '/Users/drewj/Documents/Urops/Muthukrishna/data/'

    def classifier_df(self):
        """
        Create DataFrame used for RF Classifier

        DF: (filename abr, flux, error, classification)
        """
        print('Creating RF dataframe')

        light_curves= load_data()['light_curves']
        all_transients = load_data()['all_transients']

        df_dict=[]

        #define columns
        columns = ['Filename', 'Flux','Error', 'Class']

        #loop through AT and create dict for name_abr -> class
        at_files = all_transients.loc[:,'IAU name']
        at_class = all_transients.loc[:, 'classification']

        at_dict={}

        for i in range(len(at_files)):
            at_dict[at_files[i]] = at_class[i]

        #loop through light_curve file
        for lc in tqdm(light_curves):
            #extract filename, flux, error
            flux = np.array(lc.loc[:, 'cts'])
            error = np.array(lc.loc[:, 'e_cts'])
            filename = np.array(lc.loc[:,'filename'])[0]

            #normalize flux & error b/w [-1,1]
            flux = 2 * (flux-np.min(flux))/(np.max(flux)-np.min(flux)) - 1
            error = 2 * (error-np.min(error))/(np.max(error)-np.min(error)) - 1

            #process filename to be just abreviation ex: 2018evo
            name_split = filename.split('_')
            name_abr = name_split[1]

            #add data to DataFrame dict list
            df_dict.append({'Filename':name_abr,'Flux':flux,'Error':error,'Class':at_dict[name_abr]})

        #create dataframe
        df = pd.DataFrame(df_dict, columns=columns)

        #save dataframe
        df.to_csv(self.filepath+'classifier_data.csv', index=False)

        return df

    def build_classier(self):

        #initialize random forest classifier
        rf = RandomForestClassifier()

        #train using train data
        x_train = self.train_test_data[0]
        x_test = self.train_test_data[1]
        y_train = self.train_test_data[2]
        y_test = self.train_test_data[3]
        rf.fit(x_train, y_train)

        # performing predictions on the test dataset
        y_pred = rf.predict(x_test)

        #check accuracy
        print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))

        return rf


    def classify(self):
        return


def main():

    #initialize random forest
    rf=RandomForest()

    #build dataframe
    df = rf.classifier_df()

    print(df)


if __name__ == "__main__":
    main()
# %%
