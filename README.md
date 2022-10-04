# TESS-ZTF-Transient-Classification

Research project in the MIT Kavli Institute for Astrophysics and Space Research.

PI: George Ricker

Supervisor: Daniel Muthukrishna

Processes raw light curve data from the NASA TESS satelite and Zwicky Transient Facility (ZTF).

Uses a recurrent variational autoencoder model to reduce the dimensionality of the light curves.

Uses a balanced random forest classifier to classify unlabled light curves.

------------------
# Use

This project is set up as a pipeline, with the pipeline.ipynb file being the main orchestrator.

There are three steps within the pipeline:

## Step 1: Preprocessing (pre-process.py)

Reads in raw light curve data and classification info and processes the the data.

Processing steps include:
- numeric encodings of class
- use of TESS/ZTF filter IDs
- timestep creation for each filter ID occurance
- cut light curves between specific time range
- skip light curves with no data
- data augmentation so all light curves have same # of timesteps
    

## Step 2: Recurrent Variational Autoencoder (NN_model.py)
*Unsupervised*
    
Builds a variational autoencoder that takes in time-series
light curve data and produces lower-dimensional representations to be used
for classificiation.

Trains and tests the model, extracts the encoder.

Plots a 2D t-SNE representation of light curves in their latent space.

## Step 3: Balanced Random Forest Classifier (classify.py)
*Supervised*

Creates a Balanced Random Forest Classifier that takes in encoded light curves and classifies them.

Uses the trained encoder from the RVAE model to encode light curves.

Trains the classifier on labeled data, tests on both labeled and unlabeled.
