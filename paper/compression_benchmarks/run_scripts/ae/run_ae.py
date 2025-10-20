import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, Dropout
from tensorflow.keras.regularizers import L1
from tensorflow.keras import backend as K
import keras
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from joblib import Parallel, delayed

def none_fix(arr):

    n_rows, n_cols = arr.shape

    for j in range(n_cols):
        col = arr[:, j]

        null_mask = np.array([x is None for x in col])
        
        if not null_mask.any():
            continue

        non_null = col[~null_mask]

        # sample replacements
        replacements = np.argmax(non_null, size=null_mask.sum(), replace=True)
        arr[null_mask, j] = replacements

    return arr
def single_ae(dat, latent_rate, full, bootstraps, run):
    latent_dim = max(round(latent_rate * full.shape[1]), 1)
    bootstrap = bootstraps.iloc[:, run]
    trn = full.loc[bootstrap].reset_index().drop('index',axis=1)
    tst = full.drop(index=bootstrap).reset_index().drop('index',axis=1)
    trn, val = train_test_split(trn, test_size = 0.1)
    trn = trn.reset_index().drop('index', axis=1)
    val = val.reset_index().drop('index', axis=1)

    enc_true = list(trn.columns[trn.dtypes == 'object'])
    # encoding and minmax scaling, fit on train, transform on rest
    if enc_true:
        enc_cols = trn.select_dtypes('object')
        enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded = pd.DataFrame(enc.fit_transform(enc_cols), 
                          columns = enc.get_feature_names_out())
        trn.drop(enc_cols.columns, axis=1, inplace=True)
        trn = pd.concat([trn, encoded], axis=1)

        encoded=pd.DataFrame(enc.transform(tst[enc.feature_names_in_]), 
                            columns = enc.get_feature_names_out())
        tst.drop(enc_cols.columns, axis=1, inplace=True)
        tst = pd.concat([tst, encoded], axis=1)

        encoded=pd.DataFrame(enc.transform(val[enc.feature_names_in_]),
                            columns = enc.get_feature_names_out())
        val.drop(enc_cols.columns, axis=1, inplace=True)
        val  = pd.concat([val, encoded], axis=1)

    cols = trn.columns
    scaler = MinMaxScaler()
    trn = scaler.fit_transform(trn)
    tst = scaler.transform(tst)
    val = scaler.transform(val)

    # define autoencoder
    input_dim = trn.shape[1]
    input_layer = Input(shape=(input_dim, ))
    first_dim = round(input_dim - (input_dim - latent_dim) * 1/3)
    second_dim = round(input_dim - (input_dim - latent_dim) * 2/3)

    encoded = Dense(first_dim, activation='relu')(input_layer)
    encoded = Dense(second_dim, activation = 'relu')(encoded)
    latent = Dense(latent_dim,activation='relu', name="latent_space")(encoded)

    decoded = Dense(second_dim, activation='relu')(latent)
    decoded = Dense(first_dim, activation='relu')(decoded)
    output_layer = Dense(input_dim, activation='sigmoid')(decoded)

    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer='adam', loss='mse')
    # fit and write to file
    history = autoencoder.fit(
    trn, trn,
    epochs=50,
    validation_data=(val, val),
    verbose=0)
    ae = autoencoder.predict(tst)
    ae = pd.DataFrame(scaler.inverse_transform(ae), columns = cols)
    if enc_true:   
        enc_cols = none_fix(enc.inverse_transform(ae[enc.get_feature_names_out()]))
        ae = ae.drop(enc.get_feature_names_out(), axis=1)
        ae[enc.feature_names_in_] = enc_cols
    ae.to_csv("../ae_data/" + dat + '/' + str(latent_rate) + '_run' + str(run+1) +  ".csv", index=False)

def ae(dat, latent_rate = 0.2, runs=10, n_jobs=10):
    # read data, drop first col if needed, split into train/val/test

    full = pd.read_csv('../original_data/full/' + dat + '.csv')
    bootstraps = pd.read_csv('../original_data/full/bootstrap_' + dat + '.csv')
    bootstraps = bootstraps - 1

    Parallel(n_jobs = n_jobs)(
    delayed(single_ae)(dat, latent_rate, full, bootstraps, i)
    for i in range(runs)
    )
    print("Finished " + dat)
    return None


compressions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
for compression in compressions:
    ae('adult', compression)
for compression in compressions:
    ae('abalone', compression)
for compression in compressions:
    ae('bc', compression)
for compression in compressions:
    ae('car', compression)
for compression in compressions:
    ae('banknote', compression)
for compression in compressions:
    ae('churn', compression)
for compression in compressions:
    ae('credit', compression)
for compression in compressions:
    ae('diabetes', compression)
for compression in compressions:
    ae('dry_bean', compression)
for compression in compressions:
    ae('forestfires', compression)
for compression in compressions:
    ae('hd', compression)
for compression in compressions:
    ae('king', compression)
for compression in compressions:
    ae('marketing', compression)
for compression in compressions:
    ae('mushroom', compression)
for compression in compressions:
    ae('obesity', compression)
for compression in compressions:
    ae('plpn', compression)
for compression in compressions:
    ae('spambase', compression)
for compression in compressions:
    ae('student', compression)
for compression in compressions:
    ae('telco', compression)
for compression in compressions:
    ae('wq', compression)