import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.regularizers import L1
from tensorflow.keras import backend as K
import keras
from keras import ops
from keras import layers
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

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seed_generator = keras.random.SeedGenerator()

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = ops.shape(z_mean)[0]
        dim = ops.shape(z_mean)[1]
        epsilon = keras.random.normal(shape=(batch, dim), seed=self.seed_generator)
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon

class VEncoder(keras.Model):
    def __init__(self, input_dim: int, latent_dim:int, layer_1:int,
                 layer_2:int, **kwargs):
        super().__init__(**kwargs)

        encoder_inputs = keras.Input(shape=(input_dim,))
        x = layers.Dense(layer_1, activation = "relu")(encoder_inputs)
        x = layers.Dense(layer_2, activation = "relu")(x)
        z_mean = layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        self.encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    def call(self, x):
        return self.encoder(x)

class VDecoder(keras.Model):
    def __init__(self, latent_dim:int, output_dim:int, layer_1:int,
                 layer_2:int, **kwargs):
        super().__init__(**kwargs)

        latent_inputs = keras.Input(shape=(latent_dim,))
        x = layers.Dense(layer_2, activation = "relu")(latent_inputs)
        x = layers.Dense(layer_1, activation = "relu")(x)
        decoder_outputs = layers.Dense(output_dim, activation="sigmoid")(x)
        
        self.decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

    def call(self, x):
        return self.decoder(x)


class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
    
    def call(self, x):
        z_mean, z_log_var, z = self.encoder(x)
        reconstruction = self.decoder(z)
        return z_mean, z_log_var, reconstruction

    # also used during validation
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]
    
    def calculate_reconstruction_loss(self, data, reconstruction):
        """
        In case of computer vision tasks use the following:
            keras.losses.binary_crossentropy(data, reconstruction),
            axis=(1, 2),
        """
        return ops.mean(ops.sum(keras.losses.binary_crossentropy(data, reconstruction)))
    
    def calculate_kl_loss(self, z_mean, z_log_var):
        kl_loss = -0.5 * (1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))
        kl_loss = ops.mean(ops.sum(kl_loss, axis=1))
        return kl_loss
    
    def calculate_total_loss(self, reconstruction_loss, kl_loss):
        return reconstruction_loss + 0.1 * kl_loss 
        
    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            
            reconstruction_loss = self.calculate_reconstruction_loss(data, reconstruction)
            kl_loss = self.calculate_kl_loss(z_mean, z_log_var)
            total_loss = self.calculate_total_loss(reconstruction_loss, kl_loss)

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
    
    def test_step(self, data):
        z_mean, z_log_var, z = self.encoder(data, training=False)
        reconstruction = self.decoder(z)
        
        reconstruction_loss = self.calculate_reconstruction_loss(data, reconstruction)
        kl_loss = self.calculate_kl_loss(z_mean, z_log_var)
        total_loss = self.calculate_total_loss(reconstruction_loss, kl_loss)
        
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    
    
def single_vae(dat, latent_rate, full, bootstraps, run):
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
    first_dim = round(input_dim - (input_dim - latent_dim) * 1/3)
    second_dim = round(input_dim - (input_dim - latent_dim) * 2/3)
    encoder = VEncoder(input_dim, latent_dim, first_dim, second_dim)
    decoder = VDecoder(latent_dim, input_dim, first_dim, second_dim)
    vautoencoder = VAE(encoder, decoder)
    vautoencoder.compile(optimizer='adam')
    history = vautoencoder.fit(trn,
                      epochs = 50, 
                      validation_data = val,
                      shuffle = True,
                      verbose=0)
    vae = vautoencoder.predict(tst)[2]
    vae = pd.DataFrame(scaler.inverse_transform(vae), columns = cols)
    if enc_true:
        enc_cols = none_fix(enc.inverse_transform(vae[enc.get_feature_names_out()]))
        vae = vae.drop(enc.get_feature_names_out(), axis=1)
        vae[enc.feature_names_in_] = enc_cols
    vae.to_csv("../vae_data/" + dat + '/' + str(latent_rate) + '_run' + str(run+1) +  ".csv", index=False)

def vae(dat, latent_rate = 0.2, runs=10, n_jobs=5):
    # read data, drop first col if needed, split into train/val/test

    full = pd.read_csv('../original_data/full/' + dat + '.csv')
    bootstraps = pd.read_csv('../original_data/full/bootstrap_' + dat + '.csv')
    bootstraps = bootstraps - 1

    Parallel(n_jobs = n_jobs)(
    delayed(single_vae)(dat, latent_rate, full, bootstraps, i)
    for i in range(runs)
    )
    print("Finished " + dat)
    return None


compressions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

for compression in compressions:
    vae('adult', compression)
for compression in compressions:
    vae('abalone', compression)
for compression in compressions:
    vae('bc', compression)
for compression in compressions:
    vae('car', compression)
for compression in compressions:
    vae('banknote', compression)
for compression in compressions:
    vae('churn', compression)
for compression in compressions:
    vae('credit', compression)
for compression in compressions:
    vae('diabetes', compression)
for compression in compressions:
    vae('dry_bean', compression)
for compression in compressions:
    vae('forestfires', compression)
for compression in compressions:
    vae('hd', compression)
for compression in compressions:
    vae('king', compression)
for compression in compressions:
    vae('marketing', compression)
for compression in compressions:
    vae('mushroom', compression)
for compression in compressions:
    vae('obesity', compression)
for compression in compressions:
    vae('plpn', compression)
for compression in compressions:
    vae('spambase', compression)
for compression in compressions:
    vae('student', compression)
for compression in compressions:
    vae('telco', compression)
for compression in compressions:
    vae('wq', compression)