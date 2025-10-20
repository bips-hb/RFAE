from ctgan import TVAE
import pandas as pd
import torch
from joblib import Parallel, delayed

def single_tvae(dat, latent_rate, discrete_columns, full, bootstraps, run):
    bootstrap = bootstraps.iloc[:, run]
    trn = full.loc[bootstrap].reset_index().drop('index',axis=1)
    tst = full.drop(index=bootstrap).reset_index().drop('index',axis=1)
    latent_layer = max(round(latent_rate * trn.shape[1]), 1)
    TVAE_model = TVAE(embedding_dim = latent_layer)
    TVAE_model.fit(trn, discrete_columns)

    transformed = TVAE_model.transformer.transform(tst)
    transformed = torch.from_numpy(transformed.astype('float32')).to('cuda:0')
    mu, std, logvar = TVAE_model.encoder(transformed)
    emb = mu

    rec, sigmas = TVAE_model.decoder(emb)
    rec = torch.tanh(rec)
    tst_rec = rec.detach().cpu().numpy()
    tst_rec = TVAE_model.transformer.inverse_transform(tst_rec, sigmas.detach().cpu().numpy())

    tst_rec.to_csv("../tvae_data/" + dat + '/' + str(latent_rate) + '_run' + str(run+1) +  ".csv", index=False)

def tvae_parallel(dat, latent_rate=0.2, runs=10, n_jobs=5):
    full = pd.read_csv('../original_data/full/' + dat + '.csv')
    bootstraps = pd.read_csv('../original_data/full/bootstrap_' + dat + '.csv')
    bootstraps = bootstraps - 1
    discrete_columns = full.columns[full.dtypes == 'object']

    Parallel(n_jobs = n_jobs)(
    delayed(single_tvae)(dat, latent_rate, discrete_columns, full, bootstraps, i)
    for i in range(runs)
    )
    print("Finished " + dat)
    return None


compressions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

for compression in compressions:
    tvae_parallel('credit', compression)
for compression in compressions:
    tvae_parallel('adult', compression)
for compression in compressions:
    tvae_parallel('abalone', compression)
for compression in compressions:
    tvae_parallel('bc', compression)
for compression in compressions:
    tvae_parallel('car', compression)
for compression in compressions:
    tvae_parallel('banknote', compression)
for compression in compressions:
    tvae_parallel('churn', compression)
for compression in compressions:
    tvae_parallel('credit', compression)
for compression in compressions:
    tvae_parallel('diabetes', compression)
for compression in compressions:
    tvae_parallel('dry_bean', compression)
for compression in compressions:
    tvae_parallel('forestfires', compression)
for compression in compressions:
    tvae_parallel('hd', compression)
for compression in compressions:
    tvae_parallel('king', compression)
for compression in compressions:
    tvae_parallel('marketing', compression)
for compression in compressions:
    tvae_parallel('mushroom', compression)
for compression in compressions:
    tvae_parallel('obesity', compression)
for compression in compressions:
    tvae_parallel('plpn', compression)
for compression in compressions:
    tvae_parallel('spambase', compression)
for compression in compressions:
    tvae_parallel('student', compression)
for compression in compressions:
    tvae_parallel('telco', compression)
for compression in compressions:
    tvae_parallel('wq', compression)