from model import *
import pandas as pd
from joblib import Parallel, delayed
def single_ttvae(dat, latent_rate, discrete_columns, full, bootstraps, run):
    run = run
    bootstrap = bootstraps.iloc[:, run]
    trn = full.loc[bootstrap].reset_index().drop('index', axis=1)
    tst = full.drop(index=bootstrap).reset_index().drop('index', axis=1)
    latent_layer = max(round(latent_rate * full.shape[1]), 1)
    ttvae = TTVAE(latent_dim=latent_layer)
    ttvae.fit(trn, discrete_columns, '../TTVAE/ckpt')
    transformed = ttvae.transformer.transform(tst)
    transformed = torch.from_numpy(transformed.astype('float32')).to('cuda:0')

    mu, std, logvar, enc_embed = ttvae.encoder(transformed)
    synthetic_embeddings=mu
    noise = mu
    ttvae.decoder.eval()
    with torch.no_grad():
        fake, sigmas = ttvae.decoder(noise,enc_embed)
        fake = torch.tanh(fake)
    fake_np = fake.cpu().detach().numpy()
    tst_rec = ttvae.transformer.inverse_transform(fake_np)

    tst_rec.to_csv("../ttvae_data/" + dat + '/' + str(latent_rate) + '_run' + str(run+1) +  ".csv", index=False)
def ttvae_parallel(dat, latent_rate=0.2, runs=10, n_jobs=4):
    full = pd.read_csv('../original_data/full/' + dat + '.csv')
    bootstraps = pd.read_csv('../original_data/full/bootstrap_' + dat + '.csv')
    bootstraps = bootstraps - 1
    discrete_columns = full.columns[full.dtypes == 'object']

    Parallel(n_jobs = n_jobs)(
	delayed(single_ttvae)(dat, latent_rate, discrete_columns, full, bootstraps, i)
	for i in range(runs)
    )
    print("Finished " + dat)
    return None


compressions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]


for compression in compressions:
    ttvae_parallel('adult', compression)
for compression in compressions:
    ttvae_parallel('abalone', compression)
for compression in compressions:
    ttvae_parallel('bc', compression)
for compression in compressions:
    ttvae_parallel('car', compression)
for compression in compressions:
    ttvae_parallel('banknote', compression)
for compression in compressions:
    ttvae_parallel('churn', compression)
for compression in compressions:
    ttvae_parallel('credit', compression)
for compression in compressions:
    ttvae_parallel('diabetes', compression)
for compression in compressions:
    ttvae_parallel('dry_bean', compression)
for compression in compressions:
    ttvae_parallel('forestfires', compression)
for compression in compressions:
    ttvae_parallel('hd', compression)
for compression in compressions:
    ttvae_parallel('king', compression)
for compression in compressions:
    ttvae_parallel('marketing', compression)
for compression in compressions:
    ttvae_parallel('mushroom', compression)
for compression in compressions:
    ttvae_parallel('obesity', compression)
for compression in compressions:
    ttvae_parallel('plpn', compression)
for compression in compressions:
    ttvae_parallel('spambase', compression)
for compression in compressions:
    ttvae_parallel('student', compression)
for compression in compressions:
    ttvae_parallel('telco', compression)
for compression in compressions:
    ttvae_parallel('wq', compression)
