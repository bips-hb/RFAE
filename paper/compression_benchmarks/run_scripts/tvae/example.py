from sdv.metadata import SingleTableMetadata
from sdv.single_table import TVAESynthesizer
import pandas as pd
import torch

# load data
adult = pd.read_csv("decoder_sandbox/original_data/adult.csv")

# fit TVAE
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(adult)
TVAE_model = TVAESynthesizer(metadata)
TVAE_model.fit(adult)

TVAE_model._model.encoder.eval()
TVAE_model._model.decoder.eval()

# encode
adult_transformed = TVAE_model._data_processor.transform(adult)
adult_transformed = TVAE_model._model.transformer.transform(adult)
adult_transformed = torch.from_numpy(adult_transformed.astype('float32')).to("cuda:0")
mu, std, logvar = TVAE_model._model.encoder(adult_transformed)
eps = torch.randn_like(std)
emb = eps * std + mu

# decode
rec, sigmas = TVAE_model._model.decoder(emb)
adult_rec = torch.tanh(rec).detach().cpu().numpy()
adult_rec = TVAE_model._model.transformer.inverse_transform(adult_rec, sigmas.detach().cpu().numpy())
adult_rec = TVAE_model._data_processor.reverse_transform(adult_rec)
adult_rec