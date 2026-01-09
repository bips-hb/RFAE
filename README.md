
# Autoencoding Random Forests #
Autoencoding random forests ('RFAEs') provide a method to autoencode data using random forests, which involves projecting the data to a latent feature space of chosen dimensionality (usually a lower dimension), and then decoding the latent representations back into the input space. The encoding stage is useful for feature engineering and data visualisation tasks, akin to how principal component analysis ('PCA') is used , and the decoding stage is usefulfor compression and denoising tasks. At its core, 'RFAE' is a post-processing pipeline on a trained random forest model. This means that it can accept any trained RF of ranger object type: 'RF', 'URF' or ARFs'. Because of this, it inherits RFs' robust performance and capacity to seamlessly handle mixed-type tabular data. 

The package can be installed by running:
```
devtools::install_github("bips-hb/RFAE")
```
You can also clone the repository and run:
```
devtools::build("RFAE")
```
# Examples

Using Fisher's iris dataset, we train a RF and pass it through the autoencoding pipeline:

```
# Set seed
set.seed(1)
# Split training and test
trn <- sample(1:nrow(iris), 100)
tst <- setdiff(1:nrow(iris), trn)
# Train RF
rf <- ranger::ranger(Species ~ ., data = iris[trn, ], num.trees=50)
```
Encode data and project test data to create new embeddings:
```
# Fit encoder object
emap <- encode(rf, iris[trn, ], k=2)
# Embed new test samples
emb <- predict(emap, rf, iris[tst, ])
```
Decode test samples back to the input space: 
```
# Decode samples
out <- decode_knn(rf, emap, emb, k=5)$x_hat
```
Measure the reconstruction error between decoded and actual samples:
```
error <- reconstruction_error(out, iris[tst, ])
```

For more detailed examples, refer to the package vignette.

# Python Library
The Python version of RFAE is currently under development. A preliminary version is currently available at [RFAE_py](https://github.com/binhducvu/RFAE_py)

# References
- Vu, B. D., Kapar, J., Wright, M., & Watson, D. S. (2025). Autoencoding Random Forests. arXiv preprint arXiv:2505.21441. Link [here](https://arxiv.org/abs/2505.21441) - NeurIPS version coming soon!
