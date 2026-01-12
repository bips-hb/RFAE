# Decode RF Embeddings

Maps the low-dimensional KPCA embedding of a random forest back to the
input space via iterative k-nearest neighbors.

## Usage

``` r
decode_knn(rf, emap, z, x_tilde = NULL, k = 5, parallel = TRUE)
```

## Arguments

- rf:

  Pre-trained random forest object of class `ranger`.

- emap:

  Spectral embedding learned via `eigenmap`.

- z:

  Matrix of embedded data to map back to the input space.

- x_tilde:

  Supplied training data, if none supplied then the RF is used to
  generate synthetic training data according to the eForest scheme.
  Default is NULL.

- k:

  Number of nearest neighbors to evaluate.

- parallel:

  Compute in parallel? Must register backend beforehand, e.g. via
  `doParallel`.

## Value

Decoded dataset.

## Details

`decode_knn` decodes the embedded data back to the original input space
using a k-nearest neighbors (kNN) (Cover & Hart, 1967) approach. For a
given embedding vector, decoding works by first finding the k nearest
embeddings within the training set. Then, `x_tilde` is either supplied
or generated from the RF (if generated, using the 'eForest' scheme (Feng
& Zhou, 2018)), which provides a proxy for the training samples
associated with these embeddings, to avoid needing to retain training
data. Finally, data is reconstructed by weighted averaging for numerical
features, and the most likely value for categorical features.

## References

Cover, T., & Hart, P. (1967). Nearest neighbor pattern classification.
*IEEE Transactions on Information Theory, 13*(1), 21–27.

Feng, J., & Zhou, Z. H. (2018, April). Autoencoder by forest. In
*Proceedings of the AAAI conference on artificial intelligence* (Vol. 32
, No. 1).

## Examples

``` r
# Set seed
set.seed(1)

# Split training and test
trn <- sample(1:nrow(iris), 100)
tst <- setdiff(1:nrow(iris), trn)

# Train RF, learn the encodings and project test points.
rf <- ranger::ranger(Species ~ ., data = iris[trn, ], num.trees=50)
emap <- encode(rf, iris[trn, ], k=2)
emb <- predict(emap, rf, iris[tst, ])

# Decode test samples back to the input space
out <- decode_knn(rf, emap, emb, k=5)$x_hat
#> Warning: executing %dopar% sequentially: no parallel backend registered
```
