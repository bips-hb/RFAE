# Predict Spectral Embeddings

Projects test data into the forest embedding space using a pre-trained
encoding map.

## Usage

``` r
# S3 method for class 'encode'
predict(object, rf, x, parallel = TRUE, ...)
```

## Arguments

- object:

  Spectral embedding for the `rf` learned via `eigenmap`.

- rf:

  Pre-trained random forest object of class `ranger`.

- x:

  Data to be embedded.

- parallel:

  Compute in parallel? Must register backend beforehand, e.g. via
  `doParallel`.

- ...:

  Additional arguments passed to methods.

## Value

A matrix of embeddings, with `nrow(x)` rows and `k` columns, the latter
argument used to learn the `eigenmap`.

## Details

This function uses the weights learned via `eigenmap` to project new
data into the low-dimensional embedding space using the Nyström formula.
For details, see Bengio et al. (2004).

## References

Bengio, Y., Delalleau, O., Le Roux, N., Paiement, J., Vincent, P., &
Ouimet, M. (2004). Learning eigenfunctions links spectral embedding and
kernel PCA. *Neural Computation, 16*(10): 2197-2219.

## See also

[`adversarial_rf`](https://bips-hb.github.io/arf/reference/adversarial_rf.html)

## Examples

``` r
# Set seed
set.seed(1)

# Split training and test
trn <- sample(1:nrow(iris), 100)
tst <- setdiff(1:nrow(iris), trn)

# Train RF. You can also use RF variants, such as the Adversarial Random
# Forests (ARF).
rf <- ranger::ranger(Species ~ ., data = iris[trn, ], num.trees=50)

# Learn the encodings, which are found using diffusion maps.
emap <- encode(rf, iris[trn, ], k=2)

# Embed test points
emb <- predict(emap, rf, iris[tst, ])

```
