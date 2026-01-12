# Mixed-type Reconstruction Error

Computes the reconstruction error of a decoded dataset compared to the
original.

## Usage

``` r
reconstruction_error(Xhat, X)
```

## Arguments

- Xhat:

  Reconstructed dataset

- X:

  Ground truth dataset

## Value

A list containing column-wise reconstruction error, and the average
reconstruction error for categorical and numeric variables. Values lie
between 0-1, where 0 represents perfect reconstruction, and 1 represents
no reconstruction.

## Details

In standard AEs, reconstruction error is generally estimated via \\L_2\\
loss. This is not sensible with a mix of continuous and categorical
data, so we devise a measure that evaluates distortion on continuous
variables as \\1 - R^2\\, and categorical variables as prediction error.

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

# Compute the reconstruction error
error <- reconstruction_error(out, iris[tst, ])
```
