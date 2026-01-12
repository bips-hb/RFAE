# Encoding with Diffusion Maps

Computes the diffusion map of a random forest kernel, including a
spectral decomposition and associated weights.

## Usage

``` r
encode(rf, x, k = 5L, stepsize = 1L, parallel = TRUE)
```

## Arguments

- rf:

  Pre-trained random forest object of class `ranger`.

- x:

  Training data for estimating embedding weights.

- k:

  Dimensionality of the spectral embedding.

- stepsize:

  Number of steps of a random walk for the diffusion process. See
  Details.

- parallel:

  Compute in parallel? Must register backend beforehand, e.g. via
  `doParallel`.

## Value

A list with eight elements: (1) `Z`: a `k`-dimensional nonlinear
embedding of `x` implied by `rf`. (2) `A`: the normalized adjacency
matrix (3) `v`: the leading `k` eigenvectors; (4) `lambda`: the leading
`k` eigenvalues; (5) `stepsize`: the number of steps in the random walk.
(6) `leafIDs`: a matrix with `nrow(x)` rows and `rf$num.trees` columns,
representing the terminal nodes of each training sample in each tree;
(7) the number of samples in each leaf; (8) metadata about the `rf`.

## Details

`encode` learns a low-dimensional embedding of the data implied by the
adjacency matrix of the `rf`. Random forests can be understood as an
adaptive nearest neighbors algorithm, where proximity between samples is
determined by how often they are routed to the same leaves. We compute
the spectral decomposition of the model adjacencies over the training
data `X`, and take the leading `k` eigenvectors and eigenvalues. The
function returns the resulting diffusion map, eigenvectors, eigenvalues,
and leaf sizes.

Let \\K\\ be the weighted adjacency matrix of code `x` implied by `rf`.
This defines a weighted, undirected graph over the training data, which
we can also interpret as the transitions of a Markov process 'between'
data points. Spectral analysis produces the decomposition \\K = V\lambda
V^{-1}\\, where we can take leading nonconstant eigenvectors. The
diffusion map \\Z = \sqrt{n} V \lambda^{t}\\ (Coifman & Lafon, 2006)
represents the long-run connectivity structure of the graph after t time
steps of a Markov process, with some nice optimization properties (von
Luxburg, 2007). We can embed new data into this space using the Nyström
formula (Bengio et al., 2004).

## References

Bengio, Y., Delalleau, O., Le Roux, N., Paiement, J., Vincent, P., &
Ouimet, M. (2004). Learning eigenfunctions links spectral embedding and
kernel PCA. *Neural Computation, 16*(10): 2197-2219.

Coifman, R. R., & Lafon, S. (2006). Diffusion maps. *Applied and
Computational Harmonic Analysis, 21*(1), 5–30.

von Luxburg, U. (2007). A tutorial on spectral clustering. *Statistics
and Computing, 17*(4), 395–416.

## See also

[`adversarial_rf`](https://bips-hb.github.io/arf/reference/adversarial_rf.html)

## Examples

``` r
# Train ARF
arf <- arf::adversarial_rf(iris)
#> Iteration: 0, Accuracy: 78.77%
#> Iteration: 1, Accuracy: 37.58%

# Embed the data
emap <- encode(arf, iris)

```
