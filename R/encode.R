#' Laplacian Eigenmaps
#'
#' Computes the Laplacian eigenmap of a random forest, including a spectral
#' decomposition and associated weights.
#'
#' @param rf Pre-trained random forest object of class \code{ranger}.
#' @param x Training data for estimating embedding weights.
#' @param k Dimensionality of the spectral embedding.
#' @param stepsize Number of steps of a random walk for the diffusion process.
#'   See Details.
#' @param parallel Compute in parallel? Must register backend beforehand, e.g.
#'   via \code{doParallel}.
#'
#'
#' @details
#' \code{eigenmap} learns a low-dimensional embedding of the data implied by the
#' adjacency matrix of the \code{rf}. Random forests can be understood as an
#' adaptive nearest neighbors algorithm, where proximity between samples is
#' determined by how often they are routed to the same leaves. We compute the
#' graph Laplacian of the model adjacencies over the training data \code{x}, and
#' perform a spectral decomposition of the resulting matrix up to degree
#' \code{k}. The function returns the resulting eigenvectors, embedding weights,
#' node degrees, and leaf IDs.
#'
#' Let \eqn{A} be the weighted adjacency matrix of \code{x} implied by
#' \code{rf}. Let \eqn{D} be the degree matrix of \eqn{A}. Then the
#' (unnormalized) graph Laplacian is given by \eqn{L = D - A}. The symmetrically
#' normalized graph Laplacian is given by \eqn{L_{sym} = I - D^{-1/2} L D^{-1/2}}.
#' Projecting data onto the leading nonconstant eigenvectors of this matrix is a
#' form of kernel principal component analysis (Ham et al., 2004) with some
#' locality preserving optimality properties (Belkin & Niyogi, 2003). We can
#' embed new data into this space using the Nyström formula (Bengio et al.,
#' 2004).
#'
#' In some cases, instability in the spectral decomposition may result in
#' negative eigenvalues. Behavior in this case is determined by the
#' \code{approx} argument. If \code{TRUE}, \code{eigenmap} proceeds with the
#' nearest positive definite approximation, computed via Higham's (2002)
#' algorithm. If \code{FALSE}, the function proceeds with only the nonnegative
#' eigenvalues. In either case, a warning is issued.
#'
#' @return
#' A list with six elements: (1) \code{z}: a \code{k}-dimensional nonlinear
#' embedding of \code{x} implied by \code{rf}. (2) \code{w}: the embedding
#' weights that map scaled model adjacencies to eigenvectors. (3) \code{v}:
#' the leaking \code{k} eigenvectors; (4) \code{lambda}: the leading \code{k}
#' eigenvalues; (5) \code{d}: the inverse square root of the node degrees. (6)
#' \code{leafIDs}: a matrix with \code{nrow(x)} rows and \code{rf$num.trees}
#' columns, representing the terminal nodes of each training sample in each tree.
#'
#'
#' @references
#' Belkin, M. & Niyogi, P. (2003). Laplacian eigenmaps for dimensionality
#' reduction and data representation. \emph{Neural Computation, 15}(6):
#' 1373-1396.
#'
#' Bengio, Y., Delalleau, O., Le Roux, N., Paiement, J., Vincent, P., & Ouimet,
#' M. (2004). Learning eigenfunctions links spectral embedding and kernel PCA.
#' \emph{Neural Computation, 16}(10): 2197-2219.
#'
#' Ham, J., Lee, D., Sebastian, M., & Schölkopf, B. (2004). A kernel view of the
#' dimensionality reduction of manifolds. \emph{Proceedings of the 21st
#' International Conference on Machine Learning}.
#'
#' @examples
#' # Train ARF
#' arf <- adversarial_rf(iris)
#'
#' # Embed the data
#' emap <- eigenmap(arf, iris)
#'
#'
#' @seealso
#' \code{\link{arf}}
#'
#'
#' @export
#' @import ranger
#' @import Matrix
#' @importFrom stats predict
#' @importFrom RSpectra eigs
#'

encode <- function(
    rf,
    x,
    k = 5L,
    stepsize = 1L,
    parallel = TRUE) {

  # Prelimz
  n_trees <- rf$num.trees
  n_samples <- nrow(x)
  if (k >= n_samples) {
    warning('The dimensionality of the embedding space cannot exceed ',
            'nrow(x) - 1. Setting k to this upper bound.')
    k <- n_samples - 1L
  }

  # Weighted adjacency matrix
  leafIDs <- stats::predict(rf, x, type = 'terminalNodes')$predictions + 1L
  leafIDs_global_vec <- as.integer(
    leafIDs + rep(seq_len(n_trees) - 1L, each = n_samples) * max(leafIDs)
  )
  M <- sparseMatrix(i = rep(seq_len(n_samples), n_trees),
                    j = leafIDs_global_vec,
                    x = 1L)
  rm(leafIDs_global_vec)
  gc()
  leaf_sizes <- colSums(M)
  leaf_weights <- 1 / leaf_sizes
  M_norm <- M %*% Diagonal(x = leaf_weights)
  A <- M_norm %*% t(M) / n_trees
  rm(M, M_norm)
  gc()

  # Spectral decomposition
  e <- eigs(A, k + 1L)
  e_vec <- e$vectors[, 2:(k + 1)]
  e_val <- e$values[2:(k + 1)]
  if (k > 1) {
    Z <- as.matrix(sqrt(n_samples) * e_vec %*% Diagonal(k, e_val^stepsize))
  } else {
    Z <- sqrt(n_samples) * as.matrix(e_vec) * e_val^stepsize
  }

  # Metadata
  input_class <- class(x)
  x <- as.data.frame(x)
  colnames_x <- rf$forest$independent.variable.names
  x <- x[, colnames_x, drop = FALSE]
  x <- prep_x(x)$x
  n <- nrow(x)
  n_col <- ncol(x)
  factor_cols <- sapply(x, is.factor)
  if (any(factor_cols)) {
    factor_names <- colnames_x[factor_cols]
    lvls <- rbindlist(lapply(factor_names, function(j) {
      data.table('variable' = j, 'val' = rf$forest$covariate.levels[[j]])[, number := .I]
    }))
  } else {
    lvls <- NULL
  }
  deci <- rep(NA_integer_, n_col)
  if (any(!factor_cols)) {
    deci[!factor_cols] <- sapply(which(!factor_cols), function(j) {
      if (any(grepl('\\.', x[[j]]))) {
        tmp <- x[grepl('\\.', x[[j]]), j]
        out <- max(nchar(sub('.*[.]', '', tmp)))
      } else {
        out <- 0L
      }
      return(out)
    })
  }
  params <- matrix(nrow = n_col, ncol = 3)
  for (j in seq_len(n_col)) {
    if (j %in% which(!factor_cols)) {
      params[j, 1] <- mean(x[[j]])
      params[j, 2] <- min(x[[j]])
      params[j, 3] <- max(x[[j]])
    } else {
      mode_lvl <- levels(x[[j]])[which.max(tabulate(x[[j]]))]
      params[j, 1] <- lvls[variable == colnames_x[j] & val == mode_lvl, number]
      params[j, 2] <- 0.5
      params[j, 3] <- length(unique(x[[j]])) + 0.5 # I think this is wrong
    }
  }

  metadata <- data.table(
    'variable' = colnames_x,
    'class' = sapply(x, class),
    'fctr' = factor_cols,
    'decimals' = deci,
    'mu' = params[, 1],
    'min' = params[, 2],
    'max' = params[, 3]
  )
  meta <- list('metadata' = metadata, 'levels' = lvls, 'input_class' = input_class)

  # Leaf sizes
  sizes <- data.table(
    'tree' = rep(seq_len(n_trees), each = n_samples),
    'leaf' = as.integer(leafIDs)
  )
  sizes[, leaf_size := .N, by = .(tree, leaf)]
  sizes <- unique(sizes)

  # Export
  out <- list('Z' = Z, 'A' = A, 'V' = e_vec, 'lambda' = e_val, 'stepsize' = stepsize,
              'leafIDs' = leafIDs, 'sizes' = sizes, 'meta' = meta)
  return(out)

}



#' Predict Spectral Embeddings
#'
#' Projects test data into the forest embedding space using a pre-trained
#' Laplacian eigenmap.
#'
#' @param emap Spectral embedding for the \code{rf} learned via \code{eigenmap}.
#' @param rf Pre-trained random forest object of class \code{ranger}.
#' @param x Data to be embedded.
#' @param parallel Compute in parallel? Must register backend beforehand, e.g.
#'   via \code{doParallel}.
#'
#'
#' @details
#' This function uses the weights learned via \code{eigenmap} to project new
#' data into the low-dimensional embedding space using the Nyström formula.
#' For details, see Bengio et al. (2004).
#'
#'
#' @return
#' A matrix of embeddings, with \code{nrow(x)} rows and \code{k} columns, the
#' latter argument used to learn the \code{eigenmap}.
#'
#'
#' @references
#' Bengio, Y., Delalleau, O., Le Roux, N., Paiement, J., Vincent, P., & Ouimet,
#' M. (2004). Learning eigenfunctions links spectral embedding and kernel PCA.
#' \emph{Neural Computation, 16}(10): 2197-2219.
#'
#'
#' @examples
#' # Set seed
#' set.seed(1)
#'
#' # Split training and test
#' trn <- sample(1:nrow(iris), 100)
#' tst <- setdiff(1:nrow(iris), trn)
#'
#' # Train ARF
#' arf <- adversarial_rf(iris[trn, ])
#'
#' # Learn the Laplacian eigenmap
#' emap <- eigenmap(arf, iris[trn, ])
#'
#' # Embed test points
#' emb <- predict.eigenmap(emap, arf, iris[tst, ])
#'
#'
#' @seealso
#' \code{\link{arf}}
#'
#'
#' @export
#' @method predict encode
#' @import ranger
#' @importFrom stats predict
#' @importFrom foreach foreach %do% %dopar%
#'

predict.encode <- function(
    emap,
    rf,
    x,
    parallel = TRUE) {

  # Prelimz
  tmp <- as.matrix(emap$V)
  n_trees <- rf$num.trees
  trn_n <- nrow(tmp)
  d_z <- ncol(tmp)
  tst_n <- nrow(x)

  # Weighted adjacency matrix
  leafIDs_train <- emap$leafIDs
  leafIDs_test <- stats::predict(rf, x, type = 'terminalNodes')$predictions + 1L
  max_leaf <- max(leafIDs_train, leafIDs_test, na.rm = TRUE)

  leafIDs_global_train <- leafIDs_train + rep(seq_len(n_trees) - 1, each = trn_n) * max_leaf
  leafIDs_global_test <- leafIDs_test + rep(seq_len(n_trees) - 1, each = tst_n) * max_leaf

  leafIDs_global <- union(as.integer(leafIDs_global_train),
                          as.integer(leafIDs_global_test))
  leaf_id_map <- match(c(as.integer(leafIDs_global_train),
                         as.integer(leafIDs_global_test)), leafIDs_global)
  split_point <- length(as.integer(leafIDs_global_train))
  num_cols <- length(leafIDs_global)

  M_train <- sparseMatrix(
    i = rep(seq_len(trn_n), n_trees),
    j = leaf_id_map[1:split_point],
    x = 1L,
    dims = c(trn_n, num_cols)
  )

  M_test <- sparseMatrix(
    i = rep(seq_len(tst_n), n_trees),
    j = leaf_id_map[(split_point + 1):length(leaf_id_map)],
    x = 1L,
    dims = c(tst_n, num_cols)
  )

  leaf_sizes <- colSums(M_train)
  leaf_weights <- 1 / leaf_sizes
  leaf_weights[!is.finite(leaf_weights)] <- 0

  M_test_norm <- M_test %*% Diagonal(x = leaf_weights)

  A0 <- t(M_train %*% t(M_test_norm) / n_trees)

  # Embed using the Nyström formula
  Z0 <- as.matrix(A0 %*% emap$Z %*% Diagonal(x = 1 / emap$lambda))

  # Export
  return(Z0)

}


