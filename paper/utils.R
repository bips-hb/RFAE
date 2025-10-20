#' Preprocess input data
#' 
#' This function prepares input data.
#' 
#' @param x Input data.frame.
#' @keywords internal

prep_x <- function(x, to_numeric=NULL, to_factor=NULL, default = 5) {
  # Reclass all non-numeric features as factors
  x <- as.data.frame(x)
  idx_char <- sapply(x, is.character)
  if (any(idx_char)) {
    x[, idx_char] <- lapply(x[, idx_char, drop = FALSE], as.factor)
  }
  idx_logical <- sapply(x, is.logical)
  if (any(idx_logical)) {
    x[, idx_logical] <- lapply(x[, idx_logical, drop = FALSE], as.factor)
  }
  idx_integer <- sapply(x, is.integer)
  if (any(idx_integer)) {
    # Recoding integers with > 5 levels as numeric
    if (is.null(to_numeric)) {
      to_numeric <- sapply(seq_len(ncol(x)), function(j) {
        idx_integer[j] & length(unique(x[[j]])) > default
      }) 
    } else {
      to_numeric <- to_numeric[names(x)]
    }
    if (any(to_numeric)) {
      warning('Recoding integers with more than 5 unique values as numeric. ', 
              'To override this behavior, explicitly code these variables as factors.')
      x[, to_numeric] <- lapply(x[, to_numeric, drop = FALSE], as.numeric)
    } 
    if (is.null(to_factor)) {
      to_factor <- sapply(seq_len(ncol(x)), function(j) {
        idx_integer[j] & length(unique(x[[j]])) < 6
      })
    } else {
      to_factor <- to_factor[names(x)]
    }
    if (any(to_factor)) {
      warning('Recoding integers with fewer than 6 unique values as ordered factors. ', 
              'To override this behavior, explicitly code these variables as numeric.')
      x[, to_factor] <- lapply(which(to_factor), function(j) {
        lvls <- sort(unique(x[[j]]))
        factor(x[[j]], levels = lvls, ordered = TRUE)
      })
    }
  }
  return(list('x' = x, 'to_numeric'= to_numeric, 'to_factor' = to_factor))
}


#' Post-process data
#' 
#' This function prepares output data.
#' 
#' @param x Input data.frame.
#' @param meta Metadata.
#' @param round Round continuous variables to their respective maximum precision 
#'   in the real data set?
#' @param input_class Input class of \code{x}.
#' @param lvls Metadata on factor variables.
#' 
#' @import data.table
#' @keywords internal

post_x <- function(x, meta, round = TRUE) {
  
  # To avoid data.table check issues
  variable <- val <- NULL
  
  # Assign some things
  input_class <- meta$input_class
  lvls <- meta$levels
  
  # Order, classify features
  meta_tmp <- meta$metadata[variable %in% colnames(x)]
  setcolorder(x, match(meta_tmp$variable, colnames(x)))
  setDF(x)
  idx_numeric <- meta_tmp[, which(class == 'numeric')]
  idx_factor <- meta_tmp[, which(class == 'factor')]
  idx_ordered <- meta_tmp[, which(grepl('ordered', class))]
  idx_logical <- meta_tmp[, which(class == 'logical')]
  idx_integer <- meta_tmp[, which(class == 'integer')]
  
  # Recode
  if (sum(idx_numeric) > 0L & round) {
    x[, idx_numeric] <- lapply(idx_numeric, function(j) {
      round(as.numeric(x[[j]]), meta_tmp$decimals[j])
    })
  }
  if (sum(idx_factor) > 0L) {
    x[, idx_factor] <- lapply(idx_factor, function(j) {
      factor(x[[j]], levels = lvls[variable == colnames(x)[j], val])
    })
  }
  if (sum(idx_ordered) > 0L) {
    x[, idx_ordered] <- lapply(idx_ordered, function(j) {
      factor(x[[j]], levels = lvls[variable == colnames(x)[j], val], ordered = TRUE)
    })
  }
  if (sum(idx_logical) > 0L) {
    x[, idx_logical] <- lapply(x[, idx_logical, drop = FALSE], as.logical)
  }
  if (sum(idx_integer) > 0L) {
    x[, idx_integer] <- lapply(idx_integer, function(j) {
      if (is.numeric(x[[j]])) {
        if (round) {
          as.integer(round(x[[j]]))
        } else {
          x[[j]]
        }
      } else {
        as.integer(as.character(x[[j]]))
      }
    }) 
  }
  
  # Export
  if ('data.table' %in% input_class) {
    setDT(x)[]
  } else if ('tbl_df' %in% input_class & requireNamespace("tibble", quietly = TRUE)) {
    x <- tibble::as_tibble(x)
  } else if ('matrix' %in% input_class) {
    x <- as.matrix(x)
  }
  return(x)
}


#' Apply adaptive sparsity thresholds
#' 
#' This function caps the number of expected neighbors in a data-driven manner.
#' 
#' @param emap Spectral embedding for the \code{rf} learned via \code{eigenmap}.
#' @param A0 Adjacency matrix for test samples.
#' @param parallel Compute in parallel? 
#' 
#' @import data.table
#' @import Matrix
#' @import foreach 
#' @import mgcv
#' @keywords internal

sparsify <- function(emap, A0, parallel = TRUE) {
  
  # Hyperparameters
  n <- nrow(emap$z)
  m <- nrow(A0)
  
  # Estimate training adjacency matrix
  L_hat <- emap$z %*% diag(sqrt(emap$lambda)) %*% t(emap$v) 
  L_hat[L_hat < 0] <- 0
  L_hat <- as(L_hat, 'sparseMatrix')
  d_old <- 1 / emap$d # These are now square root of node degrees
  d_mat <- matrix(rep(d_old, n), nrow = n, byrow = TRUE)
  A_hat <- t(L_hat * d_mat)
  d_hat <- colSums(A_hat)
  d_mat <- matrix(rep(d_hat, n), nrow = n, byrow = TRUE)
  A_hat <- t(A_hat * d_mat)
  diag(A_hat) <- 0
  
  # Count true and estimated neighbors
  neighb_fn <- function(i) {
    data.table(k = sum(emap$A[i, ] > 0), k_hat = sum(A_hat[i, ] > 0))
  }
  if (isTRUE(parallel)) {
    df <- foreach(ii = seq_len(n), .combine = rbind) %dopar% neighb_fn(ii)
  } else {
    df <- rbindlist(lapply(seq_len(n), neighb_fn))
  }
  
  # Fit cubic spline
  f <- gam(k ~ s(k_hat, bs = 'cs'), data = df)
  
  # Predict neighbors for A0
  if (isTRUE(parallel)) {
    k_hat <- foreach(i = seq_len(m), .combine = c) %dopar% sum(A0[i, ] > 0)
  } else {
    k_hat <- sapply(seq_len(m), function(i) sum(A0[i, ] > 0))
  }
  sp_hat <- 1 - predict(f, data.table(k_hat)) / n
  sub0 <- sp_hat[sp_hat < 0]
  if (any(sub0)) {
    sp_hat[sub0] <- 0
  }
  sup1 <- sp_hat[sp_hat > 1]
  if (any(sup1)) {
    sp_hat[sup1] <- 1
  }
  
  # Zero out entries below the adaptive thresholds
  zero_out <- function(i) {
    tmp <- A0[i, ]
    thresh <- quantile(tmp, sp_hat[i], names = FALSE)
    flag <- tmp < thresh
    tmp[flag] <- 0
    return(tmp)
  }
  if (isTRUE(parallel)) {
    out <- foreach(ii = seq_len(m), .combine = rbind) %dopar% zero_out(ii)
  } else {
    out <- foreach(ii = seq_len(m), .combine = rbind) %dopar% zero_out(ii)
  }
  
  # Export
  dimnames(out) <- NULL
  out <- as(out, 'sparseMatrix')
  return(out)
  
}

