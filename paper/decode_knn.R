#' Decode RF Embeddings
#' 
#' Maps the low-dimensional KPCA embedding of a random forest back to the input
#' space via iterative k-nearest neighbors. 
#' 
#' @param rf Pre-trained random forest object of class \code{ranger}.
#' @param z Matrix of embedded data to map back to the input space.
#' @param emap Spectral embedding learned via \code{eigenmap}.
#' @param k Number of nearest neighbors to evaluate.
#' @param parallel Compute in parallel? Must register backend beforehand, e.g. 
#'   via \code{doParallel}.
#' 
#' 
#' @details 
#' 
#' 
#' @return 
#' Decoded dataset.
#' 
#' 
#' @references 
#' 
#' 
#' @examples
#' 
#' 
#' @seealso
#' 
#' 
#' @export
#' @import Matrix
#' @import data.table
#' @importFrom RANN nn2
#' @importFrom foreach foreach %do% %dopar%
#'
#'

train_decoder <- function(
    rf, 
    emap, 
    neighbors = NULL, 
    null_value = NULL,
    parallel = TRUE) {
  
  ### eForest algorithm
  
  # Prep
  n_trees <- rf$num.trees
  trees <- seq_len(n_trees)
  colnames_x <- emap$meta$metadata$variable
  if (is.null(neighbors)) {
    neighbors <- seq_len(nrow(emap$leafIDs))
  } else {
    neighbors <- unique(neighbors)
  }
  n <- length(neighbors)
  d <- length(colnames_x)
  pred <- data.table(
    'tree' = rep(trees, each = n), 
    'leaf' = as.integer(emap$leafIDs[neighbors, ]),
    'obs' = rep(neighbors, times = n_trees)
  )
  setkey(pred, tree, leaf)
  keep_leaves <- unique(pred[, .(tree, leaf)])
  factor_cols <- emap$meta$metadata$fctr
  factor_names <- emap$meta$metadata[fctr == TRUE, variable]
  lvls <- emap$meta$levels
  
  # Exclude uninformative variables
  keep_vars <- rf$forest$independent.variable.names[
    unique((unlist(rf$forest$split.varIDs) + 1)[
      unlist(do.call(rbind, rf$forest$child.nodeIDs)[,1]) != 0
      ])]
  
  # Define bound function
  bnd_fn <- function(b) {
    num_nodes <- length(rf$forest$split.varIDs[[b]])
    lb <- matrix(-Inf, nrow = num_nodes, ncol = d)
    ub <- matrix(Inf, nrow = num_nodes, ncol = d)
    for (j in seq_len(d)) {
      lb[, j] <- emap$meta$metadata$min[j]
      ub[, j] <- emap$meta$metadata$max[j]
    }
    for (i in 1:num_nodes) {
      left_child <- rf$forest$child.nodeIDs[[b]][[1]][i] + 1L
      right_child <- rf$forest$child.nodeIDs[[b]][[2]][i] + 1L
      splitvarID <- rf$forest$split.varIDs[[b]][i] + 1L
      splitval <- rf$forest$split.value[[b]][i]
      if (left_child > 1) {
        ub[left_child, ] <- ub[right_child, ] <- ub[i, ]
        lb[left_child, ] <- lb[right_child, ] <- lb[i, ]
        if (left_child != right_child) {
          # If no pruned node, split changes bounds
          ub[left_child, splitvarID] <- lb[right_child, splitvarID] <- splitval
        }
      }
    }
    all_leaves <- which(rf$forest$child.nodeIDs[[b]][[1]] == 0L) 
    leaves_b <- intersect(all_leaves, keep_leaves[tree == b, leaf])
    colnames(lb) <- colnames(ub) <- emap$meta$metadata$variable
    out <- merge(melt(data.table('tree' = b, 'leaf' = leaves_b, lb[leaves_b, , drop = FALSE]), 
                      id.vars = c('tree', 'leaf'), value.name = 'min'), 
                 melt(data.table('tree' = b, 'leaf' = leaves_b, ub[leaves_b, , drop = FALSE]), 
                      id.vars = c('tree', 'leaf'), value.name = 'max'), 
                 by = c('tree', 'leaf', 'variable'), sort = FALSE)
    
    # Filter out uninformative variables
    out <- out[variable %in% keep_vars]
    return(out)
  }
  
  # Will resulting table be too big?
  counts_leaf <- pred[, .N, by = .(tree, leaf)]
  df_nrow_M <- counts_leaf[, sum(length(keep_vars) * N) / 1e6]
  if (df_nrow_M > 100L) {
    n_tree_chunks <- ceiling(df_nrow_M / 100L)
    tree_chunks <- split(trees, cut(trees, breaks = n_tree_chunks, labels = FALSE))
    lo <- hi <- data.table()
    for (tree_ch in seq_along(tree_chunks)) {
      if (isTRUE(parallel)) {
        bnds_b <- foreach(bb = tree_chunks[[tree_ch]], .combine = rbind, .packages = 'data.table') %dopar%
          bnd_fn(bb)
      } else {
        bnds_b <- foreach(bb = tree_chunks[[tree_ch]], .combine = rbind, .packages = 'data.table') %do%
          bnd_fn(bb)
      }
      setkey(bnds_b, tree, leaf)
      df_b <- bnds_b[pred, allow.cartesian = TRUE, nomatch = 0]
      lo_b <- df_b[, max(min), by = .(obs, variable)]
      lo <- rbind(lo, lo_b)[, max(V1), by = .(obs, variable)]
      hi_b <- df_b[, min(max), by = .(obs, variable)]
      hi <- rbind(hi, hi_b)[, min(V1), by = .(obs, variable)]
      rm(bnds_b, df_b, lo_b, hi_b)
      gc()
    }
    setnames(lo, 'V1', 'min')
    setnames(hi, 'V1', 'max')
  } else {
    # Otherwise, execute in parallel
    if (isTRUE(parallel)) {
      bnds <- foreach(bb = trees, .combine = rbind, .packages = 'data.table') %dopar%
        bnd_fn(bb)
    } else {
      bnds <- foreach(bb = trees, .combine = rbind, .packages = 'data.table') %do%
        bnd_fn(bb)
    }
    setkey(bnds, tree, leaf)
    df <- bnds[pred, allow.cartesian = TRUE, nomatch = 0]
    lo <- df[, .(min = max(min)), by = .(obs, variable)] # THIS STILL RAISES ISSUES
    hi <- df[, .(max = min(max)), by = .(obs, variable)]
    rm(df, bnds, pred)
    gc()
  }
  e_forest <- merge(lo, hi, by = c('obs', 'variable'))
  rm(lo, hi)
  gc()
  
  ### Synthesize
  
  # Prep
  synth <- e_forest
  informative <- colnames_x %in% synth$variable
  
  # Any locally uninformative features?
  tmp <- emap$meta$metadata[, .(variable, g_min = min, g_max = max, mu)]
  synth <- merge(tmp, synth, by = 'variable')
  synth[, trouble := min == g_min & max == g_max]
  if (is.null(null_value)) {
    synth[trouble == TRUE, min := mu]
    synth[trouble == TRUE, max := mu]
  } else {
    synth[trouble == TRUE, min := null_value]
    synth[trouble == TRUE, max := null_value]
  }
  synth[, c('g_min', 'g_max', 'mu', 'trouble') := NULL]
  setcolorder(synth, c('obs', 'variable', 'min', 'max'))
  
  # Any globally uninformative features?
  if (any(!informative)) {
    if (is.null(null_value)) {
      tmp <- emap$meta$metadata[, .(variable, min = mu, max = mu)]
    } else {
      tmp <- emap$meta$metadata[, .(variable, min = null_value, max = null_value)]
    }
    uninf_df <- CJ(
      'obs' = neighbors,
      'variable' = colnames_x[!informative]
    )
    uninf_df <- merge(uninf_df, tmp, by = 'variable')
    setcolorder(uninf_df, c('obs', 'variable', 'min', 'max'))
    synth <- rbind(synth, uninf_df)
  }
  
  # Synthesize
  synth[, new := runif(.N, min, max)]
  synth <- merge(synth, emap$meta$metadata[, .(variable, fctr)], 
                 by = 'variable')
  synth[fctr == TRUE, new := round(new)][, fctr := NULL]
  synth <- dcast(synth, obs ~ variable, value.var = 'new')
  synth <- synth[order(match(obs, neighbors))]
  synth[, obs := NULL]
  
  # Fix factor columns
  if (any(factor_cols)) {
    for (j in factor_names) {
      tmp <- lvls[variable == j, .(val, number)]
      setnames(tmp, 'number', j)
      synth <- merge(synth, tmp, by = j, sort = FALSE)[, c(j) := NULL]
      setnames(synth, 'val', j)
    }
  }
  setcolorder(synth, colnames_x)
  synth <- post_x(synth, emap$meta)
  return(synth)
}

# Utility function for categorical data
w_max <- function(values, weights) {
  uv <- unique(values)
  scores <- sapply(uv, function(v) sum(weights[values == v]))
  uv[which.max(scores)]
}

# Decoder
decode_knn <- function(
    rf, 
    emap,
    z, 
    x_tilde = NULL,
    k = 5,
    parallel = TRUE) {
  
  # Preliminaries, metadata
  m <- nrow(z)
  if (!is.null(x_tilde)) {
    if (nrow(x_tilde) != nrow(emap$leafIDs)) {
      stop('When providing x_tilde, nrow(x_tilde) must match nrow(emap$leafIDs).')
    }
    x_tilde_provided <- TRUE
  }
  factor_cols <- emap$meta$metadata$fctr
  
  # Compute nearest neighbors and distances in embedding space
  knn <- nn2(data = emap$Z, query = z, k = k)
  if (k > 1) {
    wts <- 1 / knn$nn.dists
    if (any(!is.finite(wts))) {
      inf_idx <- which(!is.finite(wts))
      n_clones <- length(inf_idx)
      wts[inf_idx, ] <- matrix(
        rep(c(1, rep(0, k - 1)), n_clones), nrow = n_clones, byrow = TRUE
      )
    }
    wts <- wts / rowSums(wts)
  } else {
    wts <- matrix(rep(1L, m), ncol = 1)
  }
  neighbors <- as.integer(t(knn$nn.idx)) # could be reduced by unique
  
  # Optionally estimate data from leaf bounds
  if (is.null(x_tilde)) {
    x_tilde_provided <- FALSE
    x_tilde <- train_decoder(rf, emap, neighbors, parallel = parallel)
    x_tilde <- as.data.frame(x_tilde)
    if (nrow(x_tilde) < length(neighbors)) {
      rownames(x_tilde) <- as.character(unique(neighbors))
      x_tilde <- x_tilde[as.character(neighbors), ]
      rownames(x_tilde) <- NULL
    }
  }
  
  # CONTINUOUS DATA SHOULD BE A SINGLE MATRIX OPERATION
  
  # Loop of weighted means
  loop <- function(i) {
    
    # Pick out neighbors, weights
    if (isTRUE(x_tilde_provided)) {
      x_tmp <- x_tilde[knn$nn.idx[i, ], ]
    } else {
      x_tmp <- x_tilde[(i-1)*k + seq(k), ]
    }
    w <- wts[i, ]
    
    # Take most likely label or weighted mean of continuous outcomes
    out_cat <- out_cnt <- data.table()
    if (any(factor_cols)) {
      x_tmp_cat <- x_tmp[, factor_cols, drop = FALSE]
      out_cat <- t(as.data.frame(sapply(x_tmp_cat, w_max, weights = w)))
      row.names(out_cat) <- NULL
    }
    if (any(!factor_cols)) {
      x_tmp_cnt <- as.matrix(x_tmp[, !factor_cols, drop = FALSE])
      out_cnt <- as.data.frame(t(crossprod(x_tmp_cnt, w)))
    }
    
    # Export
    out <- cbind(out_cat, out_cnt)
    return(out) 
    
  }
  
  # Execute in parallel
  if (isTRUE(parallel)) {
    out <- foreach(i = seq_len(m), .combine = rbind,
                   .export = c("w_max"), 
                   .packages = c("data.table")) %dopar% loop(i)
  } else {
    out <- foreach(i = seq_len(m), .combine = rbind) %do% loop(i)
  }
  colnames(out) <- c(emap$meta$metadata[fctr == TRUE, variable],
                     emap$meta$metadata[fctr == FALSE, variable])
  
  # Polish, export
  out <- post_x(out, emap$meta)
  if (!isTRUE(x_tilde_provided)) {
    out <- list('x_hat' = out, 'x_tilde' = x_tilde)
  }
  return(out)
  
}



