#' Autoencoder forest
#' 
#' Implements an autoencoder forest, a function that maps inputs to a 
#' hyperrectangle defined by the intersection of leaves in a random forest.
#' 
#' @param rf Random forest model object of class \code{ranger}.
#' @param emap 
#' @param parallel Compute in parallel? Must register backend beforehand.
#'
#' @details
#' This function implements the autoencoder forest algorithm proposed by Feng & 
#' Zhou (2017). 
#' 
#' @return
#' An encoded dataset of maximum compatible rules for each sample.
#' 
#' @references 
#' Feng, J. and Zhou, Z. (2017). AutoEncoder by Forest. \emph{arXiv} preprint, 
#' 1709.09018. 
#' 
#' @examples
#' # Encode supervised random forest
#' library(ranger)
#' rf <- ranger(Species ~ ., data = iris)
#' z <- eForest(rf, iris)
#' 
#' # Encode unsupervised random forest
#' n <- nrow(iris)
#' x_synth <- as.data.frame(lapply(iris, sample, n, replace = TRUE))
#' dat <- rbind(data.frame(y = 1, iris), data.frame(y = 0, x_synth))
#' rf <- ranger(y ~ ., data = dat)
#' z <- eForest(rf, iris)
#' 
#' 
#' @export
#' @import data.table
#' @import ranger
#' @import foreach

eForest <- function(
    rf,
    emap,
    parallel = TRUE) {
  
  # Prelimz
  n_trees <- rf$num.trees
  trees <- seq_len(n_trees)
  n <- nrow(emap$leafIDs)
  d <- emap$meta$metadata[, .N]
  pred <- data.table(
    'tree' = rep(trees, each = n), 
    'leaf' = as.integer(emap$leafIDs),
    'obs' = rep(seq_len(n), times = n_trees)
  )
  keep_leaves <- unique(pred[, .(tree, leaf)])
  
  # Exclude uninformative variables
  split_fn <- function(tree) {
    tree_df <- as.data.table(treeInfo(rf, tree))
    out <- tree_df[terminal == FALSE, unique(splitvarName)]
    return(out)
  }
  if (isTRUE(parallel)) {
    keep_vars <- foreach(b = trees, .combine = c, 
                         .packages = c('data.table', 'ranger')) %dopar% 
      split_fn(b)
  } else {
    keep_vars <- foreach(b = trees, .combine = c, 
                         .packages = c('data.table', 'ranger')) %do% 
      split_fn(b)
  }
  keep_vars <- unique(keep_vars)
  
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
  n_rows_millions <- keep_leaves[, .N] / 1e6 * length(keep_vars)
  if (n_rows_millions > 100L) {
    # If so, update the table serially by tree
    lo <- hi <- data.table()
    for (b in trees) {
      bnds_b <- bnd_fn(b)
      df_b <- merge(bnds_b, pred[tree == b, .(leaf, obs)], by = 'leaf', 
                    allow.cartesian = TRUE, sort = FALSE)
      lo_b <- df_b[, max(min), by = .(obs, variable)]
      lo <- rbind(lo, lo_b)[, max(V1), by = .(obs, variable)]
      hi_b <- df_b[, min(max), by = .(obs, variable)]
      hi <- rbind(hi, hi_b)[, min(V1), by = .(obs, variable)]
    }
  } else {
    # Otherwise, execute in parallel
    if (isTRUE(parallel)) {
      bnds <- foreach(bb = trees, .combine = rbind, .packages = 'data.table') %dopar%
        bnd_fn(bb)
    } else {
      bnds <- foreach(bb = trees, .combine = rbind, .packages = 'data.table') %do%
        bnd_fn(bb)
    }
    df <- merge(bnds, pred, by = c('tree', 'leaf'), allow.cartesian = TRUE, 
                sort = FALSE)
    lo <- df[, max(min), by = .(obs, variable)]
    hi <- df[, min(max), by = .(obs, variable)]
  }
  setnames(lo, 'V1', 'min')
  setnames(hi, 'V1', 'max')
  out <- merge(lo, hi, by = c('obs', 'variable'))
  return(out)
  
}


