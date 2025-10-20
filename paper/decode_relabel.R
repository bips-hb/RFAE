library(ranger)
library(data.table)
library(foreach)
library(igraph)

train_decode_relabel <- function(
    rf, 
    emap,
    x, 
    z = NULL,
    n_random_splits = 1000L,
    parallel = TRUE) {
  
  # Prep
  d <- rf$num.independent.variables
  trees <- 1:rf$num.trees
  if (is.null(z)) {
    z <- emap$V
  }
  z <- as.data.frame(z)
  
  # Exclude uninformative variables
  keep_vars <- rf$forest$independent.variable.names[
    unique((unlist(rf$forest$split.varIDs) + 1)[
      unlist(do.call(rbind, rf$forest$child.nodeIDs)[,1]) != 0
    ])]
  
  # Bound leaves
  bnd_fn <- function(tree) {
    num_nodes <- length(rf$forest$split.varIDs[[tree]])
    lb <- matrix(-Inf, nrow = num_nodes, ncol = d)
    ub <- matrix(Inf, nrow = num_nodes, ncol = d)
    for (j in seq_len(d)) {
      lb[, j] <- emap$meta$metadata$min[j]
      ub[, j] <- emap$meta$metadata$max[j]
    }
    for (i in 1:num_nodes) {
      left_child <- rf$forest$child.nodeIDs[[tree]][[1]][i] + 1L
      right_child <- rf$forest$child.nodeIDs[[tree]][[2]][i] + 1L
      splitvarID <- rf$forest$split.varIDs[[tree]][i] + 1L
      splitval <- rf$forest$split.value[[tree]][i]
      if (left_child > 1) {
        ub[left_child, ] <- ub[right_child, ] <- ub[i, ]
        lb[left_child, ] <- lb[right_child, ] <- lb[i, ]
        if (left_child != right_child) {
          # If no pruned node, split changes bounds
          ub[left_child, splitvarID] <- lb[right_child, splitvarID] <- splitval
        }
      }
    }
    leaves <- which(rf$forest$child.nodeIDs[[tree]][[1]] == 0L) 
    colnames(lb) <- colnames(ub) <- emap$meta$metadata$variable
    out <- merge(melt(data.table(tree = tree, leaf = leaves, lb[leaves, , drop = FALSE]), 
                      id.vars = c('tree', 'leaf'), value.name = 'min'), 
                 melt(data.table(tree = tree, leaf = leaves, ub[leaves, , drop = FALSE]), 
                      id.vars = c('tree', 'leaf'), value.name = 'max'), 
                 by = c('tree', 'leaf', 'variable'), sort = FALSE)
    out <- out[variable %in% keep_vars]
    return(out)
  }
  
  # Execute in parallel
  if (isTRUE(parallel)) {
    bnds <- foreach(b = trees, .combine = rbind, .packages = 'data.table') %dopar%
      bnd_fn(b)
  } else {
    bnds <- foreach(b = trees, .combine = rbind, .packages = 'data.table') %do%
      bnd_fn(b)
  }
  setkey(bnds, tree, leaf)
  
  # Approximate original RF with Z splits
  rf_zspace <- approx_ranger(rf, x, z, n_random_splits, parallel)
  # rf_zspace <- approx_ranger_local(rf, x, z)
  
  # Export
  out <- list('rf_zspace' = rf_zspace, 'bnds' = bnds, 'meta' = emap$meta)
  return(out)
  
}


# Now we can decode
decode_relabel <- function(decoder, z, x = NULL) {
  
  # Extract
  rf_zspace <- decoder$rf_zspace
  bnds <- decoder$bnds
  meta <- decoder$meta
  
  # Prep
  n_trees <- rf_zspace$num.trees
  z <- as.data.frame(z)
  m <- nrow(z)
  if (!is.null(x)) {
    x <- as.data.frame(x)
  }
  colnames_x <- meta$metadata$variable
  factor_cols <- meta$metadata$fctr
  factor_names <- meta$metadata[fctr == TRUE, variable]
  lvls <- meta$levels
  input_class <- meta$input_class
  
  # Find leaves
  leaf_preds <- predict(rf_zspace, z, type = 'terminalNodes')$predictions
  leaves <- data.table(
    'tree' = rep(seq_len(n_trees), each = m),
    'leaf' = as.integer(leaf_preds + 1L),
    'idx' = rep(seq_len(m), times = n_trees)
  )
  setkey(leaves, tree, leaf)
  
  # Compute bounding boxes
  df <- bnds[leaves, allow.cartesian = TRUE, nomatch = 0L]
  lo <- df[, max(min), by = .(idx, variable)]
  hi <- df[, min(max), by = .(idx, variable)]
  setnames(lo, 'V1', 'min')
  setnames(hi, 'V1', 'max')
  df2 <- merge(lo, hi, by = c('idx', 'variable'))
  
  # Consistency check: are all intersections nonempty?
  if (df2[, !all(max >= min)]) {
    # If inconsistent, return maximal clique
    intersections <- matrix(0L, nrow = n_trees, ncol = n_trees)
    for (b1 in 2:n_trees) {
      for (b2 in 1:(b1 - 1)) {
        if (all(df[tree == b1, max] > df[tree == b2, min] & 
                df[tree == b2, max] > df[tree == b1, min])) {
          intersections[b1, b2] <- 1L
        }
      }
    }
    intersection_graph <- graph_from_adjacency_matrix(intersections, mode = 'lower')
    clique <- largest_cliques(intersection_graph)
    if (length(clique) == n_trees) {
      stop('The edge set is empty. Consider increasing sample size.')
    } else if (length(clique) == 1) {
      clique <- as.integer(clique[[1]])
    } else {
      # Randomly select
      which_c <- sample(length(clique), 1)
      clique <- as.integer(clique[[which_c]])
    }
    leaves <- leaves[tree %in% clique]
    df <- merge(bnds, leaves, by = c('tree', 'leaf'), allow.cartesian = TRUE, 
                sort = FALSE)
    lo <- df[, max(min), by = .(idx, variable)]
    hi <- df[, min(max), by = .(idx, variable)]
    setnames(lo, 'V1', 'min')
    setnames(hi, 'V1', 'max')
    df <- merge(lo, hi, by = c('idx', 'variable'))
  } else {
    # Otherwise, proceed
    df <- df2
  }
  
  # Any locally uninformative features?

  tmp <- meta$metadata[, .(variable, g_min = min, g_max = max, mu)]
  df <- merge(tmp, df, by = 'variable')
  df[, trouble := min == g_min & max == g_max]
  if (any(df$trouble)) {
    df[trouble == TRUE, min := mu]
    df[trouble == TRUE, max := mu]
  }
  df[, c('g_min', 'g_max', 'mu', 'trouble') := NULL]
  setcolorder(df, c('idx', 'variable', 'min', 'max'))
  
  # Any globally uninformative features?
  informative <- colnames_x %in% df$variable
  if (any(!informative)) {
    uninf_df <- CJ(
      'idx' = seq_len(m),
      'variable' = colnames_x[!informative]
    )
    uninf_df[, min := null_value][, max := null_value]
    df <- rbind(df, uninf_df)
  }
  
  # Synthesize
  df[, new := runif(.N, min, max)]
  df[, c('min', 'max') := NULL]
  
  # Round factors
  if (any(factor_cols)) {
    df[variable %in% factor_names, new := round(new)]
  }
  
  # Cast
  res <- dcast(df, idx ~ variable, value.var = 'new')
  res <- res[order(match(idx, seq_len(m)))]
  res[, idx := NULL]
  
  # Fix factor columns
  if (any(factor_cols)) {
    for (j in factor_names) {
      tmp2 <- lvls[variable == j, .(val, number)]
      setnames(tmp2, 'number', j)
      res <- merge(res, tmp2, by = j, sort = FALSE)[, c(j) := NULL]
      setnames(res, 'val', j)
    }
  }
  
  # Reshuffle, polish
  setcolorder(res, colnames_x)
  res <- post_x(res, meta)
  
  # Export
  return(res)
  
}


# # Example
# library(Matrix)
# library(RSpectra)
# source('encode.R')
# source('utils.R')
# source('approx_ranger.R')
# 
# # Train iris model
# set.seed(123)
# x <- iris[, -5]
# y <- iris[, 5]
# rf <- ranger(x = x, y = y, num.trees = 50)
# emap <- encode(rf, x, k = 2)
# 
# # Train the decoder, decode
# decoder <- train_decode_relabel(rf, emap, x)
# x_new <- decode_relabel(decoder, emap$Z, x)






