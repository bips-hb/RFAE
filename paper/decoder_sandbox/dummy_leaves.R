#' Decode RF Embeddings
#' 
#' Maps the low-dimensional KPCA embedding of a random forest back to the input
#' space via constrained optimization and greedy search. 
#' 
#' @param z Matrix of embedded data to map back to the input space.
#' @param rf Pre-trained random forest object of class \code{ranger}.
#' @param emap Spectral embedding for the \code{rf} learned via \code{eigenmap}.
#' @param A True adjacency matrix.
#' @param x Training data. 
#' @param max_mem Maximum RAM.
#' @param parallel Compute in parallel? Must register backend beforehand, e.g. 
#'   via \code{doParallel}.
#' 
#' 
#' @details 
#' 
#' 
#' @return 
#' A list with 
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
#' @import ranger 
#' @import mgcv
#' @import Matrix
#' @import glmnet
#' @import ExclusiveLasso
#' @import data.table
#' @importFrom RANN nn2
#' @importFrom igraph graph_from_adjacency_matrix largest_cliques
#' @importFrom foreach foreach %do% %dopar%
#'

dummy_leaves <- function(
    z, 
    rf,
    emap,
    A,
    x,
    max_mem = 32,
    parallel = TRUE) {
  
  # Preliminaries, metadata
  num_trees <- rf$num.trees
  trees <- seq_len(num_trees)
  if (!is.matrix(z)) {
    stop('z must be a matrix with one row per sample to decode.')
  }
  m <- nrow(z)
  d_z <- ncol(z)
  colnames_z <- paste0('z', seq_len(d_z))
  colnames(z) <- colnames_z
  n <- nrow(emap$leafIDs)
  d <- emap$meta$metadata[, .N]
  colnames_x <- emap$meta$metadata$variable
  factor_cols <- emap$meta$metadata$fctr
  factor_names <- emap$meta$metadata[fctr == TRUE, variable]
  lvls <- emap$meta$levels
  input_class <- emap$meta$input_class
  if (ncol(z) != ncol(emap$z)) {
    stop('ncol(z) must match ncol(emap$z).')
  }
  
  # Training data supplied?
  if (!is.null(x)) {
    x <- as.data.frame(x)
  }
  
  # Exclude uninformative variables
  split_vars <- unique(unlist(rf$forest$split.varIDs))
  keep_vars <- colnames_x[split_vars]
  d_tmp <- length(keep_vars)
  if (d > d_tmp) {
    uninf_idx <- setdiff(seq_len(d), split_vars)
    uninformative <- emap$meta$metadata[uninf_idx, variable]
  } else {
    uninformative <- character()
  }
  
  # Bound leaves, potentially write to disk
  bnd_fn <- function(tree, ram) {
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
    if (isTRUE(ram)) {
      return(out)
    } else {
      saveRDS(out, paste0('bnds', tree, '.rds'))
    }
  }
  
  # Write to disk if table expected to exceed max_mem
  nl <- sapply(trees, function(b) sum(treeInfo(rf, tree = b)['terminal']))
  Nl <- sum(nl)
  bnds_nrow <- Nl * d_tmp
  estimated_memory <- bnds_nrow * 28 / 1024^3 # Each row = 28 bits, converting to GB
  ram_flag <- estimated_memory < max_mem
  if (isTRUE(ram_flag)) {
    if (isTRUE(parallel)) {
      bnds <- foreach(tree = trees, .combine = rbind, .packages = 'data.table') %dopar%
        bnd_fn(tree, ram_flag)
    } else {
      bnds <- foreach(tree = trees, .combine = rbind, .packages = 'data.table') %do%
        bnd_fn(tree, ram_flag)
    }
  } else {
    warning('Writing bound tables to disk. This will use an estimated ',
            round(estimated_memory, 2), 'GB of space.')
    if (isTRUE(parallel)) {
      foreach(tree = trees, .packages = 'data.table') %dopar% bnd_fn(tree, ram_flag)
    } else {
      foreach(tree = trees, .packages = 'data.table') %do% bnd_fn(tree, ram_flag)
    }
  }
  cat('Leaves bounded...\n')
  
  ### Big loop ###
  loop <- function(i) {
    
    # Who are my neighbors?
    neighbors <- which(A[i, ] > 0)
    k <- length(neighbors)
    y <- A[i, neighbors]
    up <- 1L
    
    # Where do they live?
    df <- emap$leafIDs[neighbors, trees] + 1L
    df <- unique(data.table(
      'tree' = rep(trees, each = k),
      'leaf' = as.integer(df)
    ))
    df[, n := .N, by = tree]
    
    # Function for updating candidate leaves
    candidate_list <- function(dat, lo, hi, ooc) {
      if (isTRUE(ram_flag)) {
        if ('coef' %in% colnames(dat)) {
          out <- merge(dat[tree %in% ooc, .(tree, leaf, variable, min, max, coef)],
                       lo, by = 'variable')
        } else {
          out <- merge(dat[tree %in% ooc, .(tree, leaf, variable, min, max)], 
                       lo, by = 'variable')
        }
        out <- merge(out, hi, by = 'variable')
      } else {
        # REVISIT
        bnds <- rbindlist(lapply(ooc, function(b) {
          readRDS(paste0('bnds', b, '.rds'))
        }))
        out <- merge(bnds, lo, by = 'variable')
        out <- merge(out, hi, by = 'variable')
      }
      # Feasibility constraint
      out[, feasible := all(min < f_max & max > f_min), by = .(tree, leaf)]
      out <- out[feasible == TRUE][, feasible := NULL]
      # Some variables may be uninformative
      tmp <- unique(out[, .(variable, min, max)])
      tmp[, n := .N, by = variable]
      if (tmp[, any(n == 1)]) {
        out <- out[variable %in% tmp[n > 1, unique(variable)]]
      }
      return(out)
    }
    
    # Some trees might be singletons 
    if (df[, any(n == 1)]) {
      if (df[, all(n == 1)]) {
        # If all are, we're done
        assignments <- df[, .(tree, leaf)]
      } else {
        # Otherwise, prepare to loop
        singletons <- df[n == 1, tree]
        leaf_cnt <- bnds[!tree %in% singletons, length(unique(leaf)), by = tree]
        leaf_cnt <- rbind(leaf_cnt, data.table('tree' = singletons, 'V1' = 1))
        cands <- merge(bnds[tree %in% singletons], df[n == 1, .(tree, leaf)])
        cands <- rbind(cands, bnds[!tree %in% singletons])
        bnds_lo <- emap$meta$metadata[variable %in% keep_vars, .(variable, min)]
        setnames(bnds_lo, 'min', 'f_min')
        bnds_hi <- emap$meta$metadata[variable %in% keep_vars, .(variable, max)]
        setnames(bnds_hi, 'max', 'f_max')
        assignments <- data.table('tree' = integer(), 'leaf' = integer())
        clique <- integer()
        ooc <- trees
        solo_flag <- TRUE
        while (isTRUE(solo_flag)) {
          # Find, append singletons
          clique <- append(clique, singletons)
          ooc <- setdiff(trees, clique)
          new_assignments <- unique(cands[tree %in% singletons, .(tree, leaf)])
          assignments <- rbind(assignments, new_assignments)
          if (assignments[, .N] == num_trees) break
          # Update the feasible region
          singleton_lo <- cands[tree %in% singletons, max(min), by = variable]
          singleton_hi <- cands[tree %in% singletons, min(max), by = variable]
          bnds_lo[, f_min := pmax(f_min, singleton_lo$V1)]
          bnds_hi[, f_max := pmin(f_max, singleton_hi$V1)]
          cands <- candidate_list(cands, bnds_lo, bnds_hi, ooc)
          # Check for singletons
          leaf_cnt <- cands[, length(unique(leaf)), by = tree]
          solo_flag <- any(leaf_cnt$V1 == 1)
          singletons <- leaf_cnt[V1 == 1, tree]
        }
      }
    } else {
      assignments <- data.table('tree' = integer(), 'leaf' = integer())
      clique <- integer()
      ooc <- trees
    }
    
    if (assignments[, .N] < num_trees) {
      
      # Continuing...
      trees_i <- sort(setdiff(df$tree, assignments$tree))
      df <- df[tree %in% trees_i]
      setorder(df, tree, leaf)
      B <- length(trees_i)
      
      # Compute reduced Phi
      phi <- function(b, sparse = TRUE) {
        node <- factor(emap$leafIDs[neighbors, b] + 1L)
        out <- model.matrix(~ 0 + node)
        if (isTRUE(sparse)) {
          out <- as(out, 'sparseMatrix') 
        }
        return(out)
      }
      Phi <- foreach(tree = trees_i, .combine = cbind, .packages = 'Matrix') %do% 
        phi(tree, Nl > 2000)
      
      # Lasso
      f <- glmnet(Phi, y, family = 'gaussian',
                  standardize = FALSE, intercept = FALSE,
                  lower.limits = 0, upper.limits = up, lambda = 1e-4)
      df[, coef := coef(f, s = 1e-4)[-1]]
      
      # Start with top leaves
      top_leaves <- df[, leaf[which.max(coef)], by = tree]
      setnames(top_leaves, 'V1', 'leaf')
      top_leaves <- merge(top_leaves, df, by = c('tree', 'leaf'))
      top_leaves <- top_leaves[coef > 0]
      top_leaves[, n := NULL]
      if (assignments[, .N] > 0) {
        tmp <- copy(assignments)
        tmp[, coef := 1]
        top_leaves <- rbind(top_leaves, tmp)
        setorder(top_leaves, tree, leaf)
      }
      trees_i <- top_leaves[, tree]
      B <- length(trees_i)
      if (!isTRUE(ram_flag)) {
        bnds <- rbindlist(lapply(top_leaves[, unique(tree)], function(b) {
          readRDS(paste0('bnds', b, '.rds'))
        }))
      }
      extrema <- merge(top_leaves, bnds[tree %in% top_leaves[, tree]], 
                       by = c('tree', 'leaf'))
      
      # Graph action
      intersections <- matrix(0L, nrow = B, ncol = B)
      for (b1 in 2:B) {
        for (b2 in 1:(b1 - 1)) {
          if (all(extrema[tree == trees_i[b1], max] > extrema[tree == trees_i[b2], min] & 
                  extrema[tree == trees_i[b2], max] > extrema[tree == trees_i[b1], min])) {
            intersections[b1, b2] <- 1L
          }
        }
      }
      intersection_graph <- graph_from_adjacency_matrix(intersections, mode = 'lower')
      clique <- largest_cliques(intersection_graph)
      if (length(clique) == B) {
        stop('The edge set is empty. Consider increasing sample size.')
      } else if (length(clique) == 1) {
        clique <- as.integer(clique[[1]])
      } else {
        scores <- sapply(clique, function(cl) {
          top_leaves[tree %in% top_leaves$tree[cl], sum(coef)]
        })
        clique <- as.integer(clique[[which.max(scores)]])
      }
      clique <- top_leaves[, tree][clique]
      ooc <- setdiff(trees, clique)
      assignments <- unique(extrema[tree %in% clique, .(tree, leaf)])
      
      # Search for consistent assignments till graph is complete
      if (assignments[, .N] < num_trees) {
        
        # Prune the search space to only the feasible leaves
        bnds_lo <- extrema[tree %in% clique, max(min), by = variable]
        setnames(bnds_lo, 'V1', 'f_min')
        bnds_hi <- extrema[tree %in% clique, min(max), by = variable]
        setnames(bnds_hi, 'V1', 'f_max')
        cands <- candidate_list(bnds, bnds_lo, bnds_hi, ooc)
        cands <- merge(cands, df, by = c('tree', 'leaf'), all.x = TRUE)
        cands[is.na(coef), coef := 0]
        # In case any variables have been dropped
        if (cands[, length(unique(variable))] < bnds_lo[, length(unique(variable))]) {
          now_vars <- cands[, unique(variable)]
          bnds_lo <- bnds_lo[variable %in% now_vars]
          bnds_hi <- bnds_hi[variable %in% now_vars]
        }
        
        # Stopping criterion
        stop_flag <- (assignments[, .N] == num_trees) | (cands[, .N] == 0L)
        
        # Greedy loop through remaining candidates, with special care for singletons
        while (!isTRUE(stop_flag)) {
          leaf_cnt <- cands[, length(unique(leaf)), by = tree]
          solo_flag <- any(leaf_cnt$V1 == 1)
          if (isTRUE(solo_flag)) {
            while (isTRUE(solo_flag)) {
              singletons <- leaf_cnt[V1 == 1, tree]
              clique <- append(clique, singletons)
              ooc <- setdiff(trees, clique)
              new_assignments <- unique(cands[tree %in% singletons, .(tree, leaf)])
              assignments <- rbind(assignments, new_assignments)
              if (assignments[, .N] == num_trees) break
              singleton_lo <- cands[tree %in% singletons, max(min), by = variable]
              singleton_hi <- cands[tree %in% singletons, min(max), by = variable]
              bnds_lo[, f_min := pmax(f_min, singleton_lo$V1)]
              bnds_hi[, f_max := pmin(f_max, singleton_hi$V1)]
              cands <- candidate_list(cands, bnds_lo, bnds_hi, ooc)
              if (cands[, .N] == 0L) break
              # In case any variables have been dropped
              if (cands[, length(unique(variable))] < bnds_lo[, length(unique(variable))]) {
                now_vars <- cands[, unique(variable)]
                bnds_lo <- bnds_lo[variable %in% now_vars]
                bnds_hi <- bnds_hi[variable %in% now_vars]
              }
              leaf_cnt <- cands[, length(unique(leaf)), by = tree]
              solo_flag <- any(leaf_cnt$V1 == 1)
            }
          }
          stop_flag <- (assignments[, .N] == num_trees) | (cands[, .N] == 0L)
          if (isTRUE(stop_flag)) break
          setorder(cands, -coef) 
          top_tree <- cands[, head(tree, 1)]
          top_leaf <- cands[tree == top_tree, head(leaf, 1)]
          tmp <- cands[tree == top_tree & leaf == top_leaf]
          clique <- append(clique, top_tree)
          ooc <- setdiff(trees, clique)
          new_assignments <- unique(tmp[, .(tree, leaf)])
          assignments <- rbind(assignments, new_assignments)
          if (assignments[, .N] == num_trees) break
          bnds_hi[, f_max := pmin(f_max, tmp$max)]
          bnds_lo[, f_min := pmax(f_min, tmp$min)]
          cands <- candidate_list(cands, bnds_lo, bnds_hi, ooc) 
          if (cands[, .N] == 0L) break
          # In case any variables have been dropped
          if (cands[, length(unique(variable))] < bnds_lo[, length(unique(variable))]) {
            now_vars <- cands[, unique(variable)]
            bnds_lo <- bnds_lo[variable %in% now_vars]
            bnds_hi <- bnds_hi[variable %in% now_vars]
          }
          stop_flag <- (assignments[, .N] == num_trees) | (cands[, .N] == 0L)
        }
      }
    }
    assignments[, idx := i]
    setorder(assignments, tree)
    # cat(paste0('Completed sample ', i, '...\n'))
    # flush.console()
    return(assignments)
  }
  
  cat('Greedy leaf approximation underway...\n')
  pkgs <- c('data.table', 'Matrix', 'glmnet', 
            'ExclusiveLasso', 'igraph', 'foreach')
  if (isTRUE(parallel)) {                                             
    P_hat <- foreach(i = seq_len(m), .combine = rbind, .packages = pkgs,
                     .export = c('bnds')) %dopar% loop(i)
  } else {
    P_hat <- foreach(i = seq_len(m), .combine = rbind) %do% loop(i)
  }
  
  
  ### Reconstruct data for output ###
  if (isTRUE(ram_flag)) {
    bnds2 <- merge(bnds, unique(P_hat[, .(tree, leaf)]), by = c('tree', 'leaf'), 
                   sort = FALSE)
  } else {
    bnds2 <- rbindlist(lapply(trees, function(b) {
      merge(readRDS(paste0('bnds', b, '.rds')), 
            unique(P_hat[tree == b, .(tree, leaf)]),
            by = c('tree', 'leaf'), sort = FALSE)
    }))
    file.remove(paste0('bnds', b, '.rds'))
  }
  df <- merge(bnds2, P_hat, by = c('tree', 'leaf'), allow.cartesian = TRUE, 
              sort = FALSE) 
  lo <- df[, max(min), by = .(idx, variable)]
  hi <- df[, min(max), by = .(idx, variable)]
  setnames(lo, 'V1', 'min')
  setnames(hi, 'V1', 'max')
  df <- merge(lo, hi, by = c('idx', 'variable'))
  
  # Any locally uninformative features?
  tmp <- emap$meta$metadata[, .(variable, g_min = min, g_max = max)]
  tmp2 <- merge(tmp, df, by = 'variable')
  tmp2[, trouble := min == g_min & max == g_max]
  if (any(tmp2$trouble)) {
    if (is.null(x)) {
      warning('Values for uninformative features will be drawn uniformly from ',
              'empirical bounds. This may produce suboptimal results for ',
              'low-entropy variables. Consider supplying training data x.')
    }
    df <- merge(tmp2[, .(variable, idx, trouble)], df, by = c('variable', 'idx'))
    df <- df[trouble == FALSE, .(idx, variable, min, max)]
  }
  tmp2 <- tmp2[trouble == TRUE, .(idx, variable, min, max)]
  
  # Synthesize
  df[, new := runif(.N, min, max)]
  df[, c('min', 'max') := NULL]
  
  # Deal with uninformative features
  if (d > d_tmp | any(tmp2$trouble)) {
    uninf_fn <- function(i) {
      # Who is uninformative?
      if (tmp2[idx == i, .N] > 0) {
        new_uninf <- tmp2[idx == i, variable]
        uninf_i <- c(uninformative, new_uninf)
      } else {
        uninf_i <- uninformative
      }
      # Do we have x data?
      if (!is.null(x)) {
        leaf_mat <- as.matrix(
          dcast(P_hat, idx ~ tree, value.var = 'leaf')[, idx := NULL]
        )
        # Who are my neighbors?
        tmp <- matrix(rep(leaf_mat[i, ], times = n), ncol = num_trees, byrow = TRUE)
        leafIDs <- predict(rf, x, type = 'terminalNodes')$predictions + 1L
        leafIDs <- leafIDs[, trees]
        adj <- rowSums(leafIDs == tmp, na.rm = TRUE)
        neighbors <- which(adj > 0)
        # Sample uninformative features
        out <- rbindlist(lapply(uninf_i, function(j) {
          data.table('variable' = j,
                     'new' = sample(x[neighbors, j], 1, prob = adj[neighbors]))
        }))
      } else {
        out <- emap$meta$metadata[variable %in% uninf_i, .(variable, min, max)]
        out[, new := runif(.N, min, max)]
        out[, c('min', 'max') := NULL]
      }
      out[, idx := i]
      return(out)
    }
    if (isTRUE(parallel)) {
      tmp <- foreach(ii = seq_len(m), .combine = rbind,
                     .packages = c('ranger')) %dopar% uninf_fn(ii)
    } else {
      tmp <- foreach(ii = seq_len(m), .combine = rbind) %do% uninf_fn(ii)
    }
    setcolorder(tmp, c('idx', 'variable', 'new'))
  }
  # if factor, must convert to int value to match rest of df
  if (any(tmp$variable %in% factor_names)) {
    tmp <- merge(tmp, lvls, by.x = c("variable", "new"), 
                 by.y = c("variable", "val"), all.x = TRUE)
    # warnings because we work with NA, but we can suppress since it's not
    # an issue ?
    suppressWarnings(tmp[, new := fifelse(is.na(number), 
                                          as.numeric(new), number)])
    tmp[, number := NULL]
  }
  
  # Cast
  if (any(factor_cols)) {
    df[variable %in% factor_names, new := round(new)]
  }
  df <- rbind(df, tmp)
  
  recovered <- dcast(df, idx ~ variable, value.var = 'new')[, idx := NULL]
  
  # Fix factor columns
  if (any(factor_cols)) {
    for (j in factor_names) {
      tmp2 <- lvls[variable == j, .(val, number)]
      setnames(tmp2, 'number', j)
      recovered <- merge(recovered, tmp2, by = j, sort = FALSE)[, c(j) := NULL]
      setnames(recovered, 'val', j)
    }
  }
  
  # Reshuffle, polish
  setcolorder(recovered, colnames_x)
  recovered <- post_x(recovered, emap$meta)
  
  # Export
  out <- list('recovered' = recovered, 'P_hat' = P_hat)
  cat('Done!')
  return(out)
}


