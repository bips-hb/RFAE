
library(Rcpp)
library(foreach)

Rcpp::sourceCpp("approx_ranger.cpp")

approx_ranger <- function(rf, x_old, x_new, num_random_splits = 1000L, parallel = FALSE) {
  rf2 <- data.table::copy(rf)
  
  if (parallel) {
    tree_fun <- function(tree) {
      approx_ranger_par_cpp(rf2$forest$child.nodeIDs[[tree]], rf2$forest$split.varIDs[[tree]], 
                            rf2$forest$split.values[[tree]],
                            x_old = x_old, x_new = x_new, 
                            num_random_splits = num_random_splits)
      list(split.varIDs = rf2$forest$split.varIDs[[tree]], 
           split.values = rf2$forest$split.values[[tree]])
    }
    res <- foreach(b = seq_len(rf2$num.trees)) %dopar% tree_fun(b)
    #res <- mclapply(seq_len(rf2$num.trees), tree_fun, mc.cores = cores)
    
    for (i in seq_len(rf2$num.trees)) {
      rf2$forest$split.varIDs[[i]] <- res[[i]]$split.varIDs
      rf2$forest$split.values[[i]] <- res[[i]]$split.values
    }
  } else {
    approx_ranger_cpp(rf2$forest$child.nodeIDs, rf2$forest$split.varIDs, rf2$forest$split.values,
                      x_old = x_old, x_new = x_new, num_random_splits = num_random_splits)
  }
  
  rf2$forest$independent.variable.names <- colnames(x_new)
  rf2
}

approx_ranger_local <- function(rf, x_old, x_new) {
  rf2 <- data.table::copy(rf)
  approx_ranger_local_cpp(rf2$forest$child.nodeIDs, rf2$forest$split.varIDs, rf2$forest$split.values,
                    x_old = as.matrix(x_old), x_new = as.matrix(x_new))
  rf2$forest$independent.variable.names <- colnames(x_new)
  rf2
}



# Test 1 ------------------------------------------------------------------
# library(ranger)
# 
# x <- iris[, -5]
# y <- iris[, 5]
# z <- prcomp(x)$x[, 1:3]
# 
# rf <- ranger(y = y, x = x, num.trees = 2, max.depth = 2)
# rf2 <- approx_ranger(rf, x, z)
# 
# rf$forest$split.varIDs
# rf$forest$split.values
# 
# rf2$forest$split.varIDs
# rf2$forest$split.values

# Test 2 ------------------------------------------------------------------
# library(ranger)
# 
# x <- iris[, -5]
# y <- iris[, 5]
# z <- prcomp(x)$x[, 1:2, drop = FALSE]
# 
# rf <- ranger(y = y, x = x, num.trees = 50)
# rf2 <- approx_ranger(rf, x, z)
# 
# pred1 <- predict(rf, x, type = "terminalNodes")$predictions
# pred2 <- predict(rf2, z, type = "terminalNodes")$predictions
# 
# mean(pred1 == pred2)
