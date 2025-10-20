
library(Rcpp)

Rcpp::sourceCpp("approx_ranger_gini.cpp")

approx_ranger_gini <- function(rf, x_old, x_new) {
  rf2 <- data.table::copy(rf)
  approx_ranger_gini_cpp(rf2$forest$child.nodeIDs, rf2$forest$split.varIDs, rf2$forest$split.values,
                    x_old = x_old, x_new = x_new)
  rf2$forest$independent.variable.names <- colnames(x_new)
  rf2
}

approx_ranger_gini_local <- function(rf, x_old, x_new) {
  rf2 <- data.table::copy(rf)
  approx_ranger_local_gini_cpp(rf2$forest$child.nodeIDs, rf2$forest$split.varIDs, rf2$forest$split.values,
                          x_old = as.matrix(x_old), x_new = as.matrix(x_new))
  rf2$forest$independent.variable.names <- colnames(x_new)
  rf2
}