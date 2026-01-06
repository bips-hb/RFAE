library(Matrix)
library(doParallel)
library(tidyr)
library(dplyr)
library(RANN)
library(mgcv)
library(data.table)
library(ranger)
library(caret)
library(arf)
library(RSpectra)
# source('R/encode.R')
# source("R/decode_knn.R")
# source("R/utils.R")
# source("R/errors.R")

x = fread('sanity.csv')
names(x) = make.names(names(x))
rf <- ranger(y ~ ., data = x, num.trees = 200,
classification = TRUE, max.depth = 3)


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
e <- eigs(A, 6)
e
fwrite(as.matrix(A), 'sanity_adj.csv')
