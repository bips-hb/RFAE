# Load libraries
library(data.table)
library(ranger)
library(ggplot2)
library(ggsci)
library(Matrix)
library(RSpectra)
library(igraph)
library(glmnet)
library(doMC)
registerDoMC(12)

# Load internal functions
setwd('~/Library/Mobile\ Documents/com~apple~CloudDocs/Documents/Kings/tree_kernels')
source('eigenmap.R')
source('decode.R')
source('utils.R')

# Data
prep_data <- function(dat) {
  colnames(dat) <- c('age', 'workclass', 'fnlwgt', 'education', 'education.num',
                     'marital.status', 'occupation', 'relationship', 'race', 'sex',
                     'capital.gain', 'capital.loss', 'hours.per.week', 
                     'native.country', 'income')
  dat[, income := fifelse(grepl('>50K', income), 1L, 0L)]
  # dat[, income := NULL]    # This is binarized
  dat[, education := NULL] # This is covered by education-num
  dat[, fnlwgt := NULL]    # This is for demographic weighting, not prediction
  dat[, age := as.numeric(age)]
  dat[, education.num := as.numeric(education.num)]
  dat[, capital.gain := as.numeric(capital.gain)]
  dat[, capital.loss := as.numeric(capital.loss)]
  dat[, hours.per.week := as.numeric(hours.per.week)]
  dat <- as.data.table(prep_x(dat))
  return(dat)
}
trn <- prep_data(fread('./decoder_sandbox/adult/adult.data'))
tst <- prep_data(fread('./decoder_sandbox/adult/adult.test'))
d <- ncol(trn) - 1L


# Train model
set.seed(123)
rf <- ranger(income ~ ., data = trn, classification = TRUE, 
             respect.unordered.factors = TRUE, 
             max.depth = 15, num.trees = 200)

# How does it do?
# yhat <- predict(rf, tst)$predictions
# table(tst$y, yhat)

# Learn the embedding on a subset
n_trn <- trn[, .N]
trn_idx <- sample(n_trn, round(n_trn/5))
emap <- eigenmap(rf, trn[trn_idx, ], k = d - 1L)

# Evaluate reconstruction on a test set
n_tst <- tst[, .N]
tst_idx <- sample(n_tst, 1000)
# p_true <- predict(rf, tst, type = 'terminalNodes')$predictions
z <- predict.eigenmap(emap, rf, tst[tst_idx, ])

# Loop for different embedding dimensions
loop <- function(k) {
  
  # Reset dimension
  emap_tmp <- emap
  emap_tmp$z <- emap_tmp$z[, seq_len(k)]
  emap_tmp$w <- emap_tmp$w[, seq_len(k)]
  emap_tmp$v <- emap_tmp$v[, seq_len(k)]
  emap_tmp$lambda <- emap_tmp$lambda[seq_len(k)]
  z_tmp <- z[, seq_len(k)]
  if (k == 1) {
    z_tmp <- as.matrix(z_tmp, nrow = 1)
  }
  
  # Decode
  res <- decode(z_tmp, rf, emap_tmp, x = trn[trn_idx, ], max_mem = 100)
  
  # Write to disk
  saveRDS(res$recovered, paste0('./decoder_sandbox/adult_res_k=', k, '.rds'))
  
  # How we doing?
  cat(paste0('Completed k = ', k, '...\n'))
  
}
foreach(kk = 1:(d - 1)) %do% loop(kk)

# Now vary num_trees with fixed k
k <- round(0.2 * d) # k=2
emap_tmp <- emap
emap_tmp$z <- emap_tmp$z[, seq_len(k)]
emap_tmp$w <- emap_tmp$w[, seq_len(k)]
emap_tmp$v <- emap_tmp$v[, seq_len(k)]
emap_tmp$lambda <- emap_tmp$lambda[seq_len(k)]
z_tmp <- z[, seq_len(k)]

loop2 <- function(b) {
  
  # Decode
  res <- decode(z_tmp, rf, emap_tmp, x = trn[trn_idx, ], max_mem = 100,
                num_trees = b)
  
  # Write to disk
  saveRDS(res$recovered, paste0('./decoder_sandbox/adult_res_b=', b, '.rds'))
  
  # How we doing?
  cat(paste0('Completed b = ', b, '...\n'))
  
}
foreach(bb = seq(20, 200, 20)) %do% loop2(bb)







# 
# 
# 
# emap_tmp <- emap
# 
# # Find true adjacency vector
# leafIDs <- predict(rf, trn[trn_idx, ], type = 'terminalNodes')$predictions + 1L
# preds <- matrix(
#   rep(predict(rf, tst[1, ], type = 'terminalNodes')$predictions + 1L,
#       times = length(trn_idx)), ncol = 200, byrow = TRUE
# )
# # leafIDs <- predict(rf, trn, type = 'terminalNodes')$predictions + 1L
# # preds <- matrix(
# #   rep(predict(rf, tst[1, ], type = 'terminalNodes')$predictions + 1L,
# #       times = n_trn), ncol = 200, byrow = TRUE
# # )
# A <- rowSums(leafIDs == preds)
# neighbors <- which(A > 0)
# k <- length(neighbors)
# y <- A[neighbors]
# 
# z <- predict.eigenmap(emap, rf, tst[1:2, ])
# sparse <- round <- parallel <- TRUE
# num_trees <- NULL
# exclusive <- FALSE
# max_mem <- 100
# x <- trn[trn_idx, ]


