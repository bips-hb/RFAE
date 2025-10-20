# Load libraries

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

source("pca.R")
source('encode.R')
source("decode_knn.R")
source("utils.R")
source("errors.R")

# Register cores
cl <- makeCluster(8)
registerDoParallel(cl)

loop <- function(dat, latent_rate=0.2, runs=10) {
  # Import data
  for (i in seq_along(dat)) {
    data <- dat[i]
    full <- fread(paste0('../../original_data/', data, '.csv'
    ),header=TRUE)
    bootstraps <- as.matrix(fread(paste0('../../original_data/bootstrap_', data, '.csv')))
    spambase_cols <- colnames(full)
    d <- ncol(full)
    dz <- max(1, round(d * latent_rate))
    # aurf loop
    for (j in seq(runs)) {
      bootstrap = bootstraps[, j]
      trn_og <- full[bootstrap, ]
      set.seed(1234)
      trn_obj <- prep_x(trn_og, default = 1)
      tst <- full[setdiff(seq_len(nrow(full)), bootstrap)]
      tst <- prep_x(tst, trn_obj[[2]], trn_obj[[3]], default = 1)[[1]]
      trn_og <- trn_obj[[1]]
      setDT(trn_og)
      setDT(tst)
      setnames(trn_og, paste0("V", seq_len(ncol(trn_og))))
      setnames(tst, paste0("V", seq_len(ncol(tst))))
      
      rf <- adversarial_rf(trn_og, num_trees = 500)
      #rf <- ranger(y ~ ., data = trn, num.trees = 200,
      #classification = TRUE, respect.unordered.factors = 'order')
      emap <- encode(rf, trn_og, k=dz)
      z <- predict.encode(emap, rf, tst)
      out <- decode_knn(rf, emap, z, k = 20)$x_hat
      colnames(out) <- spambase_cols
      fwrite(out, paste0('./decoder_sandbox/rfae_data/', data, '/', latent_rate, '_run', j, '.csv'))
      
    }
  }
}  

compressions <- seq(0.1, 1, 0.1)

set.seed(1234)


for (i in seq_along(compressions)) {
  loop('spambase', latent_rate=compressions[i])
}
