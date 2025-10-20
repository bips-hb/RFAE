# Load libraries, too many and scared to remove 
library(glmnet)
library(Matrix)
library(stats)
library(foreach)
library(RSpectra)
library(doParallel)
library(tidyr)
library(dplyr)
library(RANN)
library(mgcv)
library(data.table)
library(ranger)
library(caret)
library(arf)
library(MASS)

source("pca.R")
source('encode.R')
source("decode_knn.R")
source("decode_relabel.R")
source("decode_lasso.R")
source("decoder_sandbox/eForest.R")
source("decoder_sandbox/eForestSynth.R")
source('approx_ranger.R')
source("utils.R")
source("errors.R")

# Register cores
cl <- makeCluster(8)
registerDoParallel(cl)

loop <- function(dat, latent_rate=0.2, runs=5) {
  # Import data
  for (i in seq_along(dat)) {
    data <- dat[i]
    full <- fread(paste0('../original_data/full/', data, '.csv'
    ),header=TRUE)
    bootstraps <- as.matrix(fread(paste0('../original_data/full/bootstrap_', data, '.csv')))
    colnames(full) = make.names(colnames(full))
    d <- ncol(full)
    dz <- max(1, round(d * latent_rate))
    # aurf loop
    for (j in seq(runs)) {
      bootstrap = bootstraps[, j]
      # Unsupervised Random Forest Setup
      # Then, need to change emap hyp to trn_og
      # trn_og <- full[bootstrap, ]
      # set.seed(1234)
      # synth <- as.data.frame(lapply(trn_og, sample,
      #                               nrow(trn_og) , replace = TRUE))
      # trn <- rbind(data.frame(y = 1, trn_og),
      #                    data.frame(y = 0, synth))
      # setDT(trn)
      # trn_obj <- prep_x(trn)
      # tst <- full[setdiff(seq_len(nrow(full)), bootstrap)]
      # tst <- prep_x(tst, trn_obj[[2]], trn_obj[[3]])[[1]]
      # trn <- trn_obj[[1]]
      # setDT(trn)
      # setDT(tst)
      # 
      # trn_og <- trn[y == 1]
      # trn_og[, y := NULL]
      
      trn <- full[bootstrap, ]
      setDT(trn)
      
      trn_obj <- prep_x(trn)
      tst <- full[setdiff(seq_len(nrow(full)), bootstrap)]
      tst <- prep_x(tst, trn_obj[[2]], trn_obj[[3]])[[1]]
      trn <- trn_obj[[1]]
      setDT(trn)
      setDT(tst)
      set.seed(1234)
      rf <- adversarial_rf(trn, num_trees = 50)
      emap <- encode(rf, trn, k=dz)
      z <- predict.encode(emap, rf, tst)
      out <- decode_knn(rf, emap, z, k = 20)$x_hat
      fwrite(out, paste0('decode_data/rfae_data/', data, '/', latent_rate, '_run', j, '.csv'))
      
      decoder <- train_decode_relabel(rf, emap, trn, parallel = FALSE)
      out <- decode_relabel(decoder, z, trn)
      fwrite(out, paste0('decode_data/relabel_data/', data, '/', latent_rate, '_run', j, '.csv'))
      set.seed(4321)
      out <- decode_lasso(z, rf, emap, sparsity = 'global', k = 100, parallel = TRUE)
      out <- out$recovered
      fwrite(out, paste0('decode_data/lasso_data/', data, '/', latent_rate, '_run', j, '.csv'))
    }
  }
}  

compressions <- c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1)



for (i in seq_along(compressions)) {
  loop('student', latent_rate=compressions[i])
}
for (i in seq_along(compressions)) {
  loop('credit', latent_rate=compressions[i])
}


