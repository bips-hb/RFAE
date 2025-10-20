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

source("encode.R")
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

      rf <- adversarial_rf(trn, num_trees = 500)
      #rf <- ranger(x = trn, y = trn[,sample(c(0,1), .N, replace = T)], classification = T,
      #             num.trees = 500, mtry = ncol(trn), splitrule = "extratrees", num.random.splits = 1, respect.unordered.factors = "order")
      emap <- encode(rf, trn, k=dz)
      z <- predict.encode(emap, rf, tst)
      out <- decode_knn(rf, emap, z, k = 20)$x_hat
      fwrite(out, paste0(data, '/', latent_rate, '_run', j, '.csv'))
      
    }
  }
}  

compressions <- c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1)

set.seed(1234)

for (i in seq_along(compressions)) {
  loop('abalone', latent_rate=compressions[i])
}

for (i in seq_along(compressions)) {
  loop('adult', latent_rate=compressions[i])
}

for (i in seq_along(compressions)) {
  loop('banknote', latent_rate=compressions[i])
}

for (i in seq_along(compressions)) {
  loop('bc', latent_rate=compressions[i])
}

for (i in seq_along(compressions)) {
  loop('car', latent_rate=compressions[i])
}

for (i in seq_along(compressions)) {
  loop('churn', latent_rate=compressions[i])
}

for (i in seq_along(compressions)) {
  loop('credit', latent_rate=compressions[i])
}

for (i in seq_along(compressions)) {
  loop('diabetes', latent_rate=compressions[i])
}

for (i in seq_along(compressions)) {
  loop('dry_bean', latent_rate=compressions[i])
}

for (i in seq_along(compressions)) {
  loop('forestfires', latent_rate=compressions[i])
}

for (i in seq_along(compressions)) {
  loop('hd', latent_rate=compressions[i])
}

for (i in seq_along(compressions)) {
  loop('king', latent_rate=compressions[i])
}

for (i in seq_along(compressions)) {
  loop('marketing', latent_rate=compressions[i])
}

for (i in seq_along(compressions)) {
  loop('obesity', latent_rate=compressions[i])
}

for (i in seq_along(compressions)) {
  loop('plpn', latent_rate=compressions[i])
}

for (i in seq_along(compressions)) {
  loop('wq', latent_rate=compressions[i])
}

for (i in seq_along(compressions)) {
  loop('student', latent_rate=compressions[i])
}

for (i in seq_along(compressions)) {
  loop('telco', latent_rate=compressions[i])
}

for (i in seq_along(compressions)) {
  loop('mushroom', latent_rate=compressions[i])
}