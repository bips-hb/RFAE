# Load libraries, too many and scared to remove 
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

source('encode.R')
source("decode_knn.R")

source("utils.R")
source("errors.R")

registerDoParallel(32)

loop <- function(dat, latent_rate=0.2, runs=10) {
  # Import data
  for (i in seq_along(dat)) {
    data <- dat[i]
    full <- fread(paste0('./decoder_sandbox/original_data/full/', data, '.csv'
    ),header=TRUE)
    bootstraps <- as.matrix(fread(paste0('./decoder_sandbox/original_data/full/bootstrap_', data, '.csv')))
    colnames(full) = make.names(colnames(full))
    d <- ncol(full)
    dz <- max(1, round(d * latent_rate))
    training_time <- c()
    inference_time <- c()
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
      # 
      trn <- full[bootstrap, ]
      setDT(trn)
      trn_obj <- prep_x(trn)
      tst <- full[setdiff(seq_len(nrow(full)), bootstrap)]
      tst <- prep_x(tst, trn_obj[[2]], trn_obj[[3]])[[1]]
      trn <- trn_obj[[1]]
      setDT(trn)
      setDT(tst)
      start.time <- Sys.time()
      rf <- adversarial_rf(trn, num_trees = 500)
      # rf <- ranger(x = trn, y = trn[,sample(c(0,1), .N, replace = T)], classification = T,
      #              num.trees = 500, mtry = ncol(trn), splitrule = "extratrees", num.random.splits = 1, respect.unordered.factors = "order")
      emap <- encode(rf, trn, k=dz)
      end.time <- Sys.time()
      training_time <- append(training_time, 
                              as.numeric(difftime(end.time, start.time, units='secs')))
      start.time <- Sys.time()
      z <- predict.encode(emap, rf, tst)
      out <- decode_knn(rf, emap, z, k = 20)$x_hat
      end.time <- Sys.time()
      inference_time <- append(inference_time, 
                               as.numeric(difftime(end.time, start.time, units='secs')))
    }
  }
  return(list(training_time = training_time, inference_time = inference_time))
}  

plpn <- loop('plpn')
bc <- loop('bc')
car <- loop('car')
student <- loop('student')
credit <- loop('credit')
wq <- loop('wq')
mushroom <- loop('mushroom')
abalone <- loop('abalone')
churn <- loop('churn')
diabetes <- loop('diabetes')
dry_bean <- loop('dry_bean')
forestfires <- loop('forestfires')
hd <- loop('hd')
king <- loop('king')
adult <- loop('adult')
marketing <- loop('marketing')
obesity <- loop('obesity')
telco <- loop('telco')

results_list <- list(
  plpn = plpn, bc = bc, car = car, student = student, credit = credit,
  wq = wq, mushroom = mushroom, abalone = abalone, churn = churn,
  diabetes = diabetes, dry_bean = dry_bean, forestfires = forestfires,
  hd = hd, king = king, adult = adult, marketing = marketing,
  obesity = obesity, telco = telco
)

summary_stats <- lapply(names(results_list), function(name) {
  res <- results_list[[name]]
  training_time <- res$training_time
  inference_time <- res$inference_time
  total_time <- training_time + inference_time
  
  data.frame(
    dataset = name,
    training_mean = mean(training_time),
    training_var = var(training_time),
    inference_mean = mean(inference_time),
    inference_var = var(inference_time),
    total_mean = mean(total_time),
    total_var = var(total_time)
  )
})

summary_table <- do.call(rbind, summary_stats)
print(summary_table)

fwrite(summary_table, "training_inference_summary.csv")
