  library(data.table)
  library(doParallel)
  library(ranger)
  library(torchvision)
  library(Matrix)
  library(RSpectra)
  library(stats)
  library(RANN)
  source('encode.R')
  source('decode_knn.R')
  source('utils.R')
  source('./decoder_sandbox/eForest.R')
  source('./decoder_sandbox/eForestSynth.R')
  
  registerDoParallel(48)
  set.seed(42)
  
  mnist_train_generator <- mnist_dataset(
    root = "visual_experiments/data",
    download = TRUE,
    train = TRUE,
    transform = NULL
  )
  
  mnist_test_generator <- mnist_dataset(
    root = "visual_experiments/data",
    download = TRUE,
    train = FALSE,
    transform = NULL
  )
  
  mnist_train_array <- mnist_train_generator$data/255
  mnist_test_array <- mnist_test_generator$data/255
  
  mnist_train_y <- as.numeric(mnist_train_generator$targets)
  
  # reshape to 28x28
  mnist_train <- as.data.table(matrix(mnist_train_array, nrow = nrow(mnist_train_array), ncol = 28 * 28))
  mnist_test <- as.data.table(matrix(mnist_test_array, nrow = nrow(mnist_test_array), ncol = 28 * 28))
  
  # varying parameters
  params <- c("B", "n_train_enc", "d_Z", "t", "k")
  
  # defaults
  B_default = 1000
  n_train_enc_default <- 60000 # results for 30k in appendix
  d_Z_default <- 32
  t_default <- 1
  k_default <- 50
  
  params_defaults <- list(
    B = B_default,
    n_train_enc = n_train_enc_default,
    d_Z = d_Z_default,
    t = t_default,
    k = k_default
  )
  
  # ranges
  B_range <- c(50, 100, 200, 500, 1000)
  n_train_enc_range <- c(1000, 5000, 10000, 30000, 60000)
  d_Z_range <- c(2, 4, 8, 16, 32)
  t_range <- c(0, 1, 2, 3, 10, 50)
  k_range <- c(1, 2, 3, 5, 10, 20, 50)
  
  params_ranges <- list(
    B = B_range,
    n_train_enc = n_train_enc_range,
    d_Z = d_Z_range,
    t = t_range,
    k = k_range
  )
  
  rf <- NULL
  train_enc_idx_list <- list("1000" = sample(nrow(mnist_train), 1000),
                             "5000" = sample(nrow(mnist_train), 5000),
                             "10000" = sample(nrow(mnist_train), 10000),
                             "30000" = sample(nrow(mnist_train), 30000),
                             "60000" = sample(nrow(mnist_train), 60000))
  test_idx <- sapply(seq(10), \(digit) which(as.numeric(mnist_test_generator$targets) == digit)[1])  # first occurrences of each digit in test
  
  for (param in params) {
    params_run <- params_defaults
    
    # if folder does not exist create
    if (!dir.exists(paste0("visual_experiments/reconstructions/mnist/", param))) {
      dir.create(paste0("visual_experiments/reconstructions/mnist/", param), recursive = TRUE)
    }
    
    # save original test
    fwrite(mnist_test[test_idx, ], file = paste0("visual_experiments/reconstructions/mnist/", param, "/original.csv"))
    
    for (param_value in params_ranges[[param]]) {
      params_run[[param]] <- param_value
      
      # print all params_run
      cat("Running with parameters:\n")
      for (p in params) {
        cat(paste0(p, ": ", params_run[[p]], "\n"))
      }
      
      forest_retrain <- is.null(rf) || params_run$B != rf$num.trees
      
      if (forest_retrain) {
        rf <- ranger(x = mnist_train, y = mnist_train_y, num.trees = params_run$B, classification = TRUE, num.threads = 48)
      }
  
      train_enc_idx <- train_enc_idx_list[[as.character(params_run$n_train_enc)]]
      
      embedding_retrain <- !(param == "k" & (params_run$k != params_ranges$k[1]))
      
      if (embedding_retrain) {
        emap <- encode(rf, mnist_train[train_enc_idx, ], k = params_run$d_Z, steps = params_run$t, parallel = TRUE)
        test_enc <- predict.encode(emap, rf, mnist_test[test_idx, ], parallel = TRUE)
      }
      
      # decode
      reconstructed <- decode_knn(rf, emap, test_enc, k = params_run$k, parallel = TRUE)$x_hat
   
      fwrite(reconstructed, file = paste0("visual_experiments/reconstructions/mnist/", param, "/RFAE_", param, "_", param_value, ".csv"))
    }
  }
  

  ### selected combinations
  
  # for method comparison with ConvAE
  
  set.seed(42)
  
  rf <- ranger(x = mnist_train, y = mnist_train_y, num.trees = 1000, classification = TRUE, num.threads = 48)
  emap <- encode(rf, mnist_train, k = 32, stepsize = 1, parallel = TRUE)
  test_enc <- predict.encode(emap, rf, mnist_test[test_idx, ], parallel = TRUE)
  reconstructed <- decode_knn(rf, emap, test_enc, k = 50, parallel = TRUE)$x_hat
  fwrite(reconstructed, file = paste0("visual_experiments/reconstructions/mnist/", "method_comparison", "/RFAE.csv"))

  # unsupervised (pak::pkg_install("imbs-hl/ranger@completely_random_forests"))
  
  set.seed(42)
  
  urf <- ranger(x = mnist_train, y = mnist_train_y, num.trees = 1000, classification = TRUE, mtry = ncol(mnist_train),
                splitrule = "extratrees", num.random.splits = 1, num.threads = 48)
  emap <- encode(urf, mnist_train, k = 32, parallel = TRUE)
  test_enc <- predict.encode(emap, urf, mnist_test[test_idx, ], parallel = TRUE)
  reconstructed <- decode_knn(urf, emap, test_enc, k = 50, parallel = TRUE)$x_hat
  fwrite(reconstructed, file = paste0("visual_experiments/reconstructions/mnist/", "method_comparison", "/URFAE.csv"))
