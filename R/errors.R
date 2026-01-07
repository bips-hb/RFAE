#' Mixed-type Reconstruction Error
#'
#'
#' @param Xhat Reconstructed dataset
#' @param X Ground truth dataset
#'
#'
#' @details
#' In standard AEs, reconstruction error is generally estimated via \eqn{L_2}
#' loss. This is not sensible with a mix of continuous and categorical data, so
#' we devise a measure that evaluates distortion on continuous variables as
#' \eqn{1 - R^2}, and categorical variables as accuracy.
#'
#' @return
#' A list containing column-wise reconstruction accuracy, and the average
#' reconstruction error for categorical and numeric variables. Values lie
#' between 0-1, where 1 represents perfect reconstruction, and 0 represents no
#' reconstruction. Note that in the original paper, this is presented as the
#' distortion - this is because we do 1 - error in the plotting stage.
#'
#' @examples
#' # Set seed
#' set.seed(1)
#'
#' # Split training and test
#' trn <- sample(1:nrow(iris), 100)
#' tst <- setdiff(1:nrow(iris), trn)
#'
#' # Train RF, learn the encodings and project test points.
#' rf <- ranger::ranger(Species ~ ., data = iris[trn, ], num.trees=50)
#' emap <- encode(rf, iris[trn, ], k=2)
#' emb <- predict(emap, rf, iris[tst, ])
#'
#' # Decode test samples back to the input space
#' out <- decode_knn(rf, emap, emb, k=5)$x_hat
#'
#' # Compute the reconstruction error
#'
#' @export
#' @import caret
#'
reconstruction_error <- function(Xhat, X) {
  num_error <-  list()
  cat_error <-  list()
  ovr_error <- 0
  for (i in colnames(X)) {
    if (is.numeric(X[[i]])) {
      #min <- min(X[[i]])
      #max <- max(X[[i]])
      #error <- sqrt(mean((Xhat[[i]] - X[[i]])^2))/(max - min)
      #error <- 1 - cor(Xhat[[i]], X[[i]]) ^ 2
      rss <- sum((X[[i]] - Xhat[[i]])^2)
      tss <- sum((X[[i]] - mean(X[[i]]))^2)
      num_error[[i]] <- max(1 - (rss/tss), 0)
    }
    else {
      yhat <- as.character(Xhat[[i]])
      y <- as.character(X[[i]])
      error <- sum(yhat == y) / nrow(X)
      #error <- f1_score(X[[i]], Xhat[[i]])
      cat_error[[i]] <- error
    }
  }
  if (length(num_error)) {
    num_avg = mean(unlist(num_error))
    ovr_error = ovr_error + sum(unlist(num_error))
  } else {
    num_avg = 'No variables'
  }

  if (length(cat_error)) {
    cat_avg = mean(unlist(cat_error))
    ovr_error = ovr_error + sum(unlist(cat_error))
  } else {
    cat_avg = 'No variables'
  }
  ovr_error <- ovr_error/ncol(X)
  out <- list(num_error = num_error, cat_error = cat_error,
              num_avg = num_avg,
              cat_avg = cat_avg,
              ovr_error = ovr_error)
  return(out)
}
