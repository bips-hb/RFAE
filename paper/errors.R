#' Calculate reconstruction errors. Function calculates seperate reconstruction
#' errors for categorical and numerical variables 
#' 
#' 
#' @param Xhat Reconstructed dataset
#' @param X Ground truth dataset
#' 
#' 
#' @details 
#' 
#' 
#' @return 
#' Column-wise reconstruction error, and the average reconstruction 
#' error for categorical and numeric variables
#' 
#' 
#' @references 
#' 
#' 
#' @examples
#' 
#' 
#' @seealso
#' 
#' 
#' @export
#' @import caret
#'

f1_score <- function(y, yhat) {
  lvls <- union(y, yhat)
  y <- factor(as.character(y), levels = lvls)
  yhat <- factor(as.character(yhat), levels = lvls)
  cm <- suppressWarnings(confusionMatrix(yhat, y))
  
  if (nlevels(as.factor(y)) > 2) {
    f1 <- cm$byClass[, "F1"]
    prevalence <- cm$byClass[, "Prevalence"]
    f1[is.na(f1)] <- 0
    
    weighted_f1 <- sum(f1 * prevalence)
  } else {
    weighted_f1 <- cm$byClass["F1"]
  }
  
  return(weighted_f1)
}
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
