#' Synthesize data from autoencoder forest
#' 
#' Takes uniform draws from hyperrectangles computed by an autoencoder forest.
#' 
#' @param rf Random forest model object of class \code{ranger}.
#' @param e_forest Output of \code{eForest}
#'
#' @details
#' This function creates synthetic samples from an autoencoder forest. 
#' 
#' @return
#' A dataset of synthetic samples.
#' 
#' @references 
#' Feng, J. and Zhou, Z. (2017). AutoEncoder by Forest. \emph{arXiv} preprint, 
#' 1709.09018. 
#' 
#' @examples
#' # Encode model
#' library(ranger)
#' rf <- ranger(Species ~ ., data = iris)
#' z <- eForest(rf, iris)
#' 
#' # Synthesize
#' synth <- eForestSynth(rf, iris, z, 100)
#' 
#' @export
#' @import data.table
#' @import ranger

eForestSynth <- function(rf, emap, e_forest) {
  
  # Prelimz
  num_trees <- rf$num.trees
  colnames_x <- emap$meta$metadata$variable
  n <- nrow(emap$leafIDs)
  d <- length(colnames_x)
  factor_cols <- emap$meta$metadata$fctr
  factor_names <- emap$meta$metadata[fctr == TRUE, variable]
  lvls <- emap$meta$levels
  synth <- e_forest
  informative <- colnames_x %in% synth$variable
  
  # Any globally uninformative features?
  if (any(!informative)) {
    uninf_vars <- colnames_x[!informative]
    uninf_df <- rbindlist(lapply(uninf_vars, function(j) {
      data.table('obs' = synth[, unique(obs)], 
                 'variable' = j, 
                 'min' = emap$meta$metadata[variable == j, min],
                 'max' = emap$meta$metadata[variable == j, max])
    }))
    synth <- rbind(synth, uninf_df)
  }
  
  # Synthesize
  synth[, fctr := rep(factor_cols, n)]
  synth[, new := runif(.N, min, max)]
  synth[fctr == TRUE, new := round(new)]
  synth <- dcast(synth, obs ~ variable, value.var = 'new')[, obs := NULL]
  
  # Fix factor columns
  if (any(factor_cols)) {
    for (j in factor_names) {
      tmp <- lvls[variable == j, .(val, number)]
      setnames(tmp, 'number', j)
      synth <- merge(synth, tmp, by = j, sort = FALSE)[, c(j) := NULL]
      setnames(synth, 'val', j)
    }
  }
  setcolorder(synth, colnames_x)
  synth <- post_x(synth, emap$meta)
  return(synth)
  
}