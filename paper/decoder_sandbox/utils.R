#' Preprocess input data
#' 
#' This function prepares input data for ARFs.
#' 
#' @param x Input data.frame.
#' 
#' @import arf

col_rename <- function(cn, old_name) {
  k <- 1L
  converged <- FALSE
  while (!isTRUE(converged)) {
    new_name <- paste0(old_name, k)
    if (!new_name %in% cn) {
      converged <- TRUE
    } else {
      k <- k + 1L
    }
  }
  return(new_name)
}

prep_x <- function(x) {
  x <- as.data.frame(x)
  idx_char <- sapply(x, is.character)
  if (any(idx_char)) {
    x[, idx_char] <- lapply(x[, idx_char, drop = FALSE], as.factor)
  }
  idx_logical <- sapply(x, is.logical)
  if (any(idx_logical)) {
    x[, idx_logical] <- lapply(x[, idx_logical, drop = FALSE], as.factor)
  }
  idx_integer <- sapply(x, is.integer)
  if (any(idx_integer)) {
    # Recoding integers with > 5 levels as numeric
    to_numeric <- sapply(seq_len(ncol(x)), function(j) {
      idx_integer[j] & length(unique(x[[j]])) > 5
    })
    if (any(to_numeric)) {
      warning('Recoding integers with more than 5 unique values as numeric. ', 
              'To override this behavior, explicitly code these variables as factors.')
      x[, to_numeric] <- lapply(x[, to_numeric, drop = FALSE], as.numeric)
    }
    to_factor <- sapply(seq_len(ncol(x)), function(j) {
      idx_integer[j] & length(unique(x[[j]])) < 6
    })
    if (any(to_factor)) {
      warning('Recoding integers with fewer than 6 unique values as ordered factors. ', 
              'To override this behavior, explicitly code these variables as numeric.')
      x[, to_factor] <- lapply(which(to_factor), function(j) {
        lvls <- sort(unique(x[[j]]))
        factor(x[[j]], levels = lvls, ordered = TRUE)
      })
    }
  }
  # Rename annoying columns
  if ('y' %in% colnames(x)) {
    colnames(x)[which(colnames(x) == 'y')] <- col_rename(x, 'y')
  }
  if ('obs' %in% colnames(x)) {
    colnames(x)[which(colnames(x) == 'obs')] <- col_rename(x, 'obs')
  }
  if ('tree' %in% colnames(x)) {
    colnames(x)[which(colnames(x) == 'tree')] <- col_rename(x, 'tree')
  } 
  if ('leaf' %in% colnames(x)) {
    colnames(x)[which(colnames(x) == 'leaf')] <- col_rename(x, 'leaf')
  }
  if ('cnt' %in% colnames(x)) {
    colnames(x)[which(colnames(x) == 'cnt')] <- col_rename(x, 'cnt')
  }
  if ('N' %in% colnames(x)) {
    colnames(x)[which(colnames(x) == 'N')] <- col_rename(x, 'N')
  }
  return(x)
}

