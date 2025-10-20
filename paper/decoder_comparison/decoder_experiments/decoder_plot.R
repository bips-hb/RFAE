# Load libraries
library(ggplot2)
library(ggsci)
library(Matrix)
library(stats)
library(tidyr)
library(dplyr)
library(stringr)
library(data.table)
library(RColorBrewer)

# Load internal functions
#setwd('~/Library/Mobile\ Documents/com~apple~CloudDocs/Documents/Kings/tree_kernels')

source("utils.R")
source("errors.R")
loop <- function(dat) {
  # Import data
  # make list for errors
  results <- list()
  raw_errors <- data.frame(dataset = character(), x = numeric(),
                           run = integer(), method = character(),
                           error = numeric()
  )
  for (i in seq_along(dat)) {
    data <- dat[i]
    full <- fread(paste0('./decoder_sandbox/original_data/full/', data, '.csv'
    ),header=TRUE)
    bootstraps <- as.matrix(fread(paste0('./decoder_sandbox/original_data/full/bootstrap_', data, '.csv')))
    
    errors <- list()
    compressions <- as.character(seq(0.1, 1, 0.1))
    for (c in compressions) {
      
      rfae <- lasso <- relabel <- c()
      for (k in seq(5)) {
        
        bootstrap = bootstraps[, k]
        trn_og <- full[bootstrap, ]
        setDT(trn_og)
        trn_obj <- prep_x(trn_og, default = 1)
        tst <- full[setdiff(seq_len(nrow(full)), bootstrap)]
        tst <- prep_x(tst, trn_obj[[2]], trn_obj[[3]])[[1]]
        setDT(tst)
        for (method in c('rfae', 'lasso', 'relabel')) {
          out <- fread(paste0('./decoder_sandbox/decode_data/', 
                              method, '_data/', data, '/', c, '_run', k, '.csv'))
          setnames(out, gsub("\\.", "-", colnames(out)))
          
          err <- reconstruction_error(out, tst)$ovr_error
          if (!is.na(err)) {
            assign(method, append(get(method), err))
            
            raw_errors <- rbind(raw_errors, data.frame(dataset = data,
                                                       x = as.numeric(c), run = k, method = method, error = err))
          }

        }
      }
      errors[[c]]$lasso <- lasso
      errors[[c]]$relabel <- relabel
      errors[[c]]$rfae <- rfae
    }
    results[[i]] <- errors
    
  }
  
  row_names <- c("LASSO", "Relabel", "RFAE")
  methods <- c("lasso", "relabel", "rfae")
  plot_data <- data.frame(Method = character(0), x = numeric(0), mean = numeric(0),     
                          se = numeric(0), dataset = character(0))
  for (i in seq_along(dat)) {
    for (c in compressions) {
      for (j in methods) {
        plot_data <- rbind(plot_data, 
                           data.frame(Method = j, 
                                      x = as.numeric(c), 
                                      mean = mean(results[[i]][[c]][[j]]),
                                      se = sd(results[[i]][[c]][[j]])/sqrt(10),
                                      dataset = dat[[i]]))
      }
    }
  }  
  plot <- ggplot(plot_data, aes(x, mean, color = Method, fill = Method)) + 
    geom_line() + 
    geom_ribbon(aes(ymax = mean + se, ymin = mean - se), 
                alpha = 0.5, color = NA) + 
    scale_color_d3() +
    scale_fill_d3() +
    labs(x = 'Compression Factor', y = 'Performance') +
    theme_bw() + 
    facet_wrap(~ dataset, scales = 'free_y')
  return(list(plot = plot, plot_data = plot_data, raw = raw_errors))
}

decode_plots <- loop(c('student', 'credit'))

df <- as.data.table(decode_plots$plot_data)
raw <- as.data.table(decode_plots$raw)
raw[, error := 1 - error]
setnames(raw, 'error', 'Distortion')
cols <- data.table(
  method = c('RFAE', 'LASSO', 'RELABEL'),
  color = c('#FF7F00', '#1F78B4',  '#33A02C')
)


# Capitalize method, reorder levels
df[, Method := toupper(Method)]
df[, Method := factor(Method, levels = c('RFAE', 'LASSO', 'RELABEL'))]

# Plot
g <- ggplot(df, aes(x, 1 - mean, color = Method, fill = Method)) + 
  geom_line() + 
  geom_ribbon(aes(ymax = 1 - mean + se, ymin = 1 - mean - se), 
              alpha = 0.5, color = NA) + 
  scale_color_manual(values = cols$color) +
  scale_fill_manual(values = cols$color) +
  labs(x = 'Compression Factor', y = 'Distortion') +
  theme_bw() + 
  theme(legend.position = 'bottom') +
  guides(colour = guide_legend(nrow = 1)) +
  facet_wrap(~ dataset, scales = 'free_y', nrow = 1) # Assumes 20 datasets

ggsave('decoder.pdf', g, width=6, height=4)
