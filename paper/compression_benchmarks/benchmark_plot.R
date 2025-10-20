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


source("utils.R")
source("errors.R")

# Loop to read datasets, collect errors 
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
    full <- fread(paste0('original_data/full/', data, '.csv'
    ),header=TRUE)
    bootstraps <- as.matrix(fread(paste0('original_data/full/bootstrap_', data, '.csv')))
    
    errors <- list()
    compressions <- as.character(seq(0.1, 1, 0.1))
    for (c in compressions) {
  
      ae <- vae <- tvae <- ttvae <- rfae <- c()
      for (k in seq(10)) {
    
        bootstrap = bootstraps[, k]
        trn_og <- full[bootstrap, ]
        setDT(trn_og)
        trn_obj <- prep_x(trn_og, default = 1)
        tst <- full[setdiff(seq_len(nrow(full)), bootstrap)]
        tst <- prep_x(tst, trn_obj[[2]], trn_obj[[3]])[[1]]
        setDT(tst)
        for (method in c('ae', 'vae', 'tvae', 'ttvae', 'rfae')) {
          out <- fread(paste0(method, '_data/', data, '/', c, '_run', k, '.csv'))
          if (method %in% c('rfae')) {
            setnames(out, gsub("\\.", "-", colnames(out)))
          }
          err <- reconstruction_error(out, tst)$ovr_error
          assign(method, append(get(method), err))
          
          raw_errors <- rbind(raw_errors, data.frame(dataset = data,
            x = as.numeric(c), run = k, method = method, error = err))
        }
      }
      errors[[c]]$ae <- ae
      errors[[c]]$vae <- vae
      errors[[c]]$tvae <- tvae
      errors[[c]]$ttvae <- ttvae
      errors[[c]]$rfae <- rfae
    }
    results[[i]] <- errors
    
  }

  row_names <- c("AE","VAE", "TVAE", "TTVAE", "rfae")
  methods <- c("ae", "vae", "tvae", "ttvae", "rfae")
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

plots <- loop(c('abalone', 'adult', 'banknote', 'bc', 'car', 'churn',
                'credit', 'diabetes', 'dry_bean', 
                'forestfires', 'hd', 'king', 
                'marketing', 'mushroom', 'obesity', 
                'plpn', 'spambase', 'student', 
                'telco', 'wq'))

fwrite(plots$plot_data, 'plot_data.csv')
fwrite(plots$raw, 'raw.csv')
df <- fread('plot_data.csv')
raw <- fread('raw.csv')
raw[, error := 1 - error]
setnames(raw, 'error', 'Distortion')
cols <- data.table(
  method = c('RFAE', 'TVAE', 'TTVAE', 'AE', 'VAE'),
  color = c('#FF7F00', '#A6CEE3', '#1F78B4', '#B2DF8A', '#33A02C')
)


# Capitalize method, reorder levels
df[, Method := toupper(Method)]
df[, Method := factor(Method, levels = c('RFAE', 'TVAE', 'TTVAE', 'AE', 'VAE'))]

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
  facet_wrap(~ dataset, scales = 'free_y', nrow = 4) # Assumes 20 datasets

ggsave('reconstruction.pdf', width=10, height=7)


# Generate a table of results and rankings
summary_dt <- raw[, .(
  mean_distortion = mean(Distortion),
  sd_distortion = sd(Distortion) / 10  # Standard error of the mean
), by = .(dataset, method)]

summary_dt[, formatted := sprintf("%.3f (%.3f)", mean_distortion, sd_distortion)]
summary_dt[, is_min := mean_distortion == min(mean_distortion), by = dataset]
summary_dt[is_min == TRUE, formatted := paste0("**", formatted, "**")]
wide_table <- dcast(summary_dt, dataset ~ method, value.var = "formatted")
rank_dt <- summary_dt[, .(dataset, method, rank = rank(mean_distortion)), by = dataset]
avg_rank <- rank_dt[, .(Average_Rank = mean(rank)), by = method]
avg_rank[, formatted := sprintf("%.2f", Average_Rank)]
avg_rank_wide <- dcast(avg_rank, . ~ method, value.var = "formatted")[, . := NULL]

wide_table_final <- bind_rows(
  wide_table,
  c(dataset = "average rank", as.list(avg_rank_wide[1]))
)
