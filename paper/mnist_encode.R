# Load libraries
library(dslabs)
library(data.table)
library(Matrix)
library(RSpectra)
library(ranger)
library(ggplot2)
library(ggsci)
library(RANN)
library(doMC)
registerDoMC(12)

# Load internal functions
source('encode.R')
source('utils.R')
source('decode_knn.R')

# Import MNIST
mnist <- read_mnist(getwd())
x_trn <- mnist$train$images
x_trn <- as.data.frame(matrix(as.numeric(x_trn), nrow = nrow(x_trn)))
y_trn <- as.factor(mnist$train$labels)
x_tst <- mnist$test$images
x_tst <- as.data.frame(matrix(as.numeric(x_tst), nrow = nrow(x_tst)))
y_tst <- as.factor(mnist$test$labels)

# Now just 3/8, 4/9
keep <- y_trn %in% c('3', '4', '8', '9')
x_trn <- x_trn[keep, ]
y_trn <- y_trn[keep]
y_trn <- factor(y_trn, levels = c('3', '4', '8', '9'))

# Set seed
set.seed(42)
n_trn <- sum(keep)
trn_idx <- sample(n_trn, 5000)
b <- 100

train_loop <- function(depth) {
  
  rf <- ranger(x = x_trn, y = y_trn, classification = TRUE, 
               max.depth = depth, num.threads = 12, num.trees = b)
  emap <- encode(rf, x_trn[trn_idx, ], k = 2)
  df <- as.data.table(emap$Z)
  df[, Label := y_trn[trn_idx]]
  df[, Depth := paste('tree depth =', depth)]
  return(df)
  
}
depths <- c(1, 2, 4, 8, 16)
df <- foreach(dd = depths, .combine = rbind) %do% train_loop(dd)
setnames(df, 'V1', 'KPC1')
setnames(df, 'V2', 'KPC2')
df[, Depth := factor(Depth, levels = paste('tree depth =', depths))]
fwrite(df, './ignore/mnist_embeddings.csv')

g <- ggplot(df, aes(KPC1, KPC2, color = Label)) + 
  geom_hline(yintercept = 0L, color = 'grey') +
  geom_vline(xintercept = 0L, color = 'grey') +
  geom_point(size = 1.5, alpha = 0.4) + 
  scale_color_d3() +
  theme_bw() + 
  theme(axis.title = element_text(size = 24),
        strip.text.x = element_text(size = 24),
        strip.text.y = element_text(size = 24),
        legend.title = element_text(size = 24),
        legend.text = element_text(size = 24)) + 
  facet_wrap(~ Depth, nrow = 1)


ggsave('~/Downloads/depth_mnist.pdf', height = 5, width = 30)