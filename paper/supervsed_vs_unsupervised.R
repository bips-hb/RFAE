# Load libraries
library(data.table)
library(ranger)
library(arf)
library(Matrix)
library(RSpectra)
library(ggplot2)
library(ggsci)
library(RANN)
library(doMC)
registerDoMC(12)

# Load internal functions
source('encode.R')
source('utils.R')
source('decode_knn.R')

# Denoising dataset
library(cancerclass)
data(GOLUB1)
x <- t(exprs(GOLUB1))
y <- pData(GOLUB1)$class
dat <- as.data.frame(matrix(as.numeric(x), nrow = nrow(x)))
colnames(dat) <- paste0('g', 1:ncol(dat))

# Learn unsupervised embedding
set.seed(123)
arf <- adversarial_rf(dat, num_trees = 200)
emap <- encode(arf, dat, k = 2)
df <- as.data.table(emap$Z)
setnames(df, 'V1', 'KPC1')
setnames(df, 'V2', 'KPC2')
df[, Class := y][, learning := 'Unsupervised']
# g <- ggplot(df, aes(KPC1, KPC2, color = Class)) +
#   geom_hline(yintercept = 0L, color = 'grey') +
#   geom_vline(xintercept = 0L, color = 'grey') +
#   geom_point(size = 2) +
#   scale_color_npg() +
#   theme_bw()

# Learn supervised embedding
dat$y <- factor(y)
rf <- ranger(y ~ ., data = dat, num.trees = 200)
emap <- encode(rf, dat, k = 2)
df2 <- as.data.table(emap$Z)
setnames(df2, 'V1', 'KPC1')
setnames(df2, 'V2', 'KPC2')
df2[, Class := y][, learning := 'Supervised']

# Plot results
df <- rbind(df, df2)

g <- ggplot(df, aes(KPC1, KPC2, color = Class)) +
  geom_hline(yintercept = 0L, color = 'grey') +
  geom_vline(xintercept = 0L, color = 'grey') +
  geom_point(size = 4) +
  scale_color_npg() +
  theme_bw() +
  theme(axis.title = element_text(size = 15),
        strip.text.x = element_text(size = 15),
        strip.text.y = element_text(size = 15),
        legend.title = element_text(size = 15),
        legend.text = element_text(size = 15),
        legend.position = 'bottom') +
  facet_wrap(~ learning, scales = 'free')

ggsave('~/downloads/golub.pdf', height = 7, width = 14)