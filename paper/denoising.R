# Load libraries
library(data.table)
library(ranger)
library(arf)
library(Matrix)
library(RSpectra)
library(ggplot2)
library(ggsci)
library(cowplot)
library(RANN)
library(batchelor)
library(scRNAseq)
library(scuttle)
library(scran)
library(scater)
library(Rtsne)
library(doMC)
registerDoMC(12)

# Load internal functions
source('encode.R')
source('utils.R')
source('decode_knn.R')



# Import scRNA-seq data
sce1 <- ZeiselBrainData()
sce2 <- TasicBrainData()

sce1 <- addPerCellQC(sce1, subsets=list(Mito=grep("mt-", rownames(sce1))))
qc1 <- quickPerCellQC(colData(sce1), sub.fields="subsets_Mito_percent")
sce1 <- sce1[,!qc1$discard]

sce2 <- addPerCellQC(sce2, subsets=list(Mito=grep("mt_", rownames(sce2))))
qc2 <- quickPerCellQC(colData(sce2), sub.fields="subsets_Mito_percent")
sce2 <- sce2[,!qc2$discard]
universe <- intersect(rownames(sce1), rownames(sce2))
sce1 <- sce1[universe,]
sce2 <- sce2[universe,]
out <- multiBatchNorm(sce1, sce2)
sce1 <- out[[1]]
sce2 <- out[[2]]

dec1 <- modelGeneVar(sce1)
dec2 <- modelGeneVar(sce2)
combined.dec <- combineVar(dec1, dec2)
chosen.hvgs <- getTopHVGs(combined.dec, n=5000)

combined <- correctExperiments(A=sce1, B=sce2, PARAM=NoCorrectParam())

set.seed(100)
combined <- runPCA(combined, subset_row=chosen.hvgs)
combined <- runTSNE(combined, dimred="PCA")
# plotTSNE(combined, colour_by="batch")

################################################################################

# Work in log space
x1_logcounts <- assay(sce1, 'logcounts')[chosen.hvgs, ]
x2_logcounts <- assay(sce2, 'logcounts')[chosen.hvgs, ]
tsne <- reducedDims(combined)$TSNE
df <- data.table(
  'tSNE1' = tsne[, 1], 'tSNE2' = tsne[, 2],
  'Batch' = c(rep('A', ncol(x1_logcounts)), rep('B', ncol(x2_logcounts)))
)
g <- ggplot(df, aes(tSNE1, tSNE2, color = Batch)) + 
  geom_hline(yintercept = 0L, color = 'grey') +
  geom_vline(xintercept = 0L, color = 'grey') +
  geom_point(size = 1.5, alpha = 0.5) + 
  scale_color_npg() + 
  theme_bw()


# Train RFAE
x1 <- as.data.frame(as.matrix(t(x1_logcounts)))
x2 <- as.data.frame(as.matrix(t(x2_logcounts)))
colnames(x1) <- paste0('g', 1:ncol(x1))
colnames(x2) <- paste0('g', 1:ncol(x2))
y <- rbinom(nrow(x1), 1, 0.5)
rf <- ranger(x = x1, y = y, num.trees = 1000, classification = TRUE,
             splitrule = 'extratrees', num.random.splits = 1) # Completely RF
emap <- encode(rf, x1, k = 64)
z_new <- predict.encode(emap, rf, x2)
x_tilde <- train_decoder(rf, emap)
res <- decode_knn(rf, emap, z_new, x_tilde = x_tilde, k = 100) 
x_new <- res
x_new <- as.matrix(t(x_new)) 
rownames(x_new) <- chosen.hvgs
colnames(x_new) <- colnames(sce2)

# Prep data, run tSNE
sce_new <- SingleCellExperiment(assays = list(logcounts = x_new))
combo <- correctExperiments(A = sce1[chosen.hvgs, ], B = sce_new)
combo <- runPCA(combo)
combo <- runTSNE(combo, dimred="PCA")
tsne2 <- reducedDims(combo)$TSNE

# Shape results
pca1 <- reducedDims(combined)$PCA
pca2 <- reducedDims(combo)$PCA
tsne1 <- reducedDims(combined)$TSNE
tsne2 <- reducedDims(combo)$TSNE
batch <- c(rep('Zeisel et al., 2015', ncol(sce1)), 
           rep('Tasic et al., 2016', ncol(sce_new)))

df_pca <- rbind(
  data.table('PC1' = pca1[, 1], 'PC2' = pca1[, 2], 
             'Study' = batch, 'Data' = 'Original'),
  data.table('PC1' = pca2[, 1], 'PC2' = pca2[, 2], 
             'Study' = batch, 'Data' = 'Batch Corrected')
)
df_tsne <- rbind(
  data.table('tSNE1' = tsne1[, 1], 'tSNE2' = tsne1[, 2], 
             'Study' = batch, 'Data' = 'Original'),
  data.table('tSNE1' = tsne2[, 1], 'tSNE2' = tsne2[, 2], 
             'Study' = batch, 'Data' = 'Batch Corrected')
)
df_pca[, Data := factor(Data, levels = c('Original', 'Batch Corrected'))]
df_tsne[, Data := factor(Data, levels = c('Original', 'Batch Corrected'))]

# Plots
g1 <- ggplot(df_tsne, aes(tSNE1, tSNE2, color = Study)) + 
  geom_hline(yintercept = 0L, color = 'grey') +
  geom_vline(xintercept = 0L, color = 'grey') +
  geom_point(size = 1.5, alpha = 0.3) + 
  scale_color_npg() + 
  theme_bw() + 
  theme(axis.title = element_text(size = 15),
        strip.text.x = element_text(size = 15),
        strip.text.y = element_text(size = 15)) + 
  facet_wrap(~ Data, scales = 'free')

g2 <- ggplot(df_pca, aes(PC1, PC2, color = Study)) + 
  geom_hline(yintercept = 0L, color = 'grey') +
  geom_vline(xintercept = 0L, color = 'grey') +
  geom_point(size = 1.5, alpha = 0.3) + 
  scale_color_npg() + 
  theme_bw() + 
  theme(axis.title = element_text(size = 15),
        strip.text.x = element_text(size = 15),
        strip.text.y = element_text(size = 15)) + 
  facet_wrap(~ Data, scales = 'free')

# combine both plot using plot_grid()
g <- plot_grid(g1 + theme(legend.position = 'none'), 
               g2 + theme(legend.position = 'none'), 
               labels = c('A', 'B'), nrow = 2)

# extract legend from plot1
legend <- get_legend(
  g1 + theme(legend.title = element_text(size = 15),
             legend.text = element_text(size = 15),
             legend.position = 'bottom')
)

# Combine combined plot and legend using plot_grid()
plot_grid(g, legend, nrow = 2, relwidths = c(4, 0.5))

ggsave('./scRNAseq/figure.pdf', height = 10, width = 10)





