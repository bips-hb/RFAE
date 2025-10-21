
# Autoencoding Random Forests #
This repository contains all experiments and code used in the Autoencoding Random Forests paper (<https://arxiv.org/abs/2505.21441>), a NeurIPS 2025 poster. From this repository, you can either install the Autoencoding Random Forests model (RFAE) as a package, or rerun the scripts we used to get our experimental results. Since we did not compile RFAE into a package during our experiments, you do not need to build RFAE to rerun these scripts.

The package can be installed by running:
```
devtools::install_github("bips-hb/RFAE")
```
The `paper` folder contains the scripts to rerun and reproduce the experiments performed and report in the paper. 

The `compression_benchmark` subfolder contains scripts and results for the compression/distortion benchmark, or table 1. Some specific running instructions:

- All other methods were run on a HPC unit. We include the python/R file required to run the models on HPC, but don't upload the run script to maintain anonymity. Alternatively, you can run it locally.
- Folders and subfolders need to be created for the reconstructed data to be placed, of template METHOD_data/DATASET/ (e.g., `rfae_data/adult`)
- Files are then structured as LATENT_RATE_runRUN.csv (e.g., `0.1_run1.csv`)
- The data to be used as input for the benchmark is the original data file, accompanied with a bootstrap index file of randomly generated bootstraps. We include this file to load the bootstrapped data instead of writing multiple versions of each dataset.
- TVAE uses the CTGAN package, but to run it, you need to replace the tvae.py script into ctgan/synthesizers/tvae.py and install the package that way.
- After finishing, you can run `benchmark_plots.R`.

Alternatively, you can just run `benchmark_plots` with `raw.csv` and `plot_data.csv`, which are the original results from this experiment.

The `visual_experiments` subfolder contains scripts and results for the MNIST experiments in figures 3, 7 and 8.

The `decoder_sandbox` subfolder contains miscellaneous files relating to the development of RFAE.

The `R` and `cpp` files not in any subfolder are all the scripts that were used in the experiments and benchmarks. Please do not move these if you want to rerun experiments, as some experiments may depend on these scripts.

