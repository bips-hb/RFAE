This folder contains scripts to rerun and reproduce the results found in the compression/distoriton benchmark, or figure 4.
Some specific running instructions:
- All other methods were run on a HPC unit. We include the python/R file required to run the models on HPC, but don't upload the run script to maintain anonymity. Alternatively, you can run it locally.
- Folders and subfolders need to be created for the reconstructed data to be placed, of template METHOD_data/DATASET/ (e.g., rfae_data/adult)
- Files are then structured as LATENT_RATE_runRUN.csv (e.g., 0.1_run1.csv)
- The data to be used as input for the benchmark is the original data file, accompanied with a bootstrap index file of randomly generated bootstraps. We include this file to load the bootstrapped data instead of writing multiple versions of each dataset.
- TVAE uses the CTGAN package, but to run it, you need to replace the tvae.py script into ctgan/synthesizers/tvae.py and install the package that way.
- After finishing, you can run the benchmark_plots.R
- Or, just run becnhmark_plots with raw and plot_data