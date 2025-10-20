import pandas as pd
import torch
import os
import glob
import numpy as np
import re
import matplotlib.pyplot as plt

plt.rcParams['usetex'] = True
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{amssymb}'


def natural_key(string_):
    basename = os.path.basename(string_)
    float_re = re.compile(r'([-+]?\d*\.\d+|[-+]?\d+)')
    parts = float_re.split(basename)
    key = []
    for part in parts:
        try:
            key.append(float(part))
        except ValueError:
            key.append(part.lower())
    return key

def format_latex_label(key, val, variant=None):
    # Special hardcoded override
    if key == "d_Z":
        base = r"d_{\mathcal{Z}}"
    elif "_" in key:
        prefix, sub = key.split("_", 1)
        sub = sub.replace("_", r"\_")  # escape all subscripts

        if len(prefix) == 1:
            base = rf"{prefix}_\mathrm{{{sub}}}"
        else:
            base = rf"\mathrm{{{prefix}}}_\mathrm{{{sub}}}"
    else:
        key = key.replace("_", r"\_")
        base = key if len(key) == 1 else rf"\mathrm{{{key}}}"

    label = rf"${base} = {val}"
    if variant:
        label += rf" \ \mathrm{{({variant})}}"
    label += "$"
    return label

def plot_all_reconstructions_per_folder(folder, filename, model_variant=None):
    csv_files = glob.glob(os.path.join(folder, '*.csv'))

    # Filter by model_variant
    if model_variant == "RFAE":
        csv_files = [f for f in csv_files if not os.path.basename(f).startswith("CRFAE_")]
    elif model_variant == "CRFAE":
        csv_files = [f for f in csv_files if not os.path.basename(f).startswith("RFAE_")]

    # --- Sort logic ---
    def extract_sort_info(f):
        name = os.path.splitext(os.path.basename(f))[0]

        if name.startswith(("RFAE_", "CRFAE_")):
            variant, suffix = name.split("_", 1)
            if "_" in suffix:
                key, val = suffix.rsplit("_", 1)
                return (key, float(val) if val.replace('.', '', 1).isdigit() else val, variant)
            else:
                return (suffix, "", variant)
        elif name == "RFAE":
            return ("aaa_rfae", "", "zzz")       # Top priority
        elif name == "CRFAE":
            return ("aab_crfae", "", "zzz")      # Second
        elif name == "ConvAE":
            return ("aac_convae", "", "zzz")     # Third
        elif name == "original":
            return ("aad_original", "", "zzz")   # Last
        else:
            return (name, "", "")

    if model_variant is None:
        csv_files = sorted(csv_files, key=extract_sort_info)
    else:
        def sort_key(f):
            name = os.path.basename(f).lower()
            if 'original' in name:
                return (4, natural_key(f))
            elif 'convae' in name:
                return (3, natural_key(f))
            elif 'crfae' in name:
                return (2, natural_key(f))
            elif 'rfae' in name:
                return (1, natural_key(f))
            else:
                return (0, natural_key(f))

        csv_files = sorted(csv_files, key=sort_key)

    # --- Plotting ---
    num_files = len(csv_files)
    num_images_per_row = 10

    fig, axs = plt.subplots(num_files, num_images_per_row, figsize=(num_images_per_row, num_files * 1.1))
    plt.subplots_adjust(wspace=0, hspace=0)

    for i, file in enumerate(csv_files):
        data = pd.read_csv(file).values
        images = torch.tensor(data, dtype=torch.float32).reshape(-1, 28, 28)

        if 'ConvAE' not in os.path.basename(file):
            images = images.permute(0, 2, 1)

        for j in range(num_images_per_row):
            ax = axs[i, j]
            ax.imshow(images[j], cmap='gray', vmin=0, vmax=1)
            ax.axis('off')

        # --- Label logic ---
        file_name = os.path.splitext(os.path.basename(file))[0]

        if model_variant is None:
            if file_name.startswith(("RFAE_", "CRFAE_")):
                variant, suffix = file_name.split("_", 1)
                if "_" in suffix:
                    key, val = suffix.rsplit("_", 1)
                    label = format_latex_label(key, val, variant)
                else:
                    key = suffix
                    label = rf"${{\mathrm{{{key.replace('_', r'\_')}}}}} \ \mathrm{{({variant})}}$"
            else:
                safe_name = file_name.replace("_", r"\_")
                safe_name = safe_name[0].upper() + safe_name[1:]
                label = rf"$\mathrm{{{safe_name}}}$"
        else:
            if '_' in file_name:
                suffix = file_name.split('_')[-1]
                folder_name = os.path.basename(folder)
                label = format_latex_label(folder_name, suffix)
            else:
                # Custom label formatting when model_variant is not None
                if file_name.startswith("RFAE"):
                    label = r"RFAE\ (supervised)"
                elif file_name.startswith("CRFAE"):
                    label = r"RFAE\ (completely random)"
                else:
                    safe_name = file_name.replace("_", r"\_")
                    label = rf"$\mathrm{{{safe_name[0].upper() + safe_name[1:]}}}$"

        axs[i, 0].annotate(
            label,
            xy=(-0.1, 0.5),
            xycoords='axes fraction',
            va='center',
            ha='right',
            fontsize=22,
            annotation_clip=False
        )

    plt.savefig(filename, bbox_inches='tight', pad_inches=0.0)
    plt.close()

folders = [f.path for f in os.scandir('visual_experiments/reconstructions/mnist') if f.is_dir()]

### supervised

for folder in folders:
    # get the name of the folder
    folder_name = os.path.basename(folder)
    # create a filename for the plot
    filename = os.path.join('visual_experiments', 'results', 'RFAE', f'mnist_RFAE_{folder_name}.pdf')
    # call the function
    plot_all_reconstructions_per_folder(folder, filename, model_variant="RFAE")
 
### completely random
    
for folder in folders:
    # get the name of the folder
    folder_name = os.path.basename(folder)
    # create a filename for the plot
    filename = os.path.join('visual_experiments', 'results', 'CRFAE', f'mnist_CRFAE_{folder_name}.pdf')
    # call the function
    plot_all_reconstructions_per_folder(folder, filename, model_variant="CRFAE")

### mixed plots

# for folder in folders:
#     # get the name of the folder
#     folder_name = os.path.basename(folder)
#     # create a filename for the plot
#     filename = os.path.join('visual_experiments', 'results', 'RFAE_vs_CRFAE', f'mnist_comparison_{folder_name}.pdf')
#     # call the function
#     plot_all_reconstructions_per_folder(folder, filename)
    
    
### denoising

# # get the name of the folder
# folder = 'visual_experiments/denoising'
# filename = os.path.join('visual_experiments', 'results', 'Denoising', 'RFAE_denoising.pdf')
# plot_all_reconstructions_per_folder(folder, filename)