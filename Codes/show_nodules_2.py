import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import ast
import os
from matplotlib.patches import Rectangle

ANNOTATION_EXIST = False

def plot_nodules_on_ct_scans(ct_scan_volumes, annotations_model ):
    fig, axs = plt.subplots(2, 2)
    plt.subplots_adjust(bottom=0.25)

    # Initialize slice index
    slice_index = 0

    # Plot the initial slices
    imgs = []
    for i, ax in enumerate(axs.flat):
        img = ax.imshow(ct_scan_volumes[i][slice_index], cmap='gray', vmin=-30, vmax=255)
        ax.axis('off')
        ax.set_title(f"Image {i+1} - Slice {slice_index}")
        imgs.append(img)

    # Create a slider for selecting slices
    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
    slider = Slider(ax_slider, 'Slice', 0, len(ct_scan_volumes[0]) - 1, valinit=slice_index, valstep=1)

   


    # Define function to update plots based on slider value
    def update(val):
        nonlocal slice_index
        slice_index = int(slider.val)
        for i, ax in enumerate(axs.flat):
            imgs[i].set_data(ct_scan_volumes[i][slice_index])
            ax.set_title(f"Image {i+1} - Slice {slice_index}")
            
            # Clear previous nodules
            for patch in ax.patches:
                patch.remove()

            # Plot nodules on the current slice
            nodules_coords = annotations_model[i]['centers'].apply(ast.literal_eval)
            for nodule_coords in nodules_coords:
                for z, y, x in nodule_coords:
                    if z == slice_index:
                        rect = Rectangle((x-4, y-4), 8, 8, linewidth=1, edgecolor='r', facecolor='none')
                        ax.add_patch(rect)


            # Plot nodules on the current slice MODEL
            nodules_coords = annotations_model[i]['model_centers'].apply(ast.literal_eval)
            for nodule_coords in nodules_coords:
                for z, y, x in nodule_coords:
                    if z == slice_index:
                        rect = Rectangle((x-4, y-4), 8, 8, linewidth=1, edgecolor='b', facecolor='none')
                        ax.add_patch(rect)


        fig.canvas.draw_idle()

    # Attach the update function to the slider
    slider.on_changed(update)

    plt.show()


# Paths to the CT scan volumes and annotations
# ct_scan_volume_paths = [
#     "C:\\Users\\kutay\\OneDrive\\Masaüstü\\Luna\\Preprocessed\\augmented\\positives\\1.3.6.1.4.1.14519.5.2.1.6279.6001.107351566259572521472765997306_0_0_1.npy",
#     "C:\\Users\\kutay\\OneDrive\\Masaüstü\\Luna\\Preprocessed\\augmented\\positives\\1.3.6.1.4.1.14519.5.2.1.6279.6001.107351566259572521472765997306_1_0_1.npy",
#     "C:\\Users\\kutay\\OneDrive\\Masaüstü\\Luna\\Preprocessed\\augmented\\positives\\1.3.6.1.4.1.14519.5.2.1.6279.6001.107351566259572521472765997306_0_1_1.npy",
#     "C:\\Users\\kutay\\OneDrive\\Masaüstü\\Luna\\Preprocessed\\augmented\\positives\\1.3.6.1.4.1.14519.5.2.1.6279.6001.107351566259572521472765997306_1_1_1.npy"
# ]

# ct_scan_volume_paths = [
#     "C:\\Users\\kutay\\OneDrive\\Masaüstü\\Luna\\Preprocessed\\augmented\\positives\\1.3.6.1.4.1.14519.5.2.1.6279.6001.115386642382564804180764325545_0_0_1.npy",
#     "C:\\Users\\kutay\\OneDrive\\Masaüstü\\Luna\\Preprocessed\\augmented\\positives\\1.3.6.1.4.1.14519.5.2.1.6279.6001.115386642382564804180764325545_1_0_1.npy",
#     "C:\\Users\\kutay\\OneDrive\\Masaüstü\\Luna\\Preprocessed\\augmented\\positives\\1.3.6.1.4.1.14519.5.2.1.6279.6001.115386642382564804180764325545_0_1_1.npy",
#     "C:\\Users\\kutay\\OneDrive\\Masaüstü\\Luna\\Preprocessed\\augmented\\positives\\1.3.6.1.4.1.14519.5.2.1.6279.6001.115386642382564804180764325545_1_1_1.npy"
# ]

if(ANNOTATION_EXIST):
    ct_scan_volume_paths = [
    "C:\\Users\\kutay\\OneDrive\\Masaüstü\\Luna\\Preprocessed\\augmented\\positives\\1.3.6.1.4.1.14519.5.2.1.6279.6001.107351566259572521472765997306_0_0_1.npy",
    "C:\\Users\\kutay\\OneDrive\\Masaüstü\\Luna\\Preprocessed\\augmented\\positives\\1.3.6.1.4.1.14519.5.2.1.6279.6001.107351566259572521472765997306_1_0_1.npy",
    "C:\\Users\\kutay\\OneDrive\\Masaüstü\\Luna\\Preprocessed\\augmented\\positives\\1.3.6.1.4.1.14519.5.2.1.6279.6001.107351566259572521472765997306_0_1_1.npy",
    "C:\\Users\\kutay\\OneDrive\\Masaüstü\\Luna\\Preprocessed\\augmented\\positives\\1.3.6.1.4.1.14519.5.2.1.6279.6001.107351566259572521472765997306_1_1_1.npy"
]
else:
    ct_scan_volume_paths = [
        "C:\\Users\\kutay\\OneDrive\\Masaüstü\\Luna\\Preprocessed\\augmented\\unlabeled\\1.3.6.1.4.1.14519.5.2.1.6279.6001.100530488926682752765845212286_0_0_1.npy",
        "C:\\Users\\kutay\\OneDrive\\Masaüstü\\Luna\\Preprocessed\\augmented\\unlabeled\\1.3.6.1.4.1.14519.5.2.1.6279.6001.100530488926682752765845212286_1_0_1.npy",
        "C:\\Users\\kutay\\OneDrive\\Masaüstü\\Luna\\Preprocessed\\augmented\\unlabeled\\1.3.6.1.4.1.14519.5.2.1.6279.6001.100530488926682752765845212286_0_1_1.npy",
        "C:\\Users\\kutay\\OneDrive\\Masaüstü\\Luna\\Preprocessed\\augmented\\unlabeled\\1.3.6.1.4.1.14519.5.2.1.6279.6001.100530488926682752765845212286_1_1_1.npy"
    ]



# Load the CT scan volumes
ct_scan_volumes = [np.load(path) for path in ct_scan_volume_paths]

# Extract the base names and sub_indices
base_names = [os.path.basename(path) for path in ct_scan_volume_paths]
names_sub_indices = [(name.split('_')[0], name.split('_')[1:]) for name in base_names]

print(names_sub_indices)

# Path to the annotations file
annotations_path = 'C:\\Users\\kutay\\OneDrive\\Masaüstü\\Luna\\Preprocessed\\augmented_meta.csv'
model_annotations_path = 'C:\\Users\\kutay\\OneDrive\\Masaüstü\\Luna\\Preprocessed\\model_annotations.csv'

if(ANNOTATION_EXIST):
    model_annotations_path = 'C:\\Users\\kutay\\OneDrive\\Masaüstü\\Luna\\Preprocessed\\model_annotations.csv'
else:
    model_annotations_path = 'C:\\Users\\kutay\\OneDrive\\Masaüstü\\Luna\\Preprocessed\\model_annotations_unlabeled.csv'


# preprocessed_annotations = pd.read_csv(annotations_path)
model_annotations = pd.read_csv(model_annotations_path)



# Filter annotations for each CT scan volume
filtered_annotations = []
filtered_model_annotations = []
for name, sub_index in names_sub_indices:
    sub = '_'.join(sub_index).replace(".npy", "")

    annotations = model_annotations[
        (model_annotations['seriesuid'] == name) &
        (model_annotations['sub_index'] == sub)
    ]
    filtered_model_annotations.append(annotations)

    nodules_coords = annotations['centers'].apply(ast.literal_eval)
    for nodule_coords in nodules_coords:
        for z, y, x in nodule_coords:
            print("Org Nodule On: ",z)
    
    nodules_coords = annotations['model_centers'].apply(ast.literal_eval)
    for nodule_coords in nodules_coords:
        for z, y, x in nodule_coords:
            print("Pred Nodule On: ",z)



# Plot the nodules on the CT scan volumes
plot_nodules_on_ct_scans(ct_scan_volumes, filtered_model_annotations)