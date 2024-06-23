import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
from matplotlib.widgets import Slider
from matplotlib.patches import Rectangle
import ast
import os

def plot_nodules_on_ct_scan(ct_scan_volume, annotations_df):
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    # Plot nodules on the current slice
    nodules_coords = annotations_df['centers'].apply(ast.literal_eval)
    for nodule_coords in nodules_coords:
        for z, y, x in nodule_coords:
            print("Nodule on: ",z)
    
    if(len(nodules_coords)) == 0:
        print("No Nodule")
     

    # Initialize slice index
    slice_index = 0
    
    # Plot the initial slice
    img = ax.imshow(ct_scan_volume[slice_index], cmap='gray', vmin=-30, vmax=255)
    ax.axis('off')
    ax.set_title(f"Slice {slice_index}")
    
    # Create a slider for selecting slices
    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
    slider = Slider(ax_slider, 'Slice', 0, len(ct_scan_volume) - 1, valinit=slice_index, valstep=1)
    
    # Define function to update plot based on slider value
    def update(val):
        nonlocal slice_index
        slice_index = int(slider.val)
        img.set_data(ct_scan_volume[slice_index])
        ax.set_title(f"Slice {slice_index}")
        
        # Clear previous nodules
        for patch in ax.patches:
            patch.remove()
        
        # Plot nodules on the current slice
        nodules_coords = annotations_df['centers'].apply(ast.literal_eval)
        for nodule_coords in nodules_coords:
            for z, y, x in nodule_coords:
                if z == slice_index:
                    rect = Rectangle((x-4, y-4), 8, 8, linewidth=1, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)
        
        ax.figure.canvas.draw_idle()
    
    # Attach the update function to the slider
    slider.on_changed(update)
    
    plt.show()




# Load preprocessed CT scan volume
Preprocess = True
# ct_scan_volume_path = 'C:\\Users\\kutay\\OneDrive\\Masaüstü\\Luna\\Preprocessed\\preprocessed\\positives\\1.3.6.1.4.1.14519.5.2.1.6279.6001.107351566259572521472765997306.npy'
ct_scan_volume_path = 'C:\\Users\\kutay\\OneDrive\\Masaüstü\\Luna\\Preprocessed\\preprocessed\\positives\\1.3.6.1.4.1.14519.5.2.1.6279.6001.122914038048856168343065566972.npy'
# ct_scan_volume_path = "C:\\Users\\kutay\\OneDrive\\Masaüstü\\Luna\\Preprocessed\\augmented\\positives\\1.3.6.1.4.1.14519.5.2.1.6279.6001.107351566259572521472765997306_5_0.npy"
# ct_scan_volume_path = "C:\\Users\\kutay\\OneDrive\\Masaüstü\\Luna\\Preprocessed\\augmented\\positives\\1.3.6.1.4.1.14519.5.2.1.6279.6001.107351566259572521472765997306_1_1_1.npy"

ct_scan_volume = np.load(ct_scan_volume_path)

base_name = os.path.basename(ct_scan_volume_path)  # Get the base name from the file path
name, extension = os.path.splitext(base_name)  # Split the base name into name and extension
name = base_name.replace(".npy", "")



if(not Preprocess):
    underscore_index = name.find("_") + 1
    sub_index = name[underscore_index:]
    name = name[0:underscore_index-1]
    annotations_path = 'C:\\Users\\kutay\\OneDrive\\Masaüstü\\Luna\\Preprocessed\\augmented_meta.csv'
else:
    annotations_path = 'C:\\Users\\kutay\\OneDrive\\Masaüstü\\Luna\\Preprocessed\\preprocessed_meta.csv'

print(name)

# Load preprocessed annotations DataFrame
preprocessed_annotation = pd.read_csv(annotations_path)
filtered_annotations = preprocessed_annotation[preprocessed_annotation['seriesuid'] == name]

if(not Preprocess):
    filtered_annotations = filtered_annotations[filtered_annotations['sub_index'] == sub_index]



# Now you can use the provided function to plot nodules on the CT scan volume
plot_nodules_on_ct_scan(ct_scan_volume, filtered_annotations)

