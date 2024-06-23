
# some directory for output the results:
OUTPUT_PATH = 'C:\\Users\\kutay\\OneDrive\\Masaüstü\\Luna\\Preprocessed'
RESOURCES_PATH = 'C:\\Users\\kutay\\OneDrive\\Masaüstü\\Luna\\Dataset'

PADDING_FOR_LOCALIZATION = 10
BLOCK_SIZE = 128
COORDS_CUBE_SIZE = 32
TARGET_SHAPE = (COORDS_CUBE_SIZE, COORDS_CUBE_SIZE, COORDS_CUBE_SIZE, 3, 5)
COORDS_SHAPE = (3, COORDS_CUBE_SIZE, COORDS_CUBE_SIZE, COORDS_CUBE_SIZE)
ANCHOR_SIZES = [10, 30, 60]
VAL_PCT = 0.2
TOTAL_EPOCHS = 100
DEFAULT_LR = 0.01
ANNOTATION_EXIST = True


# MODEL

import torch
from torch import nn
from torch.utils.data import Dataset
import pandas as pd
from ast import literal_eval
import numpy as np
import math
import os
import itertools
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
from matplotlib.widgets import Slider
from matplotlib.patches import Rectangle
import time


# LOSS FUNCTION

import torch
from torch import nn

def plot_nodules_on_ct_scan(ct_scan_volume, org_coords, pred_coords):
    ct_scan_volume = ct_scan_volume.squeeze().squeeze().numpy()
    print("coords: ",org_coords)
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    # Plot nodules on the current slice
    for nodule_coords in org_coords:
        z = nodule_coords[0]
        print("Nodule on Org: ",z)

    for nodule_coords in pred_coords:
        z = nodule_coords[0]
        print("Nodule on Pred: ",z)
    
    # Initialize slice index
    slice_index = 70
    
    # Plot the initial slice
    img = ax.imshow(ct_scan_volume[slice_index], cmap='gray')
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
        for nodule_coords in org_coords:
            z = nodule_coords[0]
            y = nodule_coords[1]
            x = nodule_coords[2]
            if z == slice_index:
                rect = Rectangle((x-4, y-4), 8, 8, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)

        for nodule_coords in pred_coords:
            z = nodule_coords[0]
            y = nodule_coords[1]
            x = nodule_coords[2]
            if z == slice_index:
                rect = Rectangle((x-4, y-4), 8, 8, linewidth=1, edgecolor='b', facecolor='none')
                ax.add_patch(rect)
        
        ax.figure.canvas.draw_idle()
    
    # Attach the update function to the slider
    slider.on_changed(update)
    
    plt.show()


class PostRes(nn.Module):
    def __init__(self, n_in, n_out, stride=1):
        super(PostRes, self).__init__()
        self.conv1 = nn.Conv3d(n_in, n_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(n_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(n_out, n_out, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(n_out)

        if stride != 1 or n_out != n_in:
            self.shortcut = nn.Sequential(
                nn.Conv3d(n_in, n_out, kernel_size=1, stride=stride),
                nn.BatchNorm3d(n_out))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.preBlock = nn.Sequential(
            nn.Conv3d(1, 24, kernel_size=3, padding=1),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace=True),
            nn.Conv3d(24, 24, kernel_size=3, padding=1),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace=True))
        num_blocks_forw = [2, 2, 3, 3]
        num_blocks_back = [3, 3]
        self.featureNum_forw = [24, 32, 64, 64, 64]
        self.featureNum_back = [128, 64, 64]
        for i in range(len(num_blocks_forw)):
            blocks = []
            for j in range(num_blocks_forw[i]):
                if j == 0:
                    blocks.append(PostRes(self.featureNum_forw[i], self.featureNum_forw[i + 1]))
                else:
                    blocks.append(PostRes(self.featureNum_forw[i + 1], self.featureNum_forw[i + 1]))
            setattr(self, 'forw' + str(i + 1), nn.Sequential(*blocks))

        for i in range(len(num_blocks_back)):
            blocks = []
            for j in range(num_blocks_back[i]):
                if j == 0:
                    if i == 0:
                        addition = 3
                    else:
                        addition = 0
                    blocks.append(PostRes(self.featureNum_back[i + 1] + self.featureNum_forw[i + 2] + addition,
                                          self.featureNum_back[i]))
                else:
                    blocks.append(PostRes(self.featureNum_back[i], self.featureNum_back[i]))
            setattr(self, 'back' + str(i + 2), nn.Sequential(*blocks))

        self.maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
        self.maxpool3 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
        self.maxpool4 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
        self.unmaxpool1 = nn.MaxUnpool3d(kernel_size=2, stride=2)
        self.unmaxpool2 = nn.MaxUnpool3d(kernel_size=2, stride=2)

        self.path1 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))
        self.path2 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))
        self.drop = nn.Dropout3d(p=0.5, inplace=False)
        self.output = nn.Sequential(nn.Conv3d(self.featureNum_back[0], 64, kernel_size=1),
                                    nn.ReLU(),
                                    nn.Conv3d(64, 5 * len(ANCHOR_SIZES), kernel_size=1))

    def forward(self, x, coord):
        out = self.preBlock(x)  # 16
        out_pool, indices0 = self.maxpool1(out)
        out1 = self.forw1(out_pool)  # 32
        out1_pool, indices1 = self.maxpool2(out1)
        out2 = self.forw2(out1_pool)  # 64
        out2_pool, indices2 = self.maxpool3(out2)
        out3 = self.forw3(out2_pool)  # 96
        out3_pool, indices3 = self.maxpool4(out3)
        out4 = self.forw4(out3_pool)  # 96
        rev3 = self.path1(out4)
        comb3 = self.back3(torch.cat((rev3, out3), 1))  # 96+96
        rev2 = self.path2(comb3)
        comb2 = self.back2(torch.cat((rev2, out2, coord), 1))  # 64+64
        comb2 = self.drop(comb2)
        out = self.output(comb2)
        size = out.size()
        out = out.view(out.size(0), out.size(1), -1)
        out = out.transpose(1, 2).contiguous().view(size[0], size[2], size[3], size[4], len(ANCHOR_SIZES), 5)
        return out
    
def get_original_centers(target, block_size=BLOCK_SIZE, coord_cube_size=COORDS_CUBE_SIZE, anchor_sizes=ANCHOR_SIZES):

    original_centers = []
    radii = []

    # Iterate through the target array
    for z in range(target.shape[0]):
        for y in range(target.shape[1]):
            for x in range(target.shape[2]):
                for a in range(target.shape[3]):
                    vector = target[z, y, x, a]
                    if vector[0] >= 0.95:  # Check if a nodule is present
                        # Extract the normalized coordinates and radius from the vector
                        normalized_coords = vector[1:4]
                        radius = vector[4]

                        # Reverse normalization to get actual coordinates within the block
                        actual_coords = []
                        window = block_size / coord_cube_size

                        for i, norm_coord in enumerate(normalized_coords):
                            window = block_size / coord_cube_size
                            actual_coord = (norm_coord + 1) * window
                            actual_coords.append(actual_coord)

                        # Calculate original coordinates within the entire 3D scan
                        original_coords = [
                            int(z * window + actual_coords[0]),
                            int(y * window + actual_coords[1]),
                            int(x * window + actual_coords[2])
                        ]

                        original_centers.append(original_coords)
                        radii.append(radius)

    # print(original_centers)
    return original_centers, radii

class LunaDataSet(Dataset):
    def __init__(self, indices: list, meta_dataframe: pd.DataFrame):
        self.indices = indices
        self.meta_dataframe = meta_dataframe

    def __getitem__(self, idx, split=None):
        meta = self.meta_dataframe.iloc[self.indices[idx]]
        centers = literal_eval(meta['centers'])
        radii = literal_eval(meta['radii'])
        lungs_bounding_box = literal_eval(meta['lungs_bounding_box'])
        clazz = int(meta['class'])
        file_series = [meta['seriesuid'],meta['sub_index']]

        if(ANNOTATION_EXIST):
            sub_dir = 'positives' if clazz == 1 else 'negatives'
        else:
            sub_dir = 'unlabeled'

        
        file_path = f'''{OUTPUT_PATH}/augmented/{sub_dir}/{meta['seriesuid']}_{meta['sub_index']}.npy'''
        patch = np.load(file_path)
        target = np.zeros(TARGET_SHAPE)
        if clazz == 1:
            for c in range(len(centers)):
                place = []
                point = []
                windows = []
                for ax in range(len(patch.shape)):
                    window = int(BLOCK_SIZE / TARGET_SHAPE[ax])
                    windows.append(window)

                    val_centers = centers[c][ax] // window
                    if val_centers >= COORDS_CUBE_SIZE:
                      val_centers = COORDS_CUBE_SIZE-1

                    place.append(val_centers)
                    point.append(centers[c][ax] % window)

                if radii[c] <= ANCHOR_SIZES[0] / 2:
                    place.append(0)
                elif radii[c] <= ANCHOR_SIZES[1] / 2:
                    place.append(1)
                else:
                    place.append(2)
                vector = [1]
                for p in range(len(point)):
                    vector.append(point[p] / windows[p] - 1)
                vector.append(radii[c])
                target[tuple(place)] = vector
        else:
            for c in range(len(centers)):
                point = []
                for ax in range(len(patch.shape)):
                    window = int(BLOCK_SIZE / TARGET_SHAPE[ax])
                    point.append(centers[c][ax] % window)

        out_patch = patch[np.newaxis, ]
        coords = self._get_coords(lungs_bounding_box)

        return out_patch, target, coords, file_series

    def __len__(self):
        return len(self.indices)

    @staticmethod
    def _get_coords(bb):
        div_factor = BLOCK_SIZE / COORDS_CUBE_SIZE
        coords = np.ones((3, COORDS_CUBE_SIZE, COORDS_CUBE_SIZE, COORDS_CUBE_SIZE)) * PADDING_FOR_LOCALIZATION

        bb_new = [[], []]
        for i in (0, 1, 2):
            if bb[0][i] < bb[1][i]:
                bb_new[0].append(math.floor(bb[0][i] / div_factor))
                bb_new[1].append(math.ceil(bb[1][i] / div_factor))
            else:
                bb_new[0].append(math.ceil(bb[0][i] / div_factor))
                bb_new[1].append(math.floor(bb[1][i] / div_factor))

        np_bb0 = np.array(bb_new[0], dtype=int)
        np_bb1 = np.array(bb_new[1], dtype=int)
        distances = np.abs(np_bb0 - np_bb1)
        starts = np.minimum(np_bb0, np_bb1)
        ends = np.maximum(np_bb0, np_bb1)

        if (starts > np.array([32, 32, 32])).any() or (ends < np.array([0, 0, 0])).any():
            return coords
        else:
            for i in (0, 1, 2):
                shp = [1, 1, 1]
                shp[i] = -1
                vec = np.arange(-1 * math.ceil(distances[i] / 2), math.floor(distances[i] / 2)).reshape(
                    tuple(shp)) / math.ceil(
                    distances[i] / 2)
                if bb_new[0][i] > bb_new[1][i]:
                    vec = vec * -1
                matrix = np.broadcast_to(vec, tuple(distances))
                a1 = np.maximum(0, starts)
                b1 = np.minimum(ends, COORDS_CUBE_SIZE)
                a2 = np.maximum(-1 * starts, 0)
                b2 = np.minimum(ends, COORDS_CUBE_SIZE) - starts
                coords[i, a1[0]:b1[0], a1[1]:b1[1], a1[2]:b1[2]] = matrix[a2[0]:b2[0], a2[1]:b2[1], a2[2]:b2[2]]
            return coords
        

def get_normal_loc(target, coords, block_size=BLOCK_SIZE, coord_cube_size=COORDS_CUBE_SIZE, anchor_sizes=ANCHOR_SIZES):

    original_centers = []
    radii = []

    for coord in coords:
        z = coord[0]
        y = coord[1]
        x = coord[2]
        a = coord[3]
        vector = target[z, y, x, a]

        if (vector[0] >= 0.9 and vector[4] < 4):  # Check if a nodule is present
            # Extract the normalized coordinates and radius from the vector
            normalized_coords = vector[1:4]
            radius = vector[4]

            # Reverse normalization to get actual coordinates within the block
            actual_coords = []
            window = block_size / coord_cube_size

            for i, norm_coord in enumerate(normalized_coords):
                window = block_size / coord_cube_size
                actual_coord = (norm_coord + 1) * window
                actual_coords.append(actual_coord)

            # Calculate original coordinates within the entire 3D scan
            original_coords = [
                int(z * window + actual_coords[0]),
                int(y * window + actual_coords[1]),
                int(x * window + actual_coords[2])
            ]

            original_centers.append(original_coords)
            radii.append(radius)

    return original_centers

def get_nodule_locations(target, block_size=BLOCK_SIZE, coord_cube_size=COORDS_CUBE_SIZE, anchor_sizes=ANCHOR_SIZES):
    """
    Retrieve the original centers from the target array.

    Args:
        target (numpy.ndarray): The target array containing vectors.
        block_coords (tuple): The starting coordinates of the block within the entire 3D scan.
        block_size (int): The size of each block.
        coord_cube_size (int): The size of the coordinates cube.
        anchor_sizes (list): The anchor sizes.

    Returns:
        original_centers (list): A list of original centers in the entire 3D scan.
        radii (list): A list of radii for the detected nodules.
    """
    original_centers = []
    radii = []

    # Iterate through the target array
    for z in range(target.shape[0]):
        for y in range(target.shape[1]):
            for x in range(target.shape[2]):
                for a in range(target.shape[3]):
                    vector = target[z, y, x, a]
                    if vector[0] >= 0.95:  # Check if a nodule is present
                        # Calculate original coordinates within the entire 3D scan
                        original_coords = [z,y,x,a]
                        original_centers.append(original_coords)

    return original_centers

def load_model(model_path):
    # Load the model
    model = Net()
    if torch.cuda.is_available():
        checkpoint = torch.load(model_path)
    else:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def predict(model, data):
    # Perform predictions
    with torch.no_grad():
        output = model(data)

    # Process the output as needed
    # For example, convert output to numpy array
    prediction = output.cpu().numpy()

    return prediction

def calculate_iou(pred_box, true_box):
    iou_list = []
    for pred, true in zip(pred_box, true_box):
        pz, py, px = pred
        lz, ly, lx = true

        pd = 5
        ld = 5


        # Calculate the difference between the z-coordinates of centers
        dz = abs(pz - lz)

        if dz <= 2:  # Check if the difference is small
            # Calculate half side lengths of predicted and true boxes
            half_pd = pd / 2
            half_ld = ld / 2

            # Calculate distances between centers along y and x axes
            dy = abs(py - ly)
            dx = abs(px - lx)

            # Calculate the intersection of bounding boxes along y and x axes
            intersection_y = half_pd + half_ld - dy
            intersection_x = half_pd + half_ld - dx

            # Calculate the intersection area
            intersection_area = max(0, intersection_y) * max(0, intersection_x)

            # Calculate the area of the predicted and true boxes
            pred_area = pd * pd
            true_area = ld * ld

            # Calculate the union area
            union_area = pred_area + true_area - intersection_area

            # Calculate IoU
            iou = intersection_area / union_area
        else:
            iou = 0.0  # If the difference is large, IoU is 0

        iou_list.append(iou)

    mean_iou = np.mean([float(tensor.detach().cpu().numpy()) if isinstance(tensor, torch.Tensor) else float(tensor) for tensor in iou_list])
    return mean_iou

def main(_model_path, target_area=None):

    # Load the model
    model_path = _model_path

    print("Loading Model: ",model_path)
    model = load_model(model_path)
    print("Model Loaded")
    model.eval()

    # Create Model Meta csv
    model_meta = pd.read_csv(f'{OUTPUT_PATH}/augmented_meta.csv', index_col=0)

    print("mdel: ",model_meta)
    model_meta['model_centers'] = None


    # Load the Data
    meta = pd.read_csv(f'{OUTPUT_PATH}/augmented_meta.csv', index_col=0)

    # if(target_area == "TOP_LEFT"):
    #     meta = meta[meta['sub_index'].str.contains(r'0_0_.+')]
    #     model_meta = model_meta[model_meta['sub_index'].str.contains(r'0_0_.+')]
    # elif(target_area == "TOP_RIGHT"):
    #     meta = meta[meta['sub_index'].str.contains(r'0_1_.+')]
    #     model_meta = model_meta[model_meta['sub_index'].str.contains(r'0_1_.+')]
    # elif(target_area == "BOTTOM_LEFT"):
    #     meta = meta[meta['sub_index'].str.contains(r'1_0_.+')]
    #     model_meta = model_meta[model_meta['sub_index'].str.contains(r'1_0_.+')]
    # elif(target_area == "BOTTOM_RIGHT"):
    #     meta = meta[meta['sub_index'].str.contains(r'1_1_.+')]
    #     model_meta = model_meta[model_meta['sub_index'].str.contains(r'1_1_.+')]


    meta_1 = meta.groupby('seriesuid').indices
    list_of_groups = [{seriesuid: list(indices)} for seriesuid, indices in meta_1.items()]

    train_indices = list(itertools.chain(*[list(i.values())[0] for i in list_of_groups[:]]))
    ltd = LunaDataSet(train_indices, meta)
    train_loader = DataLoader(ltd, batch_size=1, shuffle=False)

    sigmoid = nn.Sigmoid()
    classify_loss = nn.BCELoss()
    regress_loss = nn.SmoothL1Loss()

    print("Validation Data Length: ", len(train_loader))

    total_pos_correct = 0
    totoal_pos = 0
    total_neg_correct = 0
    totoal_neg = 0
    iou_list = []
    for i, (data, target, coord, file_series) in enumerate(train_loader):
            
        print("Detecting: {} : {}".format(i,file_series))
        
        data = data.float()
        target = target.float()
        coord = coord.float()

        pred = model(data, coord)

        output = pred.view(-1, 5)
        labels = target.view(-1, 5)


        org_nodule_loc = get_nodule_locations(target.squeeze().numpy())


        org_norm_loc = get_normal_loc(target.squeeze().numpy(),org_nodule_loc)
        pred_norm_loc = get_normal_loc(pred.squeeze().detach().numpy(),org_nodule_loc)

        row_index = model_meta[(meta['seriesuid'] == file_series[0][0]) & (model_meta['sub_index'] == file_series[1][0])].index[0]
        model_meta.at[row_index, "model_centers"] = pred_norm_loc


        print("org norm 1: ",org_norm_loc)
        print("org norm 2: ",pred_norm_loc)
        iou = 0

        if(len(org_norm_loc) > 0 and len(pred_norm_loc)>0):
            iou = calculate_iou(pred_norm_loc, org_norm_loc)
            iou_list.append(iou)
        else:
            iou = "No Nodule"


        # plot_nodules_on_ct_scan(data,org_norm_loc,pred_norm_loc)


        pos_idcs = labels[:, 0] > 0.5
        pos_idcs = pos_idcs.unsqueeze(1).expand(pos_idcs.size(0), 5)
        pos_output = output[pos_idcs].view(-1, 5)
        pos_labels = labels[pos_idcs].view(-1, 5)

        neg_idcs = labels[:, 0] < 0.5
        neg_output = output[:, 0][neg_idcs]
        neg_labels = labels[:, 0][neg_idcs]
        neg_prob = sigmoid(neg_output)


        if len(pos_output) > 0:

            pos_prob = sigmoid(pos_output[:, 0])
            pz, ph, pw, prd = pos_output[:, 1], pos_output[:, 2], pos_output[:, 3], pos_output[:, 4]
            lz, lh, lw, ld = pos_labels[:, 1], pos_labels[:, 2], pos_labels[:, 3], pos_labels[:, 4]
            regress_losses = [
                regress_loss(pz, lz),
                regress_loss(ph, lh),
                regress_loss(pw, lw),
                regress_loss(prd, ld)]
            regress_losses_data = [loz.item() for loz in regress_losses]
            classify_loss_d = 0.5 * classify_loss(
                pos_prob, pos_labels[:, 0]) + 0.5 * classify_loss(
                neg_prob, neg_labels)
            pos_correct = (pos_prob.data >= 0.5).sum()
            pos_total = len(pos_prob)

            neg_correct = (neg_prob.data < 0.5).sum()
            neg_total = len(neg_prob)

            total_pos_correct += pos_correct
            totoal_pos += pos_total
            total_neg_correct += neg_correct
            totoal_neg += neg_total

            print("pos_correct: ",pos_correct)
            print("pos_total: ",pos_total)

            print("neg_correct: ",neg_correct)
            print("neg_total: ",neg_total)

            print("classify_loss: ", classify_loss_d )
            print("iou: ",iou)

            # Calculate IoU for each positive sample
            print("class: ",pos_output[:, 0])
            print("regress: ",regress_losses_data)        

    print("\n\n Total Result: \n")
    print("pos_correct: ",total_pos_correct)
    print("pos_total: ",totoal_pos)

    print("neg_correct: ",total_neg_correct)
    print("neg_total: ",totoal_neg)

    print("IOU: ",np.mean(iou_list))
    print("Tnr: ", 100 * total_neg_correct/totoal_neg)
    print("Tpr: ", 100 * total_pos_correct/totoal_pos)

    model_meta.to_csv(f'{OUTPUT_PATH}/model_annotations.csv')

def main_unlabeled(_model_path):
    
    # Load the model
    model_path = _model_path

    print("Loading Model Unlabeled: ",model_path)

    model = load_model(model_path)
    print("Model Loaded Unlabeled")
    model.eval()

    # Create Model Meta csv
    model_meta = pd.read_csv(f'{OUTPUT_PATH}/augmented_meta_unlabeled.csv', index_col=0)
    model_meta['model_centers'] = None


    # Load the Data
    meta = pd.read_csv(f'{OUTPUT_PATH}/augmented_meta_unlabeled.csv', index_col=0)
    # meta = meta[meta['sub_index'].str.contains(r'1_1_1')]

    meta_1 = meta.groupby('seriesuid').indices
    list_of_groups = [{seriesuid: list(indices)} for seriesuid, indices in meta_1.items()]

    train_indices = list(itertools.chain(*[list(i.values())[0] for i in list_of_groups[:]]))
    ltd = LunaDataSet(train_indices, meta)
    train_loader = DataLoader(ltd, batch_size=1, shuffle=False)


    for i, (data, target, coord, file_series) in enumerate(train_loader):
            
        print("Detecting Unlabeled: {} : {}".format(i,file_series))
        start_time = time.time()
        
        data = data.float()
        coord = coord.float()

        pred = model(data, coord)

        pred_nodule_loc = get_nodule_locations(pred.squeeze().detach().numpy())
        pred_norm_loc = get_normal_loc(pred.squeeze().detach().numpy(),pred_nodule_loc)

        row_index = model_meta[(meta['seriesuid'] == file_series[0][0]) & (model_meta['sub_index'] == file_series[1][0])].index[0]
        model_meta.at[row_index, "model_centers"] = pred_norm_loc

        end_time = time.time()
        time_spent = end_time - start_time
        print("Time spent:", time_spent, "seconds")


    model_meta.to_csv(f'{OUTPUT_PATH}/model_annotations_unlabeled.csv')



model_path = "C:\\Users\\kutay\\OneDrive\\Masaüstü\\Luna\\Models\\elcap_new_100.ckpt"

if(ANNOTATION_EXIST):
    main(model_path)

else:
    main_unlabeled(model_path)