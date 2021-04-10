import os
import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from run_predictions import *

data_path = '../data/RedLights2011_Medium'
preds_path = '../data/hw02_preds'
split_path = '../data/hw02_splits'
figures_path = '../data/hw02_figures'
gts_path = '../data/hw02_annotations'

file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

with open(os.path.join(preds_path,'preds_test.json'),'r') as f:
    preds_test = json.load(f)

with open(os.path.join(gts_path, 'annotations_test.json'),'r') as f:
    gts_test = json.load(f)


for file_name in file_names_test:
    I = Image.open(os.path.join(data_path,file_name))
    fig, ax = plt.subplots()
    ax.imshow(I)
    ax.set_axis_off()

    pred_boxes = preds_test[file_name]
    for box in pred_boxes:
        tl_row, tl_col, br_row, br_col = box[0], box[1], box[2], box[3]
        rectangle = patches.Rectangle((tl_col, tl_row), br_col-tl_col, br_row-tl_row, linewidth=2, edgecolor='y', facecolor='none')
        ax.add_patch(rectangle)

    gts_boxes = gts_test[file_name]
    for box in gts_boxes:
        tl_row, tl_col, br_row, br_col = box[0], box[1], box[2], box[3]
        rectangle = patches.Rectangle((tl_col, tl_row), br_col-tl_col, br_row-tl_row, linewidth=2, edgecolor='g', facecolor='none')
        ax.add_patch(rectangle)

    plt.savefig(os.path.join(figures_path, file_name), bbox_inches='tight', pad_inches=0)


kernels = buildKernels(10)

heatmap_file_names = ["RL-049.jpg"]

for file_name in heatmap_file_names:
    I = Image.open(os.path.join(data_path,file_name))
    fig, ax = plt.subplots()
    #ax.imshow(I)
    ax.set_axis_off()

    I = np.asarray(I)
    heatmap, _ = candidateRedLightHeatmap(I, kernels)
    img = ax.imshow(heatmap, cmap='binary', interpolation='nearest', alpha=0.7)
    fig.colorbar(img)

    plt.savefig(os.path.join(figures_path,"heatmap_" + file_name), bbox_inches='tight', pad_inches=0)

