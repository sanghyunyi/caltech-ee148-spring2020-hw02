import os
from PIL import Image
import numpy as np
import json
from matplotlib import pyplot as plt

def augmentedKernels(kernels):
    scale_list = [1.5, 2., 3.0]
    kernels_out = []
    for kernel in kernels:
        kernels_out.append(kernel)
        kernel = Image.fromarray(kernel)
        w, h = kernel.size
        for scale in scale_list:
            kernel_new = kernel.resize((int(scale * w), int(scale * h)))
            kernel_new = np.asarray(kernel_new)
            kernels_out.append(kernel_new)
    return kernels_out

def buildKernels(n_kernel):
    json_path = './my_annotation.json'
    data_path = '../data/RedLights2011_Medium'
    split_path = '../data/hw02_splits'
    figure_path = '../data/hw02_figures'

    np.random.seed(0)
    file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
    np.random.shuffle(file_names_train)

    annotation = json.load(open(json_path,'r'))

    kernels = []

    for file_name in file_names_train:
        I = Image.open(os.path.join(data_path,file_name))
        I = np.asarray(I)

        boxes = annotation[file_name]

        for j in range(len(boxes)):
            tl_row, tl_col, br_row, br_col = boxes[j]
            K = I[tl_row: br_row, tl_col: br_col]
            kernels.append(K)

    kernels = kernels[:n_kernel]
    kernels = augmentedKernels(kernels)
    for i, k in enumerate(kernels):
        fig, ax = plt.subplots()
        ax.imshow(k, interpolation='nearest')
        ax.set_axis_off()
        plt.savefig(os.path.join(figure_path, "template_"+str(i)+".png"), bbox_inches='tight', pad_inches=0)
    kernels = [k/np.linalg.norm(k) for k in kernels]

    return kernels
