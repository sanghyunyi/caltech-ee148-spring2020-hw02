import os
import numpy as np
import json
from PIL import Image
import multiprocessing as mp
from utils import *


def compute_convolution(I, T, stride=None):
    '''
    This function takes an image <I> and a template <T> (both numpy arrays)
    and returns a heatmap where each grid represents the output produced by
    convolution at each location. You can add optional parameters (e.g. stride,
    window_size, padding) to create additional functionality.
    '''
    (n_rows,n_cols,n_channels) = np.shape(I)

    '''
    BEGIN YOUR CODE
    '''

    # this is when template has even number shape

    (T_n_rows,T_n_cols,T_n_channels) = np.shape(T)

    pad_width = ((int((T_n_rows-1)/2), int(T_n_rows/2)), (int((T_n_cols-1)/2), int(T_n_cols/2)), (0, 0))
    padded_image = np.pad(I, pad_width, mode='constant')

    heatmap = np.zeros(np.shape(I))

    for i in range(n_rows):
        for j in range(n_cols):
            patch = padded_image[i:i+T_n_rows, j:j+T_n_cols]
            patch = patch/np.linalg.norm(patch)
            heatmap[i, j] = (patch * T).sum()

    '''
    END YOUR CODE
    '''

    return heatmap

def candidateRedLightHeatmap(image, kernels):
    imageX = image.shape[0]
    imageY = image.shape[1]

    pool = mp.Pool(mp.cpu_count())
    args = [(image, kernel, None) for kernel in kernels]
    convolved_list = pool.starmap(compute_convolution, args)
    pool.close()
    pool.join()

    out = np.zeros((imageX, imageY, 3, len(kernels)))
    for i, kernel in enumerate(kernels):
        out[:,:,:,i] = convolved_list[i]

    single_channel = out.mean(axis=(2)) # Average across RGB
    max_conv = single_channel.max(axis=(-1)) # Max across kernels
    max_kernel_idxs = single_channel.argmax(axis=(-1))
    heatmap = max_conv

    return heatmap, max_kernel_idxs

def predict_boxes(heatmap, max_kernel_idxs, kernels):
    '''
    This function takes heatmap and returns the bounding boxes and associated
    confidence scores.
    '''

    output = []

    '''
    BEGIN YOUR CODE
    '''
    thresholded = heatmap >= 0.95
    x_indices, y_indices = np.where(thresholded == 1.)
    idxs = list(zip(x_indices, y_indices))

    imageX = heatmap.shape[0]
    imageY = heatmap.shape[1]

    for idx in idxs:
        x = idx[0]
        y = idx[1]
        kernel_idx = max_kernel_idxs[x, y]
        kernel = kernels[kernel_idx]
        kernelX = kernel.shape[0]
        kernelY = kernel.shape[1]
        tl_row = int(max(x - kernelX/2, 0))
        tl_col = int(max(y - kernelY/2, 0))
        br_row = int(min(x - kernelX/2 + kernelY, imageX))
        br_col = int(min(y + kernelY/2, imageY))
        score = max(min(heatmap[x, y], 1.0), 0.)
        output.append([tl_row, tl_col, br_row, br_col, score])

    '''
    END YOUR CODE
    '''

    return output

# Build kernels or templates
kernels = buildKernels(10)
print("kernels loaded")

def detect_red_light_mf(I):
    '''
    This function takes a numpy array <I> and returns a list <output>.
    The length of <output> is the number of bounding boxes predicted for <I>.
    Each entry of <output> is a list <[row_TL,col_TL,row_BR,col_BR,score]>.
    The first four entries are four integers specifying a bounding box
    (the row and column index of the top left corner and the row and column
    index of the bottom right corner).
    <score> is a confidence score ranging from 0 to 1.

    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''

    '''
    BEGIN YOUR CODE
    '''

    heatmap, max_kernel_idxs = candidateRedLightHeatmap(I, kernels)
    output = predict_boxes(heatmap, max_kernel_idxs, kernels)

    '''
    END YOUR CODE
    '''

    for i in range(len(output)):
        assert len(output[i]) == 5
        assert (output[i][4] >= 0.0) and (output[i][4] <= 1.0)

    return output


if __name__ == "__main__":
    # Note that you are not allowed to use test data for training.
    # set the path to the downloaded data:
    data_path = '../data/RedLights2011_Medium'

    # load splits:
    split_path = '../data/hw02_splits'
    file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
    file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

    # set a path for saving predictions:
    preds_path = '../data/hw02_preds'
    os.makedirs(preds_path, exist_ok=True) # create directory if needed

    # Set this parameter to True when you're done with algorithm development:
    done_tweaking = True

    '''
    Make predictions on the training set.
    '''

    preds_train = {}
    for i in range(len(file_names_train)):

        # read image using PIL:
        I = Image.open(os.path.join(data_path,file_names_train[i]))

        # convert to numpy array:
        I = np.asarray(I)

        preds_train[file_names_train[i]] = detect_red_light_mf(I)

    # save preds (overwrites any previous predictions!)
    with open(os.path.join(preds_path,'preds_train.json'),'w') as f:
        json.dump(preds_train,f)

    if done_tweaking:
        '''
        Make predictions on the test set.
        '''
        preds_test = {}
        for i in range(len(file_names_test)):

            # read image using PIL:
            I = Image.open(os.path.join(data_path,file_names_test[i]))

            # convert to numpy array:
            I = np.asarray(I)

            preds_test[file_names_test[i]] = detect_red_light_mf(I)

        # save preds (overwrites any previous predictions!)
        with open(os.path.join(preds_path,'preds_test.json'),'w') as f:
            json.dump(preds_test,f)
