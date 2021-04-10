import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

def area(box):
    top = box[0]
    left = box[1]
    bottom = box[2]
    right = box[3]

    return float((right - left)*(bottom - top))

def overlapping(box0, box1):
    top0 = box0[0]
    left0 = box0[1]
    bottom0 = box0[2]
    right0 = box0[3]

    top1 = box1[0]
    left1 = box1[1]
    bottom1 = box1[2]
    right1 = box1[3]

    if top0 > bottom1 or top1 > bottom0:
        return False
    elif left0 > right1 or left1 > right0:
        return False
    else:
        return True

def compute_iou(box_1, box_2):
    '''
    This function takes a pair of bounding boxes and returns intersection-over-
    union (IoU) of two bounding boxes.
    '''
    if overlapping(box_1, box_2):
        top1 = box_1[0]
        left1 = box_1[1]
        bottom1 = box_1[2]
        right1 = box_1[3]

        top2 = box_2[0]
        left2 = box_2[1]
        bottom2 = box_2[2]
        right2 = box_2[3]

        top = max(top1, top2)
        left = max(left1, left2)
        bottom = min(bottom1, bottom2)
        right = min(right1, right2)

        intersection = area([top, left, bottom, right])
        box_1_area = area(box_1)
        box_2_area = area(box_2)

        iou = intersection/(box_1_area + box_2_area - intersection)
    else:
        iou = 0.

    assert (iou >= 0) and (iou <= 1.0)

    return iou


def compute_counts(preds, gts, iou_thr=0.5, conf_thr=0.5):
    '''
    This function takes a pair of dictionaries (with our JSON format; see ex.)
    corresponding to predicted and ground truth bounding boxes for a collection
    of images and returns the number of true positives, false positives, and
    false negatives.
    <preds> is a dictionary containing predicted bounding boxes and confidence
    scores for a collection of images.
    <gts> is a dictionary containing ground truth bounding boxes for a
    collection of images.
    '''
    TP = 0
    FP = 0
    FN = 0

    '''
    BEGIN YOUR CODE
    '''
    for pred_file, pred in preds.items():
        gt = gts[pred_file]
        for i in range(len(gt)):
            iou_list = []
            for j in range(len(pred)):
                conf = pred[j][4]
                if conf > conf_thr:
                    iou = compute_iou(pred[j][:4], gt[i])
                    iou_list.append(iou > iou_thr)
            if all(iou == False for iou in iou_list):
                FN += 1

        for j in range(len(pred)):
            conf = pred[j][4]
            iou_list = []
            if conf > conf_thr:
                for i in range(len(gt)):
                    conf = pred[j][4]
                    if conf > conf_thr:
                        iou = compute_iou(pred[j][:4], gt[i])
                        iou_list.append(iou > iou_thr)
                if all(iou == False for iou in iou_list):
                    FP += 1
                else:
                    TP += 1


    '''
    END YOUR CODE
    '''

    return TP, FP, FN

# set a path for predictions and annotations:
preds_path = '../data/hw02_preds'
gts_path = '../data/hw02_annotations'

# load splits:
split_path = '../data/hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# Set this parameter to True when you're done with algorithm development:
done_tweaking = True

'''
Load training data.
'''
with open(os.path.join(preds_path,'preds_train.json'),'r') as f:
    preds_train = json.load(f)

with open(os.path.join(gts_path, 'annotations_train.json'),'r') as f:
    gts_train = json.load(f)

if done_tweaking:

    '''
    Load test data.
    '''

    with open(os.path.join(preds_path,'preds_test.json'),'r') as f:
        preds_test = json.load(f)

    with open(os.path.join(gts_path, 'annotations_test.json'),'r') as f:
        gts_test = json.load(f)


# For a fixed IoU threshold, vary the confidence thresholds.
# The code below gives an example on the training set for one IoU threshold.

figure_path = '../data/hw02_figures'

confidence_thrs = []
for pred_file, preds in preds_train.items():
    confidence_thrs += [pred[4] for pred in preds]

confidence_thrs = np.sort(np.array(confidence_thrs,dtype=float)) # using (ascending) list of confidence scores as thresholds

# Plot training set PR curves
n_gts_train = 0
for gts_file, gts in gts_train.items():
    n_gts_train += len(gts)

plt.figure(0)
for iou_thr in [.25, .5, .75]:
    tp_train = np.zeros(len(confidence_thrs))
    fp_train = np.zeros(len(confidence_thrs))
    fn_train = np.zeros(len(confidence_thrs))
    for i, conf_thr in enumerate(confidence_thrs):
        tp_train[i], fp_train[i], fn_train[i] = compute_counts(preds_train, gts_train, iou_thr=iou_thr, conf_thr=conf_thr)

    precision_train = np.nan_to_num(tp_train/(tp_train + fp_train), nan=1.)
    recall_train = 1 - fn_train/n_gts_train
    plt.plot(recall_train, precision_train, label="iou_thr="+str(iou_thr))
    print(iou_thr, metrics.auc(recall_train, precision_train))

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel('recall')
plt.ylabel('precision')
plt.legend()
plt.savefig(os.path.join(figure_path, 'train_PR.png'))

if done_tweaking:
    print('Code for plotting test set PR curves.')

    confidence_thrs = []
    for pred_file, preds in preds_test.items():
        confidence_thrs += [pred[4] for pred in preds]

    confidence_thrs = np.sort(np.array(confidence_thrs,dtype=float)) # using (ascending) list of confidence scores as thresholds

    n_gts_test = 0
    for gts_file, gts in gts_test.items():
        n_gts_test += len(gts)

    plt.figure(1)
    for iou_thr in [.25, .5, .75]:
        tp_test = np.zeros(len(confidence_thrs))
        fp_test = np.zeros(len(confidence_thrs))
        fn_test = np.zeros(len(confidence_thrs))
        for i, conf_thr in enumerate(confidence_thrs):
            tp_test[i], fp_test[i], fn_test[i] = compute_counts(preds_test, gts_test, iou_thr=iou_thr, conf_thr=conf_thr)

        precision_test = np.nan_to_num(tp_test/(tp_test + fp_test), nan=1.)
        recall_test = 1 - fn_test/n_gts_test
        plt.plot(recall_test, precision_test, label="iou_thr="+str(iou_thr))
        print(iou_thr, metrics.auc(recall_test,  precision_test))

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.legend()
    plt.savefig(os.path.join(figure_path, 'test_PR.png'))


