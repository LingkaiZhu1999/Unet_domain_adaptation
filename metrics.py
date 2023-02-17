# import numpy as np

import torch
from monai.metrics import compute_meandice, do_metric_reduction
import numpy as np
from torchmetrics import Metric

class Dice(Metric):
    full_state_update: bool = False
    def __init__(self, n_class=3, brats=True):
        super().__init__(dist_sync_on_step=False)
        self.n_class = n_class
        self.brats = brats
        self.add_state("loss_supervise", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("loss_contrast", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("steps", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("dice", default=torch.zeros((n_class,)), dist_reduce_fx="sum")
    def update(self, predict, label, loss_sup, loss_con):
        if self.brats:
            predict = (torch.sigmoid(predict) > 0.5).int()
            label = label
            loss_sup = loss_sup.detach()
            loss_con = loss_con.detach()

        self.steps += 1
        self.loss_supervise += loss_sup
        self.loss_contrast += loss_con
        self.dice += self.compute_metric(predict, label, compute_meandice, torch.tensor(1), torch.tensor(0))

    def compute(self):
        return self.dice / self.steps, self.loss_supervise / self.steps, self.loss_contrast / self.steps

    def compute_metric(self, predict, label, metric_function, best_metric, worst_metric):
        metric = metric_function(predict, label, include_background=self.brats)
        metric = torch.nan_to_num(metric, nan=worst_metric, posinf=worst_metric, neginf=worst_metric)
        metric = do_metric_reduction(metric, "mean_batch")[0]
        for i in range(self.n_class):
            if (label[:, i] != 1).all():
                metric[i - 1] += best_metric if (predict[:, i] != 1).all() else worst_metric
        
        return metric

class AverageLoss(Metric):
    full_state_update: bool = False
    def __init__(self):
        super().__init__(dist_sync_on_step=False)
        self.add_state("loss", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("steps", default=torch.zeros(1), dist_reduce_fx="sum")
    def update(self, loss):
        self.steps += 1
        self.loss += loss

    def compute(self):
        return self.loss / self.steps
        
def mean_iou(y_true_in, y_pred_in, print_table=False):
    if True: #not np.sum(y_true_in.flatten()) == 0:
        labels = y_true_in
        y_pred = y_pred_in

        true_objects = 2
        pred_objects = 2

        intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

        # Compute areas (needed for finding the union between all objects)
        area_true = np.histogram(labels, bins = true_objects)[0]
        area_pred = np.histogram(y_pred, bins = pred_objects)[0]
        area_true = np.expand_dims(area_true, -1)
        area_pred = np.expand_dims(area_pred, 0)

        # Compute union
        union = area_true + area_pred - intersection

        # Exclude background from the analysis
        intersection = intersection[1:,1:]
        union = union[1:,1:]
        union[union == 0] = 1e-9

        # Compute the intersection over union
        iou = intersection / union

        # Precision helper function
        def precision_at(threshold, iou):
            matches = iou > threshold
            true_positives = np.sum(matches, axis=1) == 1   # Correct objects
            false_positives = np.sum(matches, axis=0) == 0  # Missed objects
            false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
            tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
            return tp, fp, fn

        # Loop over IoU thresholds
        prec = []
        if print_table:
            print("Thresh\tTP\tFP\tFN\tPrec.")
        for t in np.arange(0.5, 1.0, 0.05):
            tp, fp, fn = precision_at(t, iou)
            if (tp + fp + fn) > 0:
                p = tp / (tp + fp + fn)
            else:
                p = 0
            if print_table:
                print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
            prec.append(p)

        if print_table:
            print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
        return np.mean(prec)

    else:
        if np.sum(y_pred_in.flatten()) == 0:
            return 1
        else:
            return 0


def batch_iou(output, target):
    output = torch.sigmoid(output).data.cpu().numpy() > 0.5
    target = (target.data.cpu().numpy() > 0.5).astype('int')
    output = output[:,0,:,:]
    target = target[:,0,:,:]

    ious = []
    for i in range(output.shape[0]):
        ious.append(mean_iou(output[i], target[i]))

    return np.mean(ious)


def mean_iou(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).data.cpu().numpy()
    target = target.data.cpu().numpy()
    ious = []
    for t in np.arange(0.5, 1.0, 0.05):
        output_ = output > t
        target_ = target > t
        intersection = (output_ & target_).sum()
        union = (output_ | target_).sum()
        iou = (intersection + smooth) / (union + smooth)
        ious.append(iou)

    return np.mean(ious)


def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)


def dice_coef(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = (torch.sigmoid(output).data.cpu() > 0.5).int()
    if torch.is_tensor(target):
        target = target.data.cpu()
    #output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    #target = target.view(-1).data.cpu().numpy()

    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)

def compute_dice(output, target):
    if torch.is_tensor(output):
        output = (torch.sigmoid(output).data.cpu() > 0.5).int()
    if torch.is_tensor(target):
        target = target.data.cpu()
    best_metric = 1
    worst_metric = 0
    n_class = 3
    dice = compute_meandice(output, target)
    dice = torch.nan_to_num(dice, nan=worst_metric, posinf=worst_metric, neginf=worst_metric)
    dice = do_metric_reduction(dice, "mean_batch")[0]
    for i in range(n_class):
            if (target[:, i] != 1).all():
                dice[i - 1] += best_metric if (output[:, i] != 1).all() else worst_metric
    return torch.mean(dice)

    
def accuracy(output, target):
    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    output = (np.round(output)).astype('int')
    target = target.view(-1).data.cpu().numpy()
    target = (np.round(target)).astype('int')
    (output == target).sum()

    return (output == target).sum() / len(output)

def ppv(output, target):
    smooth = 1e-5
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    intersection = (output * target).sum()
    return  (intersection + smooth) / \
           (output.sum() + smooth)

def sensitivity(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    intersection = (output * target).sum()

    return (intersection + smooth) / \
        (target.sum() + smooth)

#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# import numpy as np
# from medpy import metric


# def assert_shape(test, reference):

#     assert test.shape == reference.shape, "Shape mismatch: {} and {}".format(
#         test.shape, reference.shape)


# class ConfusionMatrix:

#     def __init__(self, test=None, reference=None):

#         self.tp = None
#         self.fp = None
#         self.tn = None
#         self.fn = None
#         self.size = None
#         self.reference_empty = None
#         self.reference_full = None
#         self.test_empty = None
#         self.test_full = None
#         self.set_reference(reference)
#         self.set_test(test)

#     def set_test(self, test):

#         self.test = test
#         self.reset()

#     def set_reference(self, reference):

#         self.reference = reference
#         self.reset()

#     def reset(self):

#         self.tp = None
#         self.fp = None
#         self.tn = None
#         self.fn = None
#         self.size = None
#         self.test_empty = None
#         self.test_full = None
#         self.reference_empty = None
#         self.reference_full = None

#     def compute(self):

#         if self.test is None or self.reference is None:
#             raise ValueError("'test' and 'reference' must both be set to compute confusion matrix.")

#         assert_shape(self.test, self.reference)

#         self.tp = int(((self.test != 0) * (self.reference != 0)).sum())
#         self.fp = int(((self.test != 0) * (self.reference == 0)).sum())
#         self.tn = int(((self.test == 0) * (self.reference == 0)).sum())
#         self.fn = int(((self.test == 0) * (self.reference != 0)).sum())
#         self.size = int(np.prod(self.reference.shape, dtype=np.int64))
#         self.test_empty = not np.any(self.test)
#         self.test_full = np.all(self.test)
#         self.reference_empty = not np.any(self.reference)
#         self.reference_full = np.all(self.reference)

#     def get_matrix(self):

#         for entry in (self.tp, self.fp, self.tn, self.fn):
#             if entry is None:
#                 self.compute()
#                 break

#         return self.tp, self.fp, self.tn, self.fn

#     def get_size(self):

#         if self.size is None:
#             self.compute()
#         return self.size

#     def get_existence(self):

#         for case in (self.test_empty, self.test_full, self.reference_empty, self.reference_full):
#             if case is None:
#                 self.compute()
#                 break

#         return self.test_empty, self.test_full, self.reference_empty, self.reference_full


# def dice(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
#     """2TP / (2TP + FP + FN)"""

#     if confusion_matrix is None:
#         confusion_matrix = ConfusionMatrix(test, reference)

#     tp, fp, tn, fn = confusion_matrix.get_matrix()
#     test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

#     if test_empty and reference_empty:
#         if nan_for_nonexisting:
#             return float("NaN")
#         else:
#             return 0.
#     return float(2. * tp / (2 * tp + fp + fn))


# def jaccard(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
#     """TP / (TP + FP + FN)"""

#     if confusion_matrix is None:
#         confusion_matrix = ConfusionMatrix(test, reference)

#     tp, fp, tn, fn = confusion_matrix.get_matrix()
#     test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

#     if test_empty and reference_empty:
#         if nan_for_nonexisting:
#             return float("NaN")
#         else:
#             return 0.

#     return float(tp / (tp + fp + fn))


# def precision(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
#     """TP / (TP + FP)"""

#     if confusion_matrix is None:
#         confusion_matrix = ConfusionMatrix(test, reference)

#     tp, fp, tn, fn = confusion_matrix.get_matrix()
#     test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

#     if test_empty:
#         if nan_for_nonexisting:
#             return float("NaN")
#         else:
#             return 0.

#     return float(tp / (tp + fp))


# def sensitivity(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
#     """TP / (TP + FN)"""

#     if confusion_matrix is None:
#         confusion_matrix = ConfusionMatrix(test, reference)

#     tp, fp, tn, fn = confusion_matrix.get_matrix()
#     test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

#     if reference_empty:
#         if nan_for_nonexisting:
#             return float("NaN")
#         else:
#             return 0.

#     return float(tp / (tp + fn))


# def recall(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
#     """TP / (TP + FN)"""

#     return sensitivity(test, reference, confusion_matrix, nan_for_nonexisting, **kwargs)


# def specificity(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
#     """TN / (TN + FP)"""

#     if confusion_matrix is None:
#         confusion_matrix = ConfusionMatrix(test, reference)

#     tp, fp, tn, fn = confusion_matrix.get_matrix()
#     test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

#     if reference_full:
#         if nan_for_nonexisting:
#             return float("NaN")
#         else:
#             return 0.

#     return float(tn / (tn + fp))


# def accuracy(test=None, reference=None, confusion_matrix=None, **kwargs):
#     """(TP + TN) / (TP + FP + FN + TN)"""

#     if confusion_matrix is None:
#         confusion_matrix = ConfusionMatrix(test, reference)

#     tp, fp, tn, fn = confusion_matrix.get_matrix()

#     return float((tp + tn) / (tp + fp + tn + fn))


# def fscore(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, beta=1., **kwargs):
#     """(1 + b^2) * TP / ((1 + b^2) * TP + b^2 * FN + FP)"""

#     precision_ = precision(test, reference, confusion_matrix, nan_for_nonexisting)
#     recall_ = recall(test, reference, confusion_matrix, nan_for_nonexisting)

#     return (1 + beta*beta) * precision_ * recall_ /\
#         ((beta*beta * precision_) + recall_)


# def false_positive_rate(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
#     """FP / (FP + TN)"""

#     return 1 - specificity(test, reference, confusion_matrix, nan_for_nonexisting)


# def false_omission_rate(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
#     """FN / (TN + FN)"""

#     if confusion_matrix is None:
#         confusion_matrix = ConfusionMatrix(test, reference)

#     tp, fp, tn, fn = confusion_matrix.get_matrix()
#     test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

#     if test_full:
#         if nan_for_nonexisting:
#             return float("NaN")
#         else:
#             return 0.

#     return float(fn / (fn + tn))


# def false_negative_rate(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
#     """FN / (TP + FN)"""

#     return 1 - sensitivity(test, reference, confusion_matrix, nan_for_nonexisting)


# def true_negative_rate(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
#     """TN / (TN + FP)"""

#     return specificity(test, reference, confusion_matrix, nan_for_nonexisting)


# def false_discovery_rate(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
#     """FP / (TP + FP)"""

#     return 1 - precision(test, reference, confusion_matrix, nan_for_nonexisting)


# def negative_predictive_value(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
#     """TN / (TN + FN)"""

#     return 1 - false_omission_rate(test, reference, confusion_matrix, nan_for_nonexisting)


# def total_positives_test(test=None, reference=None, confusion_matrix=None, **kwargs):
#     """TP + FP"""

#     if confusion_matrix is None:
#         confusion_matrix = ConfusionMatrix(test, reference)

#     tp, fp, tn, fn = confusion_matrix.get_matrix()

#     return tp + fp


# def total_negatives_test(test=None, reference=None, confusion_matrix=None, **kwargs):
#     """TN + FN"""

#     if confusion_matrix is None:
#         confusion_matrix = ConfusionMatrix(test, reference)

#     tp, fp, tn, fn = confusion_matrix.get_matrix()

#     return tn + fn


# def total_positives_reference(test=None, reference=None, confusion_matrix=None, **kwargs):
#     """TP + FN"""

#     if confusion_matrix is None:
#         confusion_matrix = ConfusionMatrix(test, reference)

#     tp, fp, tn, fn = confusion_matrix.get_matrix()

#     return tp + fn


# def total_negatives_reference(test=None, reference=None, confusion_matrix=None, **kwargs):
#     """TN + FP"""

#     if confusion_matrix is None:
#         confusion_matrix = ConfusionMatrix(test, reference)

#     tp, fp, tn, fn = confusion_matrix.get_matrix()

#     return tn + fp


# def hausdorff_distance(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, voxel_spacing=None, connectivity=1, **kwargs):

#     if confusion_matrix is None:
#         confusion_matrix = ConfusionMatrix(test, reference)

#     test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

#     if test_empty or test_full or reference_empty or reference_full:
#         if nan_for_nonexisting:
#             return float("NaN")
#         else:
#             return 0

#     test, reference = confusion_matrix.test, confusion_matrix.reference

#     return metric.hd(test, reference, voxel_spacing, connectivity)


# def hausdorff_distance_95(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, voxel_spacing=None, connectivity=1, **kwargs):

#     if confusion_matrix is None:
#         confusion_matrix = ConfusionMatrix(test, reference)

#     test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

#     if test_empty or test_full or reference_empty or reference_full:
#         if nan_for_nonexisting:
#             return float("NaN")
#         else:
#             return 0

#     test, reference = confusion_matrix.test, confusion_matrix.reference

#     return metric.hd95(test, reference, voxel_spacing, connectivity)


# def avg_surface_distance(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, voxel_spacing=None, connectivity=1, **kwargs):

#     if confusion_matrix is None:
#         confusion_matrix = ConfusionMatrix(test, reference)

#     test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

#     if test_empty or test_full or reference_empty or reference_full:
#         if nan_for_nonexisting:
#             return float("NaN")
#         else:
#             return 0

#     test, reference = confusion_matrix.test, confusion_matrix.reference

#     return metric.asd(test, reference, voxel_spacing, connectivity)


# def avg_surface_distance_symmetric(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, voxel_spacing=None, connectivity=1, **kwargs):

#     if confusion_matrix is None:
#         confusion_matrix = ConfusionMatrix(test, reference)

#     test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

#     if test_empty or test_full or reference_empty or reference_full:
#         if nan_for_nonexisting:
#             return float("NaN")
#         else:
#             return 0

#     test, reference = confusion_matrix.test, confusion_matrix.reference

#     return metric.assd(test, reference, voxel_spacing, connectivity)


# ALL_METRICS = {
#     "False Positive Rate": false_positive_rate,
#     "Dice": dice,
#     "Jaccard": jaccard,
#     "Hausdorff Distance": hausdorff_distance,
#     "Hausdorff Distance 95": hausdorff_distance_95,
#     "Precision": precision,
#     "Recall": recall,
#     "Avg. Symmetric Surface Distance": avg_surface_distance_symmetric,
#     "Avg. Surface Distance": avg_surface_distance,
#     "Accuracy": accuracy,
#     "False Omission Rate": false_omission_rate,
#     "Negative Predictive Value": negative_predictive_value,
#     "False Negative Rate": false_negative_rate,
#     "True Negative Rate": true_negative_rate,
#     "False Discovery Rate": false_discovery_rate,
#     "Total Positives Test": total_positives_test,
#     "Total Negatives Test": total_negatives_test,
#     "Total Positives Reference": total_positives_reference,
#     "total Negatives Reference": total_negatives_reference
# }
