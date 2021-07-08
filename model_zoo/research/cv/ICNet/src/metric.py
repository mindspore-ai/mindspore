# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Evaluation Metrics for Semantic Segmentation"""
import numpy as np

__all__ = ['SegmentationMetric', 'batch_pix_accuracy', 'batch_intersection_union']


class SegmentationMetric:
    """Computes pixAcc and mIoU metric scores"""

    def __init__(self, nclass):
        super(SegmentationMetric, self).__init__()
        self.nclass = nclass
        self.reset()

    def update(self, pred, label):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : 'NumpyArray' or list of `NumpyArray`
            The labels of the data.
        preds : 'NumpyArray' or list of `NumpyArray`
            Predicted values.
        """
        correct, labeled = batch_pix_accuracy(pred, label)
        inter, union = batch_intersection_union(pred, label, self.nclass)

        self.total_correct = correct + self.total_correct
        self.total_label = labeled + self.total_label

        self.total_inter = inter + self.total_inter
        self.total_union = union + self.total_union

    def get(self):
        """Gets the current evaluation result.

        Returns
        -------
        metrics : tuple of float
            pixAcc and mIoU
        """
        pixAcc = np.true_divide(self.total_correct, (2.220446049250313e-16 + self.total_label))  # remove c.spacing(1)
        IoU = np.true_divide(self.total_inter, (2.220446049250313e-16 + self.total_union))

        mIoU = np.mean(IoU)

        return mIoU, pixAcc

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.total_inter = np.zeros(self.nclass, dtype=np.float)
        self.total_union = np.zeros(self.nclass, dtype=np.float)
        self.total_correct = 0
        self.total_label = 0


def batch_pix_accuracy(output, target):
    """PixAcc"""

    predict = np.argmax(output, axis=1) + 1
    # （1，19， 1024，2048）-->(1, 1024,2048)
    target = target + 1

    labeled = np.array(target > 0).astype(int)
    pixel_labeled = np.sum(labeled)  # sum of pixels without 0

    pixel_correct = np.sum(np.array(predict == target).astype(int) * np.array(target > 0).astype(int))
    # Quantity of correct pixels

    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


def batch_intersection_union(output, target, nclass):
    """mIoU"""
    # inputs are numpy array, output 4D, target 3D
    predict = np.argmax(output, axis=1) + 1  # [N,H,W]
    target = target.astype(float) + 1  # [N,H,W]

    predict = predict.astype(float) * np.array(target > 0).astype(float)
    intersection = predict * np.array(predict == target).astype(float)
    # areas of intersection and union
    # element 0 in intersection occur the main difference from np.bincount. set boundary to -1 is necessary.

    area_inter, _ = np.array(np.histogram(intersection, bins=nclass, range=(1, nclass+1)))
    area_pred, _ = np.array(np.histogram(predict, bins=nclass, range=(1, nclass+1)))
    area_lab, _ = np.array(np.histogram(target, bins=nclass, range=(1, nclass+1)))

    area_all = area_pred + area_lab
    area_union = area_all - area_inter

    return area_inter, area_union
