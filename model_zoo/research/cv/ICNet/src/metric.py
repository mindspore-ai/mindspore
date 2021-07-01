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
from mindspore import Tensor
import mindspore.ops as ops
import mindspore.common.dtype as dtype

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

        self.total_correct += correct
        self.total_label += labeled

        self.total_inter += inter
        self.total_union += union

    def get(self):
        """Gets the current evaluation result.

        Returns
        -------
        metrics : tuple of float
            pixAcc and mIoU
        """
        mean = ops.ReduceMean(keep_dims=False)
        pixAcc = 1.0 * self.total_correct / (2.220446049250313e-16 + self.total_label)  # remove c.spacing(1)
        IoU = 1.0 * self.total_inter / (2.220446049250313e-16 + self.total_union)

        mIoU = mean(IoU, axis=0)

        return pixAcc, mIoU

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        zeros = ops.Zeros()
        self.total_inter = zeros(self.nclass, dtype.float32)
        self.total_union = zeros(self.nclass, dtype.float32)
        self.total_correct = 0
        self.total_label = 0


def batch_pix_accuracy(output, target):
    """PixAcc"""

    predict = ops.Argmax(output_type=dtype.int32, axis=1)(output) + 1
    # （1，19， 1024，2048）-->(1, 1024,2048)
    target = target + 1

    typetrue = dtype.float32
    cast = ops.Cast()
    sumtarget = ops.ReduceSum()
    sumcorrect = ops.ReduceSum()

    labeled = cast(target > 0, typetrue)
    pixel_labeled = sumtarget(labeled)  # sum of pixels without 0

    pixel_correct = sumcorrect(cast(predict == target, typetrue) * cast(target > 0, typetrue))  # 标记正确的像素和

    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


def batch_intersection_union(output, target, nclass):
    """mIoU"""
    # inputs are numpy array, output 4D, target 3D
    predict = ops.Argmax(output_type=dtype.int32, axis=1)(output) + 1  # [N,H,W]
    target = target.astype(dtype.float32) + 1  # [N,H,W]

    typetrue = dtype.float32
    cast = ops.Cast()
    predict = cast(predict, typetrue) * cast(target > 0, typetrue)
    intersection = cast(predict, typetrue) * cast(predict == target, typetrue)
    # areas of intersection and union
    # element 0 in intersection occur the main difference from np.bincount. set boundary to -1 is necessary.

    Range = Tensor([0.0, 20.0], dtype.float32)
    hist = ops.HistogramFixedWidth(nclass + 1)
    area_inter = hist(intersection, Range)
    area_pred = hist(predict, Range)
    area_lab = hist(target, Range)

    area_union = area_pred + area_lab - area_inter

    area_inter = area_inter[1:]
    area_union = area_union[1:]
    Sum = ops.ReduceSum()
    assert Sum(cast(area_inter > area_union, typetrue)) == 0, "Intersection area should be smaller than Union area"
    return cast(area_inter, typetrue), cast(area_union, typetrue)
