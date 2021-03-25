# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# less required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import math
import numpy as np
from src.config import config

def correct_nifti_head(img):
    """
    Check nifti object header's format, update the header if needed.
    In the updated image pixdim matches the affine.

    Args:
        img: nifti image object
    """
    dim = img.header["dim"][0]
    if dim >= 5:
        return img
    pixdim = np.asarray(img.header.get_zooms())[:dim]
    norm_affine = np.sqrt(np.sum(np.square(img.affine[:dim, :dim]), 0))
    if np.allclose(pixdim, norm_affine):
        return img
    if hasattr(img, "get_sform"):
        return rectify_header_sform_qform(img)
    return img

def get_random_patch(dims, patch_size, rand_fn=None):
    """
    Returns a tuple of slices to define a random patch in an array of shape `dims` with size `patch_size`.

    Args:
        dims: shape of source array
        patch_size: shape of patch size to generate
        rand_fn: generate random numbers

    Returns:
        (tuple of slice): a tuple of slice objects defining the patch
    """
    rand_int = np.random.randint if rand_fn is None else rand_fn.randint
    min_corner = tuple(rand_int(0, ms - ps + 1) if ms > ps else 0 for ms, ps in zip(dims, patch_size))
    return tuple(slice(mc, mc + ps) for mc, ps in zip(min_corner, patch_size))


def first(iterable, default=None):
    """
    Returns the first item in the given iterable or `default` if empty, meaningful mostly with 'for' expressions.
    """
    for i in iterable:
        return i
    return default

def _get_scan_interval(image_size, roi_size, num_image_dims, overlap):
    """
    Compute scan interval according to the image size, roi size and overlap.
    Scan interval will be `int((1 - overlap) * roi_size)`, if interval is 0,
    use 1 instead to make sure sliding window works.
    """
    if len(image_size) != num_image_dims:
        raise ValueError("image different from spatial dims.")
    if len(roi_size) != num_image_dims:
        raise ValueError("roi size different from spatial dims.")

    scan_interval = []
    for i in range(num_image_dims):
        if roi_size[i] == image_size[i]:
            scan_interval.append(int(roi_size[i]))
        else:
            interval = int(roi_size[i] * (1 - overlap))
            scan_interval.append(interval if interval > 0 else 1)
    return tuple(scan_interval)

def dense_patch_slices(image_size, patch_size, scan_interval):
    """
    Enumerate all slices defining ND patches of size `patch_size` from an `image_size` input image.

    Args:
        image_size: dimensions of image to iterate over
        patch_size: size of patches to generate slices
        scan_interval: dense patch sampling interval

    Returns:
        a list of slice objects defining each patch
    """
    num_spatial_dims = len(image_size)
    patch_size = patch_size
    scan_num = []
    for i in range(num_spatial_dims):
        if scan_interval[i] == 0:
            scan_num.append(1)
        else:
            num = int(math.ceil(float(image_size[i]) / scan_interval[i]))
            scan_dim = first(d for d in range(num) if d * scan_interval[i] + patch_size[i] >= image_size[i])
            scan_num.append(scan_dim + 1 if scan_dim is not None else 1)
    starts = []
    for dim in range(num_spatial_dims):
        dim_starts = []
        for idx in range(scan_num[dim]):
            start_idx = idx * scan_interval[dim]
            start_idx -= max(start_idx + patch_size[dim] - image_size[dim], 0)
            dim_starts.append(start_idx)
        starts.append(dim_starts)
    out = np.asarray([x.flatten() for x in np.meshgrid(*starts, indexing="ij")]).T
    return [(slice(None),)*2 + tuple(slice(s, s + patch_size[d]) for d, s in enumerate(x)) for x in out]

def create_sliding_window(image, roi_size, overlap):
    num_image_dims = len(image.shape) - 2
    if overlap < 0 or overlap >= 1:
        raise AssertionError("overlap must be >= 0 and < 1.")
    image_size_temp = list(image.shape[2:])
    image_size = tuple(max(image_size_temp[i], roi_size[i]) for i in range(num_image_dims))

    scan_interval = _get_scan_interval(image_size, roi_size, num_image_dims, overlap)
    slices = dense_patch_slices(image_size, roi_size, scan_interval)
    windows_sliding = [image[slice] for slice in slices]
    return windows_sliding, slices

def one_hot(labels):
    N, _, D, H, W = labels.shape
    labels = np.reshape(labels, (N, -1))
    labels = labels.astype(np.int32)
    N, K = labels.shape
    one_hot_encoding = np.zeros((N, config['num_classes'], K), dtype=np.float32)
    for i in range(N):
        for j in range(K):
            one_hot_encoding[i, labels[i][j], j] = 1
    labels = np.reshape(one_hot_encoding, (N, config['num_classes'], D, H, W))
    return labels

def CalculateDice(y_pred, label):
    """
    Args:
        y_pred: predictions. As for classification tasks,
            `y_pred` should has the shape [BN] where N is larger than 1. As for segmentation tasks,
            the shape should be [BNHW] or [BNHWD].
        label: ground truth, the first dim is batch.
    """
    y_pred_output = np.expand_dims(np.argmax(y_pred, axis=1), axis=1)
    y_pred = one_hot(y_pred_output)
    y = one_hot(label)
    y_pred, y = ignore_background(y_pred, y)
    inter = np.dot(y_pred.flatten(), y.flatten()).astype(np.float64)
    union = np.dot(y_pred.flatten(), y_pred.flatten()).astype(np.float64) + np.dot(y.flatten(), \
                   y.flatten()).astype(np.float64)
    single_dice_coeff = 2 * inter / (union + 1e-6)
    return single_dice_coeff, y_pred_output

def ignore_background(y_pred, label):
    """
    This function is used to remove background (the first channel) for `y_pred` and `y`.
    Args:
        y_pred: predictions. As for classification tasks,
            `y_pred` should has the shape [BN] where N is larger than 1. As for segmentation tasks,
            the shape should be [BNHW] or [BNHWD].
        label: ground truth, the first dim is batch.
    """
    label = label[:, 1:] if label.shape[1] > 1 else label
    y_pred = y_pred[:, 1:] if y_pred.shape[1] > 1 else y_pred
    return y_pred, label
