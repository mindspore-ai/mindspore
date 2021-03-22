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
# =========================================================================

import re
import numpy as np
from mindspore import log as logger
import nibabel as nib
from src.utils import correct_nifti_head, get_random_patch

MAX_SEED = np.iinfo(np.uint32).max + 1
np_str_obj_array_pattern = re.compile(r'[SaUO]')

class Dataset:
    """
    A generic dataset with a length property and an optional callable data transform
    when fetching a data sample.

    Args:
        data: input data to load and transform to generate dataset for model.
        seg: segment data to load and transform to generate dataset for model
    """
    def __init__(self, data, seg):
        self.data = data
        self.seg = seg

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        seg = self.seg[index]
        return [data], [seg]

class LoadData:
    """
    Load Image data from provided files.
    """
    def __init__(self, canonical=False, dtype=np.float32):
        """
        Args:
        canonical: if True, load the image as closest to canonical axis format.
        dtype: convert the loaded image to this data type.
        """
        self.canonical = canonical
        self.dtype = dtype

    def operation(self, filename):
        """
        Args:
            filename: path file or file-like object or a list of files.
        """
        img_array = list()
        compatible_meta = dict()
        filename = str(filename, encoding="utf-8")
        filename = [filename]
        for name in filename:
            img = nib.load(name)
            img = correct_nifti_head(img)
            header = dict(img.header)
            header["filename_or_obj"] = name
            header["affine"] = img.affine
            header["original_affine"] = img.affine.copy()
            header["canonical"] = self.canonical
            ndim = img.header["dim"][0]
            spatial_rank = min(ndim, 3)
            header["spatial_shape"] = img.header["dim"][1 : spatial_rank + 1]
            if self.canonical:
                img = nib.as_closest_canonical(img)
                header["affine"] = img.affine
            img_array.append(np.array(img.get_fdata(dtype=self.dtype)))
            img.uncache()
            if not compatible_meta:
                for meta_key in header:
                    meta_datum = header[meta_key]
                    if isinstance(meta_datum, np.ndarray) \
                        and np_str_obj_array_pattern.search(meta_datum.dtype.str) is not None:
                        continue
                    compatible_meta[meta_key] = meta_datum
            else:
                assert np.allclose(header["affine"], compatible_meta["affine"])

        img_array = np.stack(img_array, axis=0) if len(img_array) > 1 else img_array[0]
        return img_array

    def __call__(self, filename1, filename2):
        img_array = self.operation(filename1)
        seg_array = self.operation(filename2)
        return img_array, seg_array


class ExpandChannel:
    """
    Expand a 1-length channel dimension to the input image.
    """
    def operation(self, data):
        """
        Args:
            data(numpy.array): input data to expand channel.
        """
        return data[None]

    def __call__(self, img, label):
        img_array = self.operation(img)
        seg_array = self.operation(label)
        return img_array, seg_array


class Orientation:
    """
    Change the input image's orientation into the specified based on `ax`.
    """
    def __init__(self, ax="RAS", labels=tuple(zip("LPI", "RAS"))):
        """
        Args:
            ax: N elements sequence for ND input's orientation.
            labels: optional, None or sequence of (2,) sequences
                (2,) sequences are labels for (beginning, end) of output axis.
        """
        self.ax = ax
        self.labels = labels

    def operation(self, data, affine=None):
        """
        original orientation of `data` is defined by `affine`.

        Args:
            data: in shape (num_channels, H[, W, ...]).
            affine (matrix): (N+1)x(N+1) original affine matrix for spatially ND `data`. Defaults to identity.

        Returns:
            data (reoriented in `self.ax`), original ax, current ax.
        """
        if data.ndim <= 1:
            raise ValueError("data must have at least one spatial dimension.")
        if affine is None:
            affine = np.eye(data.ndim, dtype=np.float64)
            affine_copy = affine
        else:
            affine_copy = to_affine_nd(data.ndim-1, affine)
        src = nib.io_orientation(affine_copy)
        dst = nib.orientations.axcodes2ornt(self.ax[:data.ndim-1], labels=self.labels)
        spatial_ornt = nib.orientations.ornt_transform(src, dst)
        ornt = spatial_ornt.copy()
        ornt[:, 0] += 1
        ornt = np.concatenate([np.array([[0, 1]]), ornt])
        data = nib.orientations.apply_orientation(data, ornt)
        return data

    def __call__(self, img, label):
        img_array = self.operation(img)
        seg_array = self.operation(label)
        return img_array, seg_array


class ScaleIntensityRange:
    """
    Apply specific intensity scaling to the whole numpy array.
    Scaling from [src_min, src_max] to [tgt_min, tgt_max] with clip option.

    Args:
        src_min: intensity original range min.
        src_max: intensity original range max.
        tgt_min: intensity target range min.
        tgt_max: intensity target range max.
        is_clip: whether to clip after scaling.
    """
    def __init__(self, src_min, src_max, tgt_min, tgt_max, is_clip=False):
        self.src_min = src_min
        self.src_max = src_max
        self.tgt_min = tgt_min
        self.tgt_max = tgt_max
        self.is_clip = is_clip

    def operation(self, data):
        if self.src_max - self.src_min == 0.0:
            logger.warning("Divide by zero (src_min == src_max)")
            return data - self.src_min + self.tgt_min
        data = (data - self.src_min) / (self.src_max - self.src_min)
        data = data * (self.tgt_max - self.tgt_min) + self.tgt_min
        if self.is_clip:
            data = np.clip(data, self.tgt_min, self.tgt_max)
        return data

    def __call__(self, image, label):
        image = self.operation(image)
        return image, label


class RandomCropSamples:
    """
    Random crop 3d image.

    Args:
        keys: keys of the corresponding items to be transformed.
        roi_size: if `random_size` is True, it specifies the minimum crop region.
        num_samples: the amount of crop images.
    """
    def __init__(self, roi_size, num_samples=1):
        self.roi_size = roi_size
        self.num_samples = num_samples
        self.set_random_state(0)

    def set_random_state(self, seed=None):
        """
        Set the random seed to control the slice size.

        Args:
            seed: set the random state with an integer seed.
        """
        if seed is not None:
            _seed = seed % MAX_SEED
            self.rand_fn = np.random.RandomState(_seed)
        else:
            self.rand_fn = np.random.RandomState()
        return self

    def get_random_slice(self, img_size):
        slices = (slice(None),) + get_random_patch(img_size, self.roi_size, self.rand_fn)
        return slices

    def __call__(self, image, label):
        res_image = []
        res_label = []
        for _ in range(self.num_samples):
            slices = self.get_random_slice(image.shape[1:])
            img = image[slices]
            label_crop = label[slices]
            res_image.append(img)
            res_label.append(label_crop)
        return np.array(res_image), np.array(res_label)

class OneHot:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def one_hot(self, labels):
        N, K = labels.shape
        one_hot_encoding = np.zeros((N, self.num_classes, K), dtype=np.float32)
        for i in range(N):
            for j in range(K):
                one_hot_encoding[i, labels[i][j], j] = 1
        return one_hot_encoding

    def operation(self, labels):
        N, _, D, H, W = labels.shape
        labels = labels.astype(np.int32)
        labels = np.reshape(labels, (N, -1))
        labels = self.one_hot(labels)
        labels = np.reshape(labels, (N, self.num_classes, D, H, W))
        return labels

    def __call__(self, image, label):
        label = self.operation(label)
        return image, label
