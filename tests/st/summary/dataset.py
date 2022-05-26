# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
"""dataset base."""
import os

from mindspore import dataset as ds
from mindspore.common import dtype as mstype
from mindspore.dataset.transforms import transforms as C
from mindspore.dataset.vision import Inter
from mindspore.dataset.vision import transforms as CV


def create_mnist_dataset(mode='train', num_samples=2, batch_size=2):
    """create dataset for train or test"""
    mnist_path = '/home/workspace/mindspore_dataset/mnist'
    num_parallel_workers = 1

    # define dataset
    mnist_ds = ds.MnistDataset(os.path.join(mnist_path, mode), num_samples=num_samples, shuffle=False)

    resize_height, resize_width = 32, 32

    # define map operations
    resize_op = CV.Resize((resize_height, resize_width), interpolation=Inter.LINEAR)  # Bilinear mode
    rescale_nml_op = CV.Rescale(1 / 0.3081, -1 * 0.1307 / 0.3081)
    rescale_op = CV.Rescale(1.0 / 255.0, shift=0.0)
    hwc2chw_op = CV.HWC2CHW()
    type_cast_op = C.TypeCast(mstype.int32)

    # apply map operations on images
    mnist_ds = mnist_ds.map(operations=type_cast_op, input_columns="label", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=resize_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=rescale_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=rescale_nml_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=hwc2chw_op, input_columns="image", num_parallel_workers=num_parallel_workers)

    # apply DatasetOps
    mnist_ds = mnist_ds.batch(batch_size=batch_size, drop_remainder=True)

    return mnist_ds
