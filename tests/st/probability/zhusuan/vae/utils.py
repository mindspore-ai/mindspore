# Copyright 2020 Huawei Technologies Co., Ltd
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
""" Utils """

from PIL import Image
import numpy as np

from mindspore.common import dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset.transforms.vision.c_transforms as CV
from mindspore.dataset.transforms.vision import Inter


def create_dataset(data_path, batch_size=32, repeat_size=1,
                   num_parallel_workers=1):
    """ create dataset for train or test
    Args:
        data_path: Data path
        batch_size: The number of data records in each group
        repeat_size: The number of replicated data records
        num_parallel_workers: The number of parallel workers
    """
    # define dataset
    mnist_ds = ds.MnistDataset(data_path)
    #mnist_ds = ds.MnistDataset(data_path,num_samples=32)

    # define operation parameters
    resize_height, resize_width = 32, 32
    rescale = 1.0 / 255.0
    shift = 0.0

    # define map operations
    # resize images to (32, 32)
    resize_op = CV.Resize((resize_height, resize_width),
                          interpolation=Inter.LINEAR)
    rescale_op = CV.Rescale(rescale, shift)  # rescale images
    # change shape from (height, width, channel) to (channel, height, width) to fit network.
    hwc2chw_op = CV.HWC2CHW()
    # change data type of label to int32 to fit network
    type_cast_op = C.TypeCast(mstype.int32)

    # apply map operations on images
    mnist_ds = mnist_ds.map(input_columns="label", operations=type_cast_op,
                            num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(input_columns="image", operations=resize_op,
                            num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(input_columns="image", operations=rescale_op,
                            num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(input_columns="image", operations=hwc2chw_op,
                            num_parallel_workers=num_parallel_workers)

    # apply DatasetOps
    buffer_size = 10000
    # 10000 as in LeNet train script
    mnist_ds = mnist_ds.shuffle(buffer_size=buffer_size)
    mnist_ds = mnist_ds.batch(batch_size, drop_remainder=True)
    mnist_ds = mnist_ds.repeat(repeat_size)

    return mnist_ds


def save_img(data, name, size=32, num=32):
    """
    Visualize data and save to target files
    Args:
        data: nparray of size (num, size, size)
        name: output file name
        size: image size
        num: number of images
    """
    col = int(num / 8)
    row = 8

    imgs = Image.new('L', (size*col, size*row))
    for i in range(num):
        j = i/8
        img_data = data[i]
        img_data = np.resize(img_data, (size, size))
        img_data = img_data * 255
        img_data = img_data.astype(np.uint8)
        im = Image.fromarray(img_data, 'L')
        imgs.paste(im, (int(j) * size, (i % 8) * size))
    imgs.save(name)
