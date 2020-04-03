# Copyright 2019 Huawei Technologies Co., Ltd
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
import mindspore.dataset as ds
import mindspore.dataset.transforms.vision.c_transforms as vision
from mindspore.dataset.transforms.vision import Inter
import numpy as np
import sys

import mindspore.context as context
import mindspore.nn as nn
from mindspore.common.tensor import Tensor
from mindspore.common.api import _executor
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
data_path = sys.argv[1]
SCHEMA_DIR = "{0}/resnet_all_datasetSchema.json".format(data_path)


def test_me_de_train_dataset():
    data_list = ["{0}/train-00001-of-01024.data".format(data_path)]
    data_set = ds.StorageDataset(data_list, schema=SCHEMA_DIR,
                                 columns_list=["image/encoded", "image/class/label"])

    resize_height = 224
    resize_width = 224
    rescale = 1.0 / 255.0
    shift = 0.0

    # define map operations

    decode_op = vision.Decode()
    resize_op = vision.Resize(resize_height, resize_width,
                              Inter.LINEAR)  # Bilinear as default
    rescale_op = vision.Rescale(rescale, shift)

    # apply map operations on images
    data_set = data_set.map(input_columns="image/encoded", operations=decode_op)
    data_set = data_set.map(input_columns="image/encoded", operations=resize_op)
    data_set = data_set.map(input_columns="image/encoded", operations=rescale_op)
    hwc2chw_op = vision.HWC2CHW()
    data_set = data_set.map(input_columns="image/encoded", operations=hwc2chw_op)
    data_set = data_set.repeat(1)
    # apply batch operations
    batch_size = 32
    data_set = data_set.batch(batch_size, drop_remainder=True)
    return data_set


def convert_type(shapes, types):
    ms_types = []
    for np_shape, np_type in zip(shapes, types):
        input_np = np.zeros(np_shape, np_type)
        tensor = Tensor(input_np)
        ms_types.append(tensor.dtype())
    return ms_types


if __name__ == '__main__':
    data_set = test_me_de_train_dataset()
    dataset_size = data_set.get_dataset_size()
    batch_size = data_set.get_batch_size()

    dataset_shapes = data_set.output_shapes()
    np_types = data_set.output_types()
    dataset_types = convert_type(dataset_shapes, np_types)

    ds1 = data_set.device_que()
    get_next = P.GetNext(dataset_types, dataset_shapes, 2, ds1.queue_name)
    tadd = P.ReLU()


    class dataiter(nn.Cell):
        def __init__(self):
            super(dataiter, self).__init__()

        def construct(self):
            input, label = get_next()
            return tadd(input)


    net = dataiter()
    net.set_train()

    _executor.init_dataset(ds1.queue_name, 39, batch_size,
                              dataset_types, dataset_shapes, (), 'dataset')
    ds1.send()

    for data in data_set.create_tuple_iterator():
        output = net()
        print(data[0].any())
        print(
            "****************************************************************************************************")
        d = output.asnumpy()
        print(d)
        print(
            "end+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++",
            d.any())

        assert (
            (data[0] == d).all()), "TDT test execute failed, please check current code commit"
    print(
        "+++++++++++++++++++++++++++++++++++[INFO] Success+++++++++++++++++++++++++++++++++++++++++++")

