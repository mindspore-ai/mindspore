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
import numpy as np

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common.api import jit
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


class Net(nn.Cell):
    def __init__(self, crop_size):
        super(Net, self).__init__()
        self.crop_and_resize = P.CropAndResize()
        self.crop_size = crop_size

    @jit
    def construct(self, x, boxes, box_index):
        return self.crop_and_resize(x, boxes, box_index, self.crop_size)


def test_net_float32():
    batch_size = 1
    num_boxes = 5
    image_height = 256
    image_width = 256
    channels = 3
    image = np.random.normal(size=[batch_size, image_height, image_width, channels]).astype(np.float32)
    boxes = np.random.uniform(size=[num_boxes, 4]).astype(np.float32)
    box_index = np.random.uniform(size=[num_boxes], low=0, high=batch_size).astype(np.int32)
    crop_size = (24, 24)
    net = Net(crop_size=crop_size)
    output = net(Tensor(image), Tensor(boxes), Tensor(box_index))
    print(output.asnumpy())
