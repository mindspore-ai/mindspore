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
"""hub config."""
import numpy as np
from mindspore import Tensor
from src.resnet_imgnet import resnet50


def get_index(filename):
    index = []
    with open(filename) as fr:
        for line in fr:
            ind = Tensor((np.array(line.strip('\n').split(' ')[:-1])).astype(np.int32).reshape(-1, 1))
            index.append(ind)
    return index


def create_network(name, rate=0.65, index_filename='index.txt', **kwargs):
    index = get_index(index_filename)
    if name == 'resnet50-0.65x':
        return resnet50(rate=rate, index=index, **kwargs)
    raise NotImplementedError(f"{name} is not implemented in the repo")
