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
"""hub config."""
from src.resnet import resnet20
from mindspore import Tensor
import numpy as np


def create_network(name, thres_filename, *args, **kwargs):
    if name == 'resnet20':
        thres = np.load(thres_filename)
        thres = Tensor(thres.astype(np.float32))
        return resnet20(thres=thres)
    raise NotImplementedError(f"{name} is not implemented in the repo")
