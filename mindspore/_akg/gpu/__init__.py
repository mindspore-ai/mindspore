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

"""__init__"""
from .equal import Equal
from .equal import gpu_schedule_Equal
from .tile import Tile
from .tile import gpu_schedule_Tile
from .cast import Cast
from .cast import gpu_schedule_Cast
from .relu6 import ReLU6, gpu_schedule_ReLU6
from .relu6_grad import ReLU6Grad, gpu_schedule_ReLU6Grad
from .squeeze import Squeeze, gpu_schedule_Squeeze
from .squeeze_grad import SqueezeGrad, gpu_schedule_SqueezeGrad
from .mean import SimpleMean, gpu_schedule_SimpleMean
from .mean_grad import SimpleMeanGrad, gpu_schedule_SimpleMeanGrad
from .mul import Mul, gpu_schedule_Mul
from .hsigmoid import HSigmoid, gpu_schedule_HSigmoid
from .hsigmoid_grad import HSigmoidGrad, gpu_schedule_HSigmoidGrad
from .hswish import HSwish, gpu_schedule_HSwish
from .hswish_grad import HSwishGrad, gpu_schedule_HSwishGrad
from .logical_or import LogicalOr, gpu_schedule_LogicalOr
from .logical_not import LogicalNot, gpu_schedule_LogicalNot
from .logical_and import LogicalAnd, gpu_schedule_LogicalAnd
from .sub import Sub, gpu_schedule_Sub
from .less_equal import LessEqual, gpu_schedule_LessEqual
