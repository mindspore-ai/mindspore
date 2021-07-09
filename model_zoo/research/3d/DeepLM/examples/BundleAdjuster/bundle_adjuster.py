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
# ===========================================================================
"""DeepLM BA demo."""
import argparse
from time import time

import mindspore as ms
from mindspore.ops.functional import tensor_mul
from mindspore import Tensor
from mindspore import context

from lm_solver.solver import solve
from ba_core.loss import SnavelyReprojectionError
from ba_core.io import load_bal_from_file


DOUBLE_ENABLE = False

FLOAT_DTYPE = ms.float64 if DOUBLE_ENABLE else ms.float32
INT_DYTPE = ms.int32

parser = argparse.ArgumentParser(description='Bundle adjuster')
parser.add_argument('--balFile', default='data/problem-1723-156502-pre.txt')
parser.add_argument('--device', default='gpu')
args = parser.parse_args()

filename = args.balFile
device = args.device
if device == "gpu" or "cuda":
    device_target = "GPU"
elif device == "cpu":
    device_target = "CPU"

context.set_context(device_target=device_target)
context.set_context(mode=context.PYNATIVE_MODE)

# Load BA data
points, cameras, features, pt_idx, cam_idx = load_bal_from_file(filename=filename,
                                                                feature_dim=2,
                                                                camera_dim=9,
                                                                point_dim=3,
                                                                double=DOUBLE_ENABLE)

points = Tensor(points, FLOAT_DTYPE)
cameras = Tensor(cameras, FLOAT_DTYPE)
features = Tensor(features, FLOAT_DTYPE)
pt_idx = Tensor(pt_idx, INT_DYTPE)
cam_idx = Tensor(cam_idx, INT_DYTPE)

# Optionally use CUDA, move data to device
points = tensor_mul(points, Tensor(1.0, FLOAT_DTYPE))
cameras = tensor_mul(cameras, Tensor(1.0, FLOAT_DTYPE))
features = tensor_mul(features, Tensor(1.0, FLOAT_DTYPE))
pt_idx = tensor_mul(pt_idx, Tensor(1, INT_DYTPE))
cam_idx = tensor_mul(cam_idx, Tensor(1, INT_DYTPE))

t1 = time()
solve(variables=[points, cameras], constants=[features], indices=[pt_idx, cam_idx],
      fn=SnavelyReprojectionError(), num_iterations=15, num_success_iterations=15)
t2 = time()

print("Time used %f secs." % (t2 - t1))
