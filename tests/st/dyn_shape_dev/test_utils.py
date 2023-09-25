# Copyright 2023 Huawei Technologies Co., Ltd
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

from functools import wraps
from mindspore import nn
import mindspore as ms
from mindspore import Tensor
import numpy as np

ms.set_context(jit_syntax_level=ms.STRICT)


class Net(nn.Cell):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def construct(self, *inputs):
        return self.func(*inputs)


def run_with_cell(fn):
    if fn is None:
        raise ValueError("fn cannot be none!")

    @wraps(fn)
    def wrapper(*args):
        cell_obj = Net(fn)
        return cell_obj(*args)

    return wrapper


def to_cell_obj(fn):
    cell_obj = Net(fn)
    return cell_obj


def compare(output, expect):
    '''
    :param output: Tensor, including tuple/list of tensor
    :param expect: Numpy array, including tuple/list of Numpy array
    :return:
    '''
    if isinstance(output, (tuple, list)):
        for o_ele, e_ele in zip(output, expect):
            compare(o_ele, e_ele)
    else:
        if expect.dtype == np.float32:
            rtol, atol = 1e-4, 1e-4
        else:
            rtol, atol = 1e-3, 1e-3
        assert np.allclose(output.asnumpy(), expect, rtol, atol)


def get_inputs_np(shapes, dtypes):
    np.random.seed(10)
    inputs_np = []
    for shape, dtype in zip(shapes, dtypes):
        inputs_np.append(np.random.randn(*shape).astype(dtype))
    return inputs_np


def get_inputs_tensor(inputs_np):
    inputs = []
    for input_np in inputs_np:
        inputs.append(Tensor(input_np))
    return inputs
