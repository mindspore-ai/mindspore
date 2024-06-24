# Copyright 2024 Huawei Technologies Co., Ltd
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
from tests.mark_utils import arg_mark
import pytest
import numpy as np
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP
import mindspore as ms
from mindspore import mint

def generate_random_input(shape, dtype):
    return np.random.randint(1, 10, size=shape).astype(dtype)

def pad_constant_func(x, padding, value=0.0):
    return mint.nn.functional.pad(x, padding, 'constant', value)

def pad_reflect_func(x, padding):
    return mint.nn.functional.pad(x, padding, 'reflect')

def pad_replicate_func(x, padding):
    return mint.nn.functional.pad(x, padding, 'replicate')

@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_ops_pad_constant_dynamic():
    """
    Feature: pyboost function.
    Description: dynamic test for mint.nn.functional.pad. mode = "constant".
    Expectation: expect correct result.
    """
    input1 = generate_random_input((2, 3), np.float32)
    padding1 = (1, 1)
    value1 = 0

    input2 = generate_random_input((2, 3, 4), np.float32)
    padding2 = (1, 2)
    value2 = 1

    TEST_OP(pad_constant_func, [[ms.Tensor(input1), padding1, value1], [ms.Tensor(input2), padding2, value2]],
            'constant_pad_nd', disable_mode=['GRAPH_MODE'], disable_nontensor_dynamic_type='MUTABLE_LEN',
            disable_yaml_check=True)

@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_ops_pad_reflect_1d_dynamic():
    """
    Feature: pyboost function.
    Description: dynamic test for mint.nn.functional.pad. reflection_pad_1d.
    Expectation: expect correct result.
    """
    input1 = generate_random_input((2, 3), np.float32)
    padding1 = (1, 1)

    input2 = generate_random_input((2, 3, 4), np.float32)
    padding2 = (1, 2)

    TEST_OP(pad_reflect_func, [[ms.Tensor(input1), padding1], [ms.Tensor(input2), padding2]],
            'reflection_pad_1d', disable_mode=['GRAPH_MODE'], disable_nontensor_dynamic_type='MUTABLE_LEN',
            disable_yaml_check=True)

@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_ops_pad_reflect_2d_dynamic():
    """
    Feature: pyboost function.
    Description: dynamic test for mint.nn.functional.pad. reflection_pad_2d.
    Expectation: expect correct result.
    """
    input1 = generate_random_input((2, 4, 3), np.float32)
    padding1 = (1, 1, 1, 2)

    input2 = generate_random_input((2, 3, 4, 4), np.float32)
    padding2 = (1, 2, 1, 1)

    TEST_OP(pad_reflect_func, [[ms.Tensor(input1), padding1], [ms.Tensor(input2), padding2]],
            'reflection_pad_2d', disable_mode=['GRAPH_MODE'], disable_nontensor_dynamic_type='MUTABLE_LEN',
            disable_yaml_check=True)

@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_ops_pad_reflect_3d_dynamic():
    """
    Feature: pyboost function.
    Description: dynamic test for mint.nn.functional.pad. reflection_pad_3d.
    Expectation: expect correct result.
    """
    input1 = generate_random_input((2, 3, 4, 4), np.float32)
    padding1 = (1, 1, 1, 2, 1, 1)

    input2 = generate_random_input((2, 3, 3, 4, 4), np.float32)
    padding2 = (1, 1, 1, 2, 1, 2)

    TEST_OP(pad_reflect_func, [[ms.Tensor(input1), padding1], [ms.Tensor(input2), padding2]],
            'reflection_pad_3d', disable_mode=['GRAPH_MODE'], disable_nontensor_dynamic_type='MUTABLE_LEN',
            disable_yaml_check=True)

@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_ops_pad_replicate_1d_dynamic():
    """
    Feature: pyboost function.
    Description: dynamic test for mint.nn.functional.pad. replication_pad_1d.
    Expectation: expect correct result.
    """
    input1 = generate_random_input((2, 3), np.float32)
    padding1 = (1, 1)

    input2 = generate_random_input((2, 3, 4), np.float32)
    padding2 = (1, 2)

    TEST_OP(pad_replicate_func, [[ms.Tensor(input1), padding1], [ms.Tensor(input2), padding2]],
            'replication_pad_1d', disable_mode=['GRAPH_MODE'], disable_nontensor_dynamic_type='MUTABLE_LEN',
            disable_yaml_check=True)

@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_ops_pad_replicate_2d_dynamic():
    """
    Feature: pyboost function.
    Description: dynamic test for mint.nn.functional.pad. replication_pad_2d.
    Expectation: expect correct result.
    """
    input1 = generate_random_input((2, 4, 3), np.float32)
    padding1 = (1, 1, 1, 2)

    input2 = generate_random_input((2, 3, 4, 4), np.float32)
    padding2 = (1, 2, 1, 1)

    TEST_OP(pad_replicate_func, [[ms.Tensor(input1), padding1], [ms.Tensor(input2), padding2]],
            'replication_pad_2d', disable_mode=['GRAPH_MODE'], disable_nontensor_dynamic_type='MUTABLE_LEN',
            disable_yaml_check=True)

@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_ops_pad_replicate_3d_dynamic():
    """
    Feature: pyboost function.
    Description: dynamic test for mint.nn.functional.pad. replication_pad_3d.
    Expectation: expect correct result.
    """
    input1 = generate_random_input((2, 3, 4, 4), np.float32)
    padding1 = (1, 1, 1, 2, 1, 1)

    input2 = generate_random_input((2, 3, 3, 4, 4), np.float32)
    padding2 = (1, 1, 1, 2, 1, 2)

    TEST_OP(pad_replicate_func, [[ms.Tensor(input1), padding1], [ms.Tensor(input2), padding2]],
            'replication_pad_3d', disable_mode=['GRAPH_MODE'], disable_nontensor_dynamic_type='MUTABLE_LEN',
            disable_yaml_check=True)
