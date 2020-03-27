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
""" test_tensor_slice """
import numpy as np
import pytest

from mindspore import Tensor
from mindspore import context
from mindspore.nn import Cell

from ....mindspore_test_framework.mindspore_test import mindspore_test
from ....mindspore_test_framework.pipeline.forward.compile_forward \
    import pipeline_for_compile_forward_ge_graph_for_case_by_case_config


class NetWorkSlicePositive(Cell):
    def __init__(self):
        super(NetWorkSlicePositive, self).__init__()
        self.tensor_ret0 = Tensor(np.ones([1, 2, 2], np.int32))
        self.tensor_ret1 = Tensor(np.ones([4, 7, 4], np.int32))
        self.tensor_ret2 = Tensor(np.ones([6, 8, 10], np.int32))
        self.tensor_ret3 = Tensor(np.ones([3, 8, 10], np.int32))

    def construct(self, tensor):
        ret0 = tensor[3:4:3, 1:5:2, 3:6:2] + self.tensor_ret0
        ret1 = tensor[-6:4:1, 7:-8:-1, ::3] + self.tensor_ret1
        ret2 = tensor[::, ::, ::] + self.tensor_ret2
        ret3 = tensor[::2] + self.tensor_ret3
        return ret0, ret1, ret2, ret3


class NetWorkReduceDimension(Cell):
    def __init__(self):
        super(NetWorkReduceDimension, self).__init__()
        self.tensor_ret0 = Tensor(np.ones([2, 4, 1], np.int32))
        self.tensor_ret1 = Tensor(np.ones([3, 4], np.int32))
        self.tensor_ret2 = Tensor(np.ones([6, 8], np.int32))
        self.tensor_ret3 = Tensor(np.array(8, np.int32))
        self.tensor_ret4 = Tensor(np.ones([8, 10], np.int32))

    def construct(self, tensor):
        ret0 = tensor[0:6:3, 1:5:1, 3:5:2] + self.tensor_ret0
        ret1 = tensor[::2, 1, ::3] + self.tensor_ret1
        ret2 = tensor[::, ::, 0] + self.tensor_ret2
        ret3 = tensor[3, 2, 5] + self.tensor_ret3
        ret4 = tensor[1] + self.tensor_ret4
        return ret0, ret1, ret2, ret3, ret4


class NetWorkStepNegative(Cell):
    def __init__(self):
        super(NetWorkStepNegative, self).__init__()
        self.tensor_ret = Tensor(np.ones([6, 5, 10], np.int32))

    def construct(self, tensor):
        ret = tensor[::1, -5::, ::-1] + self.tensor_ret
        return ret


class NetWorkReduceToScalar(Cell):
    def __init__(self):
        super(NetWorkReduceToScalar, self).__init__()
        self.tensor_ret = Tensor(np.array(9, np.int32))

    def construct(self, tensor):
        ret = tensor[2, 3, 4] + self.tensor_ret
        return ret


test_cases = [
    ('SlicePositive', {
        'block': NetWorkSlicePositive(),
        'desc_inputs': [Tensor(np.ones([6, 8, 10], np.int32))],
    }),
    ('SliceReduceDimension', {
        'block': NetWorkReduceDimension(),
        'desc_inputs': [Tensor(np.ones([6, 8, 10], np.int32))],
    }),
    ('SliceNegative', {
        'block': NetWorkStepNegative(),
        'desc_inputs': [Tensor(np.ones([6, 8, 10], np.int32))],
    }),
    ('SliceReduceToScalar', {
        'block': NetWorkReduceToScalar(),
        'desc_inputs': [Tensor(np.ones([6, 8, 10], np.int32))],
    }),

]


@mindspore_test(pipeline_for_compile_forward_ge_graph_for_case_by_case_config)
def test_compile():
    context.set_context(mode=context.GRAPH_MODE)
    return test_cases


def test_tensor_slice_reduce_out_of_bounds_neg():
    class NetWork(Cell):
        def __init__(self):
            super(NetWork, self).__init__()
            self.tensor_ret = Tensor(np.array(9, np.int32))

        def construct(self, tensor):
            ret = tensor[-7, 3, 4]
            return ret

    input_tensor = Tensor(np.ones([6, 8, 10], np.int32))
    net = NetWork()
    with pytest.raises(ValueError) as ex:
        net(input_tensor)
    assert "The `begin[0]` should be an int and must greater or equal to -6, but got -7" in str(ex.value)


def test_tensor_slice_reduce_out_of_bounds_positive():
    class NetWork(Cell):
        def __init__(self):
            super(NetWork, self).__init__()
            self.tensor_ret = Tensor(np.array(9, np.int32))

        def construct(self, tensor):
            ret = tensor[6, 3, 4]
            return ret

    input_tensor = Tensor(np.ones([6, 8, 10], np.int32))
    net = NetWork()
    with pytest.raises(ValueError) as ex:
        net(input_tensor)
    assert "The `begin[0]` should be an int and must less than 6, but got 6" in str(ex.value)
