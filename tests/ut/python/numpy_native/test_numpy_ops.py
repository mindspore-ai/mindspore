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
""" test numpy ops """
import numpy as np

import mindspore.numpy as mnp
from mindspore import Tensor
from mindspore.nn import Cell
import mindspore.context as context
from ....mindspore_test_framework.mindspore_test import mindspore_test
from ....mindspore_test_framework.pipeline.forward.compile_forward \
    import pipeline_for_compile_forward_ge_graph_for_case_by_case_config

context.set_context(mode=context.GRAPH_MODE)

class MeshGrid(Cell):
    def construct(self, a, b, c, d):
        ret = mnp.meshgrid(a, b, c, d)
        return ret


class Choose(Cell):
    def construct(self, a, b):
        ret = mnp.choose(a, b)
        return ret


class Histogram(Cell):
    def construct(self, a):
        ret = mnp.histogram(a)
        return ret


class Norm(Cell):
    def construct(self, a):
        ret = mnp.norm(a)
        return ret


class Cross(Cell):
    def construct(self, a, b):
        ret = mnp.cross(a, b)
        return ret


class Stack(Cell):
    def construct(self, a, b):
        ret = mnp.stack((a, b))
        return ret


class Correlate(Cell):
    def construct(self, a, b):
        ret = mnp.correlate(a, b)
        return ret


class Split(Cell):
    def construct(self, tensor):
        a = mnp.split(tensor, indices_or_sections=1)
        b = mnp.split(tensor, indices_or_sections=3)
        c = mnp.array_split(tensor, indices_or_sections=1)
        d = mnp.array_split(tensor, indices_or_sections=3, axis=-1)
        return a, b, c, d


class MatrixPower(Cell):
    def construct(self, tensor):
        a = mnp.matrix_power(tensor, 3)
        return a


class RavelMultiIndex(Cell):
    def construct(self, tensor):
        a = mnp.ravel_multi_index(tensor, (7, 6))
        b = mnp.ravel_multi_index(tensor, (7, 6), order='F')
        c = mnp.ravel_multi_index(tensor, (4, 6), mode='clip')
        d = mnp.ravel_multi_index(tensor, (4, 4), mode='wrap')
        return a, b, c, d


class GeomSpace(Cell):
    def construct(self, start):
        a = mnp.geomspace(1, 256, num=9)
        b = mnp.geomspace(1, 256, num=8, endpoint=False)
        c = mnp.geomspace(start, [1000, 2000, 3000], num=4)
        d = mnp.geomspace(start, [1000, 2000, 3000], num=4, endpoint=False, axis=-1)
        return a, b, c, d


class Arange(Cell):
    def construct(self):
        a = mnp.arange(10)
        b = mnp.arange(0, 10)
        c = mnp.arange(0.1, 9.9)
        return a, b, c


class Eye(Cell):
    def construct(self):
        res = []
        for n in range(1, 5):
            for k in range(0, 5):
                res.append(mnp.eye(10, n, k))
        return res


class Trace(Cell):
    def construct(self, arr):
        a = mnp.trace(arr, offset=-1, axis1=0, axis2=1)
        b = mnp.trace(arr, offset=0, axis1=1, axis2=0)
        return a, b


class Pad(Cell):
    def construct(self, arr1, arr2):
        a = mnp.pad(arr1, ((1, 1), (2, 2), (3, 4)))
        b = mnp.pad(arr1, ((1, 1), (2, 2), (3, 4)), mode="mean", stat_length=((1, 2), (2, 10), (3, 4)))
        c = mnp.pad(arr1, ((1, 1), (2, 2), (3, 4)), mode="edge")
        d = mnp.pad(arr1, ((1, 1), (2, 2), (3, 4)), mode="wrap")
        e = mnp.pad(arr1, ((1, 3), (5, 2), (3, 0)), mode="linear_ramp", end_values=((0, 10), (9, 1), (-10, 99)))
        f = mnp.pad(arr2, ((10, 13), (5, 12), (3, 0), (2, 6)), mode='symmetric', reflect_type='even')
        g = mnp.pad(arr2, ((10, 13)), mode='reflect', reflect_type='even')
        return a, b, c, d, e, f, g


class Where(Cell):
    def construct(self, a, b, c):
        ret = mnp.where(a, b, c)
        return ret


class Select(Cell):
    def construct(self, a, b):
        ret = mnp.select(a, b)
        return ret


class IsClose(Cell):
    def construct(self, a, b):
        ret = mnp.isclose(a, b)
        return ret


class ArgMax(Cell):
    def construct(self, a):
        ret = mnp.argmax(a)
        return ret


class Average(Cell):
    def construct(self, a):
        ret = mnp.average(a)
        return ret


class Remainder(Cell):
    def construct(self, a, b):
        ret = mnp.remainder(a, b)
        return ret


class Diff(Cell):
    def construct(self, a):
        ret1 = mnp.diff(a)
        ret2 = mnp.ediff1d(a)
        return ret1, ret2


class Trapz(Cell):
    def construct(self, arr):
        a = mnp.trapz(arr, x=[-2, 1, 2], axis=1)
        b = mnp.trapz(arr, dx=3, axis=0)
        return a, b


class Lcm(Cell):
    def construct(self, a, b):
        ret = mnp.lcm(a, b)
        return ret


class Cov(Cell):
    def construct(self, a):
        ret = mnp.cov(a, a)
        return ret


class Gradient(Cell):
    def construct(self, a):
        ret = mnp.gradient(a)
        return ret


class MultiDot(Cell):
    def construct(self, a, b, c, d):
        ret = mnp.multi_dot((a, b, c, d))
        return ret


class Histogramdd(Cell):
    def construct(self, a):
        ret = mnp.histogramdd(a)
        return ret


test_cases = [
    ('MeshGrid', {
        'block': MeshGrid(),
        'desc_inputs': [Tensor(np.full(3, 2, dtype=np.float32)),
                        Tensor(np.full(1, 5, dtype=np.float32)),
                        Tensor(np.full((2, 3), 9, dtype=np.float32)),
                        Tensor(np.full((4, 5, 6), 7, dtype=np.float32))],
    }),
    ('Norm', {
        'block': Norm(),
        'desc_inputs': [Tensor(np.ones((5, 2, 3, 7), dtype=np.float32))],
    }),
    ('Cross', {
        'block': Cross(),
        'desc_inputs': [Tensor(np.arange(18, dtype=np.int32).reshape(2, 3, 1, 3)),
                        Tensor(np.arange(9, dtype=np.int32).reshape(1, 3, 3))],
    }),
    ('Stack', {
        'block': Stack(),
        'desc_inputs': [Tensor(np.arange(9, dtype=np.int32).reshape(3, 3)),
                        Tensor(np.arange(9, dtype=np.int32).reshape(3, 3)),],
    }),
    ('Correlate', {
        'block': Correlate(),
        'desc_inputs': [Tensor(np.array([1, 2, 3, 4, 5], dtype=np.int32)),
                        Tensor(np.array([0, 1], dtype=np.int32)),],
    }),
    ('Split', {
        'block': Split(),
        'desc_inputs': [Tensor(np.arange(9, dtype=np.float32).reshape(3, 3))],
    }),
    ('MatrixPower', {
        'block': MatrixPower(),
        'desc_inputs': [Tensor(np.arange(9, dtype=np.float32).reshape(3, 3))],
    }),
    ('RavelMultiIndex', {
        'block': RavelMultiIndex(),
        'desc_inputs': [Tensor(np.array([[3, 6, 6], [4, 5, 1]], dtype=np.int32))],
    }),
    ('GeomSpace', {
        'block': GeomSpace(),
        'desc_inputs': [Tensor(np.arange(1, 7, dtype=np.float32).reshape(2, 3))],
    }),
    ('Arange', {
        'block': Arange(),
        'desc_inputs': [],
    }),
    ('Eye', {
        'block': Eye(),
        'desc_inputs': [],
    }),
    ('Trace', {
        'block': Trace(),
        'desc_inputs': [Tensor(np.ones((3, 5), dtype=np.float32))],
    }),
    ('Where', {
        'block': Where(),
        'desc_inputs': [Tensor(np.full((1, 1, 2), [False, True])),
                        Tensor(np.full((1, 3, 2), 5, dtype=np.float32)),
                        Tensor(np.full((2, 1, 1), 7, dtype=np.float32))],
    }),
    ('Select', {
        'block': Select(),
        'desc_inputs': [Tensor([[True, True, True, False, False], [False, False, True, False, True]]),
                        Tensor(np.array([[0, 1, 2, 3, 4], [0, 1, 4, 9, 16]], dtype=np.int32))],
    }),
    ('IsClose', {
        'block': IsClose(),
        'desc_inputs': [Tensor(np.array([0, 1, 2, float('inf'), float('inf'), float('nan')], dtype=np.float32)),
                        Tensor(np.array([0, 1, -2, float('-inf'), float('inf'), float('nan')], dtype=np.float32))],
    }),
    ('ArgMax', {
        'block': ArgMax(),
        'desc_inputs': [Tensor(np.array([False, True]))],
    }),
    ('Average', {
        'block': Average(),
        'desc_inputs': [Tensor(np.array([[1., 2.], [3., 4.]], dtype=np.float32))],
    }),
    ('Remainder', {
        'block': Remainder(),
        'desc_inputs': [Tensor(np.array([4, 7], dtype=np.int32)),
                        Tensor(np.array([[1, 2], [3, 4]], dtype=np.int32))],
    }),
    ('Diff', {
        'block': Diff(),
        'desc_inputs': [Tensor(np.array([1, 3, -1, 0, 4], dtype=np.int32))],
    }),
    ('Trapz', {
        'block': Trapz(),
        'desc_inputs': [Tensor(np.arange(6, dtype=np.int32).reshape(2, 3))],
    }),
    ('Lcm', {
        'block': Lcm(),
        'desc_inputs': [Tensor(np.arange(6, dtype=np.int32)),
                        Tensor(np.array(20, dtype=np.int32))],
    }),
    ('Cov', {
        'block': Cov(),
        'desc_inputs': [Tensor(np.array([[2., 3., 4., 5.], [0., 2., 3., 4.], [7., 8., 9., 10.]], dtype=np.float32))],
    }),
    ('Gradient', {
        'block': Gradient(),
        'desc_inputs': [Tensor(np.array([[2., 3., 4., 5.], [0., 2., 3., 4.], [7., 8., 9., 10.]], dtype=np.float32))],
    }),
    ('MultiDot', {
        'block': MultiDot(),
        'desc_inputs': [Tensor(np.ones((10000, 100), dtype=np.float32)),
                        Tensor(np.ones((100, 1000), dtype=np.float32)),
                        Tensor(np.ones((1000, 5), dtype=np.float32)),
                        Tensor(np.ones((5, 333), dtype=np.float32))],
    }),
]


@mindspore_test(pipeline_for_compile_forward_ge_graph_for_case_by_case_config)
def test_exec():
    context.set_context(mode=context.GRAPH_MODE)
    return test_cases
