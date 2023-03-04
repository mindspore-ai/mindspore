# Copyright 2022 Huawei Technologies Co., Ltd
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
"""test vmap in graph mode"""

import platform
import pytest
import numpy as np
import mindspore.nn as nn
import mindspore.numpy as mnp
import mindspore.context as context
import mindspore.ops.operations as P
import mindspore.ops.functional as F
from mindspore import dtype as mstype
from mindspore.common import Tensor
from mindspore.ops.functional import vmap
from mindspore.common.api import jit
from mindspore.common.parameter import Parameter

context.set_context(mode=context.GRAPH_MODE)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_vmap_cond():
    """
    Feature: vmap
    Description: This case mainly tests the following `vmap` application scenarios in graph mode:
        1. The `fn` is a `Cell`, which contains control flow operators, such as `if` and `while`.
        2. The specific VmapRule of `Switch` and `Add` operation.
        3. The `in_axes` is a single integer, which automatically match to multiple arguments.
    Expectation: success
    """
    class CondNet(nn.Cell):
        def __init__(self):
            super(CondNet, self).__init__()
            self.inner_tensor_a = Tensor(2, mstype.int32)
            self.inner_tensor_b = Tensor(5, mstype.int32)

        def construct(self, x, y):
            a = self.inner_tensor_a + 1
            b = self.inner_tensor_b
            if a < b:
                b += a
            else:
                b -= a
            b += 5
            i = 0
            while i < 4:
                x += 1
                i += 1
            out = b + x + y
            return out

    x_hat = Tensor([2, 3, 1], mstype.int32)
    y_hat = Tensor([5, 4, 3], mstype.int32)
    result = vmap(CondNet(), 0, 0)(x_hat, y_hat)
    expect_result = Tensor([24, 24, 21], mstype.int32)
    assert np.allclose(result.asnumpy(), expect_result.asnumpy())


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_vmap_gradient():
    """
    Feature: vmap
    Description: This case mainly tests the following `vmap` application scenarios in graph mode:
        1. `vmap` and `grad` are used in combination.
        2. `vmap` and `jvp` are used in combination.
    Expectation: success
    """
    def forward_fn(x, y):
        out = x + 2 * y
        out = F.sin(out)
        return F.reduce_sum(out)

    class GradNet(nn.Cell):
        def __init__(self, fn):
            super(GradNet, self).__init__()
            self.fn = fn

        def construct(self, x, y):
            out = F.grad(self.fn, grad_position=(0, 1))(x, y)
            return out

    def vmap_fn(x, y):
        output = vmap(forward_fn, 1, 0)(x, y)
        return F.reduce_sum(output)

    def jvp_fn(x, y, v):
        out = F.jvp(forward_fn, (x, y), (v, v))
        return out

    x_hat = Tensor([[1., 2., 3.], [2., 3., 4.]], mstype.float32)
    y_hat = Tensor([[2., 3., 4.], [3., 4., 5.]], mstype.float32)
    expect_x_grad = Tensor([[0.28366217, -0.14550003, 0.0044257],
                            [-0.14550003, 0.0044257, 0.13673723]], mstype.float32)
    expect_y_grad = Tensor([[0.56732434, -0.29100007, 0.0088514],
                            [-0.29100007, 0.0088514, 0.27347445]], mstype.float32)

    vmap_grad_x, vmap_grad_y = vmap(GradNet(forward_fn), 1, 1)(x_hat, y_hat)
    assert np.allclose(vmap_grad_x.asnumpy(), expect_x_grad.asnumpy(), 0.0001, 0.0001)
    assert np.allclose(vmap_grad_y.asnumpy(), expect_y_grad.asnumpy(), 0.0001, 0.0001)

    grad_vmap_x, grad_vmap_y = GradNet(vmap_fn)(x_hat, y_hat)
    assert np.allclose(grad_vmap_x.asnumpy(), expect_x_grad.asnumpy(), 0.0001, 0.0001)
    assert np.allclose(grad_vmap_y.asnumpy(), expect_y_grad.asnumpy(), 0.0001, 0.0001)

    x_hat = Tensor(np.array([[1.], [2.], [3.]]), mstype.float32)
    y_hat = Tensor(np.array([[1.], [2.], [3.]]), mstype.float32)
    v_hat = Tensor(np.array([[1.], [2.], [3.]]), mstype.float32)

    vmap_jvp_x, vmap_jvp_y = vmap(jvp_fn, 0, 0)(x_hat, y_hat, v_hat)
    expect_x_jvp = Tensor([0.141120002, -0.279415488, 0.412118465], mstype.float32)
    expect_y_jvp = Tensor([-2.96997738, 5.76102161, -8.20017242], mstype.float32)
    assert np.allclose(vmap_jvp_x.asnumpy(), expect_x_jvp.asnumpy(), 0.0001, 0.0001)
    assert np.allclose(vmap_jvp_y.asnumpy(), expect_y_jvp.asnumpy(), 0.0001, 0.0001)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_vmap_monad():
    """
    Feature: vmap
    Description: This case mainly tests the following `vmap` application scenarios in graph mode:
        1. The `fn` is a `Cell`, which contains side effect operators, such as `AssignAdd`, `Assign`,
        `Print`, `ScatterAdd`.
        2. Parameter as argument.
    Expectation: success
    """
    class AssignNet(nn.Cell):
        def __init__(self):
            super(AssignNet, self).__init__()
            self.assign = P.Assign()
            self.assign_add = P.AssignAdd()
            self.scatter_add = P.ScatterAdd()
            self.assign_ref = Parameter(Tensor([[0, 0, 0], [1, 1, 1]], mstype.float32), name='assign_ref')
            self.replace_tensor = Tensor([[1, 1, 1], [2, 2, 2]], mstype.float32)

        def construct(self, assign_add_val, assign_add_var, scatter_ref, indices, updates):
            self.assign(self.assign_ref, self.replace_tensor)
            F.print(self.assign_ref)
            self.assign_add(assign_add_var, assign_add_val)
            out = assign_add_var + self.scatter_add(scatter_ref, indices, updates)
            return out

    class VmapMonadNet(nn.Cell):
        def __init__(self, net):
            super(VmapMonadNet, self).__init__()
            self.net = net
            self.assign_add_var = Parameter(
                Tensor([[[1, 1, 1], [1, 1, 1], [1, 1, 1]], [[2, 2, 2], [2, 2, 2], [2, 2, 2]]], mstype.float32),
                name='assign_add_var')
            self.scatter_ref = Parameter(
                Tensor([[[0, 0, 0], [0, 0, 0]], [[1, 1, 1], [1, 1, 1]], [[2, 2, 2], [2, 2, 2]]], mstype.float32),
                name='scatter_ref')

        def construct(self, assign_add_val, scatter_indices, scatter_updates):
            output = vmap(self.net, (0, 1, 0, 0, None), 1)(assign_add_val, self.assign_add_var,
                                                           self.scatter_ref, scatter_indices, scatter_updates)
            return output, self.assign_add_var

    assign_add_val = Tensor([[[1, 1, 1], [2, 2, 2]], [[1, 1, 1], [2, 2, 2]], [[1, 1, 1], [2, 2, 2]]], mstype.float32)
    scatter_indices = Tensor([[[0, 1], [1, 1]], [[0, 1], [0, 1]], [[1, 1], [1, 0]]], mstype.int32)
    scatter_updates = Tensor([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]], mstype.int32)
    output, assign_add_var = VmapMonadNet(AssignNet())(assign_add_val, scatter_indices, scatter_updates)

    expect_output = Tensor([[[3, 3, 3], [7, 7, 7], [8, 8, 8]], [[13, 13, 13], [11, 11, 11], [12, 12, 12]]],
                           mstype.float32)
    expect_assign_add_var = Tensor([[[2, 2, 2], [2, 2, 2], [2, 2, 2]], [[4, 4, 4], [4, 4, 4], [4, 4, 4]]],
                                   mstype.float32)
    assert np.allclose(output.asnumpy(), expect_output.asnumpy())
    assert np.allclose(assign_add_var.asnumpy(), expect_assign_add_var.asnumpy())


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_vmap_reduce():
    """
    Feature: vmap
    Description: This case mainly tests the following `vmap` application scenarios in graph mode:
        1. The specific VmapRule of `ReduceSum` operation.
        2. The `out_axes` is a single integer, which automatically match to multiple outputs.
    Expectation: success
    """
    class ReduceNet(nn.Cell):
        def __init__(self):
            super(ReduceNet, self).__init__()
            self.reduce_sum = P.ReduceSum(keep_dims=False)
            self.reduce_sum_keep_dims = P.ReduceSum(keep_dims=True)

        def construct(self, x):
            out1 = self.reduce_sum(x)
            out2 = self.reduce_sum_keep_dims(x)
            out3 = self.reduce_sum(x, 1)
            out4 = self.reduce_sum_keep_dims(x, 1)
            out5 = self.reduce_sum(x, (0, 1))
            out6 = self.reduce_sum_keep_dims(x, (0, 1))
            output = (out1, out2, out3, out4, out5, out6)
            return output

    class VmapNet(nn.Cell):
        def __init__(self, net):
            super(VmapNet, self).__init__()
            self.net = net

        def construct(self, x):
            vmap_function = F.vmap(self.net, 1, 0)
            output = vmap_function(x)
            return output

    x_hat = Tensor(np.array([[[[1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2], [3, 3, 3, 3, 3, 3]],
                              [[4, 4, 4, 4, 4, 4], [5, 5, 5, 5, 5, 5], [6, 6, 6, 6, 6, 6]],
                              [[7, 7, 7, 7, 7, 7], [8, 8, 8, 8, 8, 8], [9, 9, 9, 9, 9, 9]]],
                             [[[1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2], [3, 3, 3, 3, 3, 3]],
                              [[4, 4, 4, 4, 4, 4], [5, 5, 5, 5, 5, 5], [6, 6, 6, 6, 6, 6]],
                              [[7, 7, 7, 7, 7, 7], [8, 8, 8, 8, 8, 8], [9, 9, 9, 9, 9, 9]]],
                             [[[1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2], [3, 3, 3, 3, 3, 3]],
                              [[4, 4, 4, 4, 4, 4], [5, 5, 5, 5, 5, 5], [6, 6, 6, 6, 6, 6]],
                              [[7, 7, 7, 7, 7, 7], [8, 8, 8, 8, 8, 8], [9, 9, 9, 9, 9, 9]]]]), mstype.float32)

    result1, result2, result3, result4, result5, result6 = VmapNet(ReduceNet())(x_hat)
    expect_result1 = Tensor([108, 270, 432], mstype.float32)
    assert np.allclose(result1.asnumpy(), expect_result1.asnumpy())
    expect_result2 = Tensor([[[[108]]], [[[270]]], [[[432]]]], mstype.float32)
    assert np.allclose(result2.asnumpy(), expect_result2.asnumpy())
    expect_result3 = Tensor([[[6, 6, 6, 6, 6, 6], [6, 6, 6, 6, 6, 6], [6, 6, 6, 6, 6, 6]],
                             [[15, 15, 15, 15, 15, 15], [15, 15, 15, 15, 15, 15], [15, 15, 15, 15, 15, 15]],
                             [[24, 24, 24, 24, 24, 24], [24, 24, 24, 24, 24, 24], [24, 24, 24, 24, 24, 24]]],
                            mstype.float32)
    assert np.allclose(result3.asnumpy(), expect_result3.asnumpy())
    expect_result4 = Tensor([[[[6, 6, 6, 6, 6, 6]], [[6, 6, 6, 6, 6, 6]], [[6, 6, 6, 6, 6, 6]]],
                             [[[15, 15, 15, 15, 15, 15]], [[15, 15, 15, 15, 15, 15]], [[15, 15, 15, 15, 15, 15]]],
                             [[[24, 24, 24, 24, 24, 24]], [[24, 24, 24, 24, 24, 24]], [[24, 24, 24, 24, 24, 24]]]],
                            mstype.float32)
    assert np.allclose(result4.asnumpy(), expect_result4.asnumpy())
    expect_result5 = Tensor([[18, 18, 18, 18, 18, 18], [45, 45, 45, 45, 45, 45], [72, 72, 72, 72, 72, 72]],
                            mstype.float32)
    assert np.allclose(result5.asnumpy(), expect_result5.asnumpy())
    expect_result6 = Tensor([[[[18, 18, 18, 18, 18, 18]]], [[[45, 45, 45, 45, 45, 45]]], [[[72, 72, 72, 72, 72, 72]]]],
                            mstype.float32)
    assert np.allclose(result6.asnumpy(), expect_result6.asnumpy())


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_vmap_general_rule():
    """
    Feature: vmap
    Description: This case mainly tests the following `vmap` application scenarios in graph mode:
        1. The general VmapRule.
        2. The specific VmapRule of `Reshape` operation.
        3. The same `vmap` object is called multiple times.
        4. The `mindspore.numpy` objects as the arguments.
    Expectation: success
    """
    def convolve(x, w):
        output = []
        for i in range(1, len(x) - 1):
            output.append(mnp.dot(x[i - 1 : i + 2], w))
        return mnp.stack(output)

    x = mnp.arange(5).astype('float32')
    w = mnp.array([1., 2., 3.])
    vmap_function = vmap(convolve)

    x1 = mnp.stack([x, x, x])
    w1 = mnp.stack([w, w, w])
    result1 = vmap_function(x1, w1)
    expect_result1 = Tensor([[8, 14, 20], [8, 14, 20], [8, 14, 20]], mstype.float32)
    assert np.allclose(result1.asnumpy(), expect_result1.asnumpy())

    x2 = mnp.stack([x, x + 1, x + 2])
    w2 = mnp.stack([w, w * 2, w * 3])
    result2 = vmap_function(x2, w2)
    expect_result2 = Tensor([[8, 14, 20], [28, 40, 52], [60, 78, 96]], mstype.float32)
    assert np.allclose(result2.asnumpy(), expect_result2.asnumpy())


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_vmap_nested_axes():
    """
    Feature: vmap
    Description: This case mainly tests the following `vmap` application scenarios in graph mode:
        1. The nested inputs as the vmap's arguments.
        2. One element of the `in_axes` is a minus integer.
        3. Some outputs of the function is scalars with destination axis non-None.
        4. The `in_axes` is nested Tuple and List.
        5. VmapRule for that operators with indefinite length as input, such as `Stack`.
    Expectation: success
    """
    class AddNet(nn.Cell):
        def __init__(self):
            super(AddNet, self).__init__()
            self.inner_tensor = Tensor([5, 6], mstype.float32)
            self.inner_para = Parameter(Tensor([5, 6], mstype.float32), name='inner_para')

        def construct(self, x, y):
            a = 1
            b = 2
            c = 3
            d = self.inner_tensor + a
            e = F.stack((self.inner_para, self.inner_para))
            return ((a, b), c), d, e

    x_hat = Tensor([[1, 2, 3], [4, 5, 6]], mstype.float32)
    y_hat = Tensor([[1, 2, 3], [4, 5, 6]], mstype.float32)
    z_hat = 1

    ((res1, res2), res3), res4, res5 = \
        vmap(AddNet(), in_axes=(1, [-1, None]), out_axes=((0, None), 0, None))(x_hat, (y_hat, z_hat))
    expect_res1 = Tensor([1, 1, 1], mstype.float32)
    expect_res2 = Tensor([2, 2, 2], mstype.float32)
    expect_res3 = 3
    expect_res4 = Tensor([[6, 7], [6, 7], [6, 7]], mstype.float32)
    expect_res5 = Tensor([[5, 6], [5, 6]], mstype.float32)

    assert np.allclose(res1.asnumpy(), expect_res1.asnumpy())
    assert np.allclose(res2.asnumpy(), expect_res2.asnumpy())
    assert res3 == expect_res3
    assert np.allclose(res4.asnumpy(), expect_res4.asnumpy())
    assert np.allclose(res5.asnumpy(), expect_res5.asnumpy())


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_vmap_with_tuple_input():
    """
    Feature: vmap
    Description: When vmap use tuple inputs in graph, it must ensure the inputs is not eliminated.
    Expectation: success
    """
    def real_fn(x, y):
        return x * y

    def foo(fn):
        @jit
        def wrapped(*args):
            def fn2(x, y):
                return F.jvp(fn, x, y)
            res = F.vmap(fn2)(args, args)
            return res
        return wrapped

    shape = (2, 3)
    a = F.ones(shape, mstype.int32)
    b = F.ones(shape, mstype.int32) * 2
    res = foo(real_fn)(a, b)

    assert isinstance(res, tuple)
    assert len(res) == 2
    assert isinstance(res[0], Tensor)
    assert isinstance(res[1], Tensor)
    assert np.allclose(res[0].asnumpy(), np.array([[2, 2, 2], [2, 2, 2]]))
    assert np.allclose(res[1].asnumpy(), np.array([[4, 4, 4], [4, 4, 4]]))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_vmap_with_celllist_input():
    """
    Feature: vmap
    Description: When vmap use celllist inputs in graph, it is executing the model ensembling parallel scenario.
    Expectation: success
    """
    class AssignNet(nn.Cell):
        def __init__(self):
            super(AssignNet, self).__init__()
            self.assign = P.Assign()
            self.ref_a = Parameter(Tensor([0, 1, 2], mstype.float32), name='ref_a')
            self.ref_b = Parameter(Tensor([0, 1, 2], mstype.float32), name='ref_b')

        def construct(self, replace_tensor):
            self.assign(self.ref_a, replace_tensor)
            out = self.ref_b + self.ref_a
            return out

    if platform.system() == "Linux":
        m1 = AssignNet()
        m2 = AssignNet()
        m3 = AssignNet()
        mm = nn.CellList([m1, m2, m3])
        replace_tensor = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], mstype.float32)

        output = F.vmap(mm, 0)(replace_tensor)

        expect_res1 = Tensor([[1, 3, 5], [4, 6, 8], [7, 9, 11]], mstype.float32)
        expect_res2 = Tensor([1, 2, 3], mstype.float32)
        expect_res3 = Tensor([4, 5, 6], mstype.float32)
        expect_res4 = Tensor([7, 8, 9], mstype.float32)
        expect_res5 = Tensor([0, 1, 2], mstype.float32)

        assert np.allclose(output.asnumpy(), expect_res1.asnumpy())
        assert np.allclose(m1.ref_a.asnumpy(), expect_res2.asnumpy())
        assert np.allclose(m2.ref_a.asnumpy(), expect_res3.asnumpy())
        assert np.allclose(m3.ref_a.asnumpy(), expect_res4.asnumpy())
        assert np.allclose(m1.ref_b.asnumpy(), expect_res5.asnumpy())
        assert np.allclose(m2.ref_b.asnumpy(), expect_res5.asnumpy())
        assert np.allclose(m3.ref_b.asnumpy(), expect_res5.asnumpy())
    else:
        pass


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_vmap_as_vmap_input():
    """
    Feature: vmap
    Description: When the output of a vmap function is used as the input of another vmap function.
    Expectation: success
    """
    class VmapNet(nn.Cell):
        def __init__(self):
            super(VmapNet, self).__init__()
            self.tensor = Tensor(np.ones((3, 4), dtype=int), mstype.float32)
            self.matmul_vmap = F.vmap(F.matmul, in_axes=(1, None), out_axes=1)
            self.relu_vmap = F.vmap(nn.ReLU(), in_axes=1, out_axes=1)

        def construct(self, x):
            x = self.matmul_vmap(x, self.tensor)
            x = self.relu_vmap(x)
            return x

    x = Tensor(np.ones((4, 4, 3), dtype=int), mstype.float32)
    output = VmapNet()(x)
    expect_res = Tensor(np.ones((4, 4, 4), dtype=int), mstype.float32) * 3
    assert np.allclose(output.asnumpy(), expect_res.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_vmap_with_celllist_nested_grad():
    """
    Feature: vmap
    Description: This case mainly tests the following `vmap` application scenarios in graph mode:
        1. `vmap` and `grad` are used in combination.
        2. `vmap` accepts celllist type inputs.
    Expectation: success
    """
    class AssignNet(nn.Cell):
        def __init__(self):
            super(AssignNet, self).__init__()
            self.assign = P.Assign()
            self.ref_a = Parameter(Tensor([0, 1, 2], mstype.float32), name='ref_a')

        def construct(self, replace_tensor):
            replace_tensor = replace_tensor * 2
            self.assign(self.ref_a, replace_tensor)
            out = self.ref_a + replace_tensor
            return out

    if platform.system() == "Linux":
        m1 = AssignNet()
        m2 = AssignNet()
        m3 = AssignNet()
        mm = nn.CellList([m1, m2, m3])
        vmap_net = F.vmap(mm)

        replace_tensor = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], mstype.float32)

        output_grad = F.grad(vmap_net)(replace_tensor)

        expect_res = Tensor([[2, 2, 2], [2, 2, 2], [2, 2, 2]], mstype.float32)
        assert np.allclose(output_grad.asnumpy(), expect_res.asnumpy())
    else:
        pass
