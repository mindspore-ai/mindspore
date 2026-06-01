# Copyright 2025 Huawei Technologies Co., Ltd
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
""" tests_custom_pyboost_cpu """

import os
import numpy as np
import mindspore as ms
from mindspore import Tensor
from mindspore._c_expression import _DisableMsDispatchMode
from mindspore.ops import CustomOpBuilder
from tests.mark_utils import arg_mark

RUN_OPS = []


def _local_ccsrc_include_path():
    test_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(test_dir, "../../../../.."))
    return os.path.join(repo_root, "mindspore", "ccsrc")


class Layout:
    pass


class DTensor(Tensor):
    def __new__(cls, local_tensor, layout):
        t = Tensor._make_subclass(cls, local_tensor)
        t._local_tensor = local_tensor
        t._layout = layout
        return t

    def __init__(self, local_tensor, layout):
        pass

    def asnumpy(self):
        return self._local_tensor.asnumpy()

    @classmethod
    def __ms_dispatch__(cls, func, args=(), kwargs=None):
        def unwrap_dtensor(x):
            if isinstance(x, DTensor):
                return x._local_tensor  # pylint: disable=protected-access
            if isinstance(x, (tuple, list)):
                return type(x)(unwrap_dtensor(item) for item in x)
            return x

        def wrap_dtensor(x):
            if isinstance(x, Tensor):
                return DTensor(x, Layout())
            if isinstance(x, (tuple, list)):
                return type(x)(wrap_dtensor(item) for item in x)
            return x

        RUN_OPS.append(func.name)
        if kwargs is None:
            kwargs = {}
        new_args = [unwrap_dtensor(arg) for arg in args]
        with _DisableMsDispatchMode():
            return wrap_dtensor(func(*new_args, **kwargs))


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard")
def test_pyboost_cpu_add():
    """
    Feature: CustomOpBuilder.
    Description: Custom add op.
    Expectation: success.
    """
    ms.set_device("CPU")
    my_ops = CustomOpBuilder("pyboost_cpu_add", ['jit_test_files/pyboost_cpu_add.cpp'], backend="CPU",
                             include_paths=[_local_ccsrc_include_path()]).load()
    x = np.random.randint(0, 100, size=(10, 10, 10)).astype(np.int32)
    y = np.random.randint(0, 100, size=(9, 9, 9)).astype(np.int32)
    # the sliced shape is [4, 4, 4]
    expect = x[:4, 6:, 3:7] + y[5:, :4, 4:8]

    x = ms.Tensor(x)
    x1 = x[:4, 6:, 3:7]
    y = ms.Tensor(y)
    y1 = y[5:, :4, 4:8]
    out1 = my_ops.add_uncontiguous(x1, y1)
    x2 = x1.contiguous()
    y2 = y1.contiguous()
    out2 = my_ops.add_contiguous(x2, y2)
    assert np.allclose(out1.asnumpy(), expect, 1e-3, 1e-3)
    assert np.allclose(out2.asnumpy(), expect, 1e-3, 1e-3)

    # case in tutoral
    x = np.array([1, 2], dtype=np.int32)
    y = np.array([3, 4], dtype=np.int32)
    z = np.array([5, 6], dtype=np.int32)
    out = my_ops.add3(ms.Tensor(x), ms.Tensor(y), ms.Tensor(z))
    expect = x + y + z
    assert np.allclose(out.asnumpy(), expect, 1e-3, 1e-3)

    out = my_ops.add4(ms.Tensor(x), ms.Tensor(y), [ms.Tensor(z)])
    assert np.allclose(out.asnumpy(), expect, 1e-3, 1e-3)

    RUN_OPS.clear()
    out = my_ops.add4(ms.Tensor(x), ms.Tensor(y), [ms.Tensor(z), DTensor(ms.Tensor(z), Layout())])
    assert isinstance(out, DTensor)
    assert np.allclose(out.asnumpy(), expect, 1e-3, 1e-3)
    assert RUN_OPS == ["CustomAdd4::Eval"]

    out = my_ops.add5(ms.Tensor(x), ms.Tensor(y), [[ms.Tensor(z)]])
    assert np.allclose(out.asnumpy(), expect, 1e-3, 1e-3)

    out = ms.mint.empty_like(ms.Tensor(x), device="cpu")
    my_ops.add6(ms.Tensor(x), ms.Tensor(y), ms.Tensor(z), [out])
    assert np.allclose(out.asnumpy(), expect, 1e-3, 1e-3)


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard")
def test_pyboost_cpu_swap():
    """
    Feature: CustomOpBuilder.
    Description: Custom swap x and y with workspace.
    Expectation: success.
    """

    ms.set_device("CPU")
    my_ops = CustomOpBuilder("pyboost_cpu_swap", ['jit_test_files/pyboost_cpu_swap.cpp'], backend="CPU").load()
    x = np.random.randint(0, 100, (10, 10, 10)).astype(np.int32)
    y = np.random.randint(0, 100, (10, 10, 10)).astype(np.int32)
    x_tensor = ms.Tensor(x)
    y_tensor = ms.Tensor(y)
    ret = my_ops.swap(x_tensor, y_tensor)
    assert ret is None
    assert np.allclose(x_tensor.asnumpy(), y, 1e-3, 1e-3)
    assert np.allclose(y_tensor.asnumpy(), x, 1e-3, 1e-3)
