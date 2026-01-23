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
""" tests_custom_pyboost_ascend """

import numpy as np
import tempfile
import mindspore as ms
from mindspore.ops import CustomOpBuilder
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=4.0)
def test_custom_single_operator():
    """
    Feature: Single-operator loading via CustomOpBuilder.
    Description: Build and execute the add4 operator defined in YAML and CPP files in GRAPH mode.
    Expectation: Execution succeeds and outputs match NumPy results.
    """
    ms.set_device("Ascend")
    ms.set_context(mode=ms.GRAPH_MODE, save_graphs=False, save_graphs_path="./graphs")

    with tempfile.TemporaryDirectory() as tmpdirname:
        class MyNet(ms.nn.Cell):
            def __init__(self):
                super(MyNet, self).__init__()
                self.my_ops = CustomOpBuilder("single_op",
                                              ["kernel_impl/add4.cpp",
                                               "kernel_impl/module.cpp"],
                                              backend="Ascend", op_def=["ops_yaml/add4.yaml"],
                                              op_doc=["ops_yaml/add4_doc.yaml"],
                                              build_dir=tmpdirname).load()

            def construct(self, x, y):
                return self.my_ops.add4(x, y, 1)

        x = np.array([1, 2, 3], dtype=np.float16)
        y = np.array([4, 5, 6], dtype=np.float16)
        output = MyNet()(ms.Tensor(x), ms.Tensor(y))
        expect = x + y
        print(output.asnumpy())
        assert np.allclose(output.asnumpy(), expect)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=4.0)
def test_custom_single_operator_tuple():
    """
    Feature: Single-operator loading via CustomOpBuilder.
    Description: Build and execute the add4 operator defined in YAML and CPP files in GRAPH mode.
    Expectation: Execution succeeds and outputs match NumPy results.
    """
    ms.set_device("Ascend")
    ms.set_context(mode=ms.GRAPH_MODE, save_graphs=False, save_graphs_path="./graphs")

    with tempfile.TemporaryDirectory() as tmpdirname:
        class MyNet(ms.nn.Cell):
            def __init__(self):
                super(MyNet, self).__init__()
                self.my_ops = CustomOpBuilder("single_op_tuple",
                                              ("kernel_impl/add4.cpp",
                                               "kernel_impl/module.cpp"),
                                              backend="Ascend", op_def=("ops_yaml/add4.yaml"),
                                              op_doc=("ops_yaml/add4_doc.yaml"),
                                              build_dir=tmpdirname).load()

            def construct(self, x, y):
                return self.my_ops.add4(x, y, 1)

        x = np.array([1, 2, 3], dtype=np.float16)
        y = np.array([4, 5, 6], dtype=np.float16)
        output = MyNet()(ms.Tensor(x), ms.Tensor(y))
        expect = x + y
        print(output.asnumpy())
        assert np.allclose(output.asnumpy(), expect)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=4.0)
def test_custom_single_operator_no_doc():
    """
    Feature: Single-operator loading without documentation YAML.
    Description: Build and execute the add4 operator defined only by operator YAML and CPP files.
    Expectation: Execution succeeds and outputs match NumPy results.
    """
    ms.set_device("Ascend")
    ms.set_context(mode=ms.GRAPH_MODE, save_graphs=False, save_graphs_path="./graphs")

    with tempfile.TemporaryDirectory() as tmpdirname:
        class MyNet(ms.nn.Cell):
            def __init__(self):
                super(MyNet, self).__init__()
                self.my_ops = CustomOpBuilder("single_op_no_doc",
                                              ["kernel_impl/add4.cpp",
                                               "kernel_impl/module.cpp"],
                                              backend="Ascend", op_def=["ops_yaml/add4.yaml"],
                                              build_dir=tmpdirname).load()

            def construct(self, x, y):
                return self.my_ops.add4(x, y, 1)

        x = np.array([1, 2, 3], dtype=np.float16)
        y = np.array([4, 5, 6], dtype=np.float16)
        output = MyNet()(ms.Tensor(x), ms.Tensor(y))
        expect = x + y
        print(output.asnumpy())
        assert np.allclose(output.asnumpy(), expect)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=4.0)
def test_custom_single_operator_func():
    """
    Feature: Single-operator execution from a JIT-compiled function.
    Description: Load add4 operator via CustomOpBuilder and call it inside a JIT function.
    Expectation: Execution succeeds and outputs match NumPy results.
    """
    ms.set_device("Ascend")
    ms.set_context(mode=ms.GRAPH_MODE, save_graphs=False, save_graphs_path="./graphs")
    with tempfile.TemporaryDirectory() as tmpdirname:
        my_ops = CustomOpBuilder("single_op_func",
                                 ["kernel_impl/add4.cpp",
                                  "kernel_impl/module.cpp"],
                                 backend="Ascend", op_def=["ops_yaml/add4.yaml"],
                                 op_doc=["ops_yaml/add4_doc.yaml"], build_dir=tmpdirname).load()

        @ms.jit()
        def add_net(x, y):
            return my_ops.add4(x, y)

        x = np.array([1, 2, 3], dtype=np.float16)
        y = np.array([4, 5, 6], dtype=np.float16)
        output = add_net(ms.Tensor(x), ms.Tensor(y))
        expect = x + y
        print(output.asnumpy())
        assert np.allclose(output.asnumpy(), expect)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=4.0)
def test_custom_multi_operator():
    """
    Feature: Multiple-operator loading and chaining.
    Description: Build both add4 and add3 operators, chain them in a single graph, and verify cumulative results.
    Expectation: Execution succeeds and final outputs equal x + y + y.
    """
    ms.set_device("Ascend")
    ms.set_context(mode=ms.GRAPH_MODE, save_graphs=False, save_graphs_path="./graphs")

    with tempfile.TemporaryDirectory() as tmpdirname:
        class MyNet(ms.nn.Cell):
            def __init__(self):
                super(MyNet, self).__init__()
                self.my_ops = CustomOpBuilder("graphmode_add_2",
                                              ["kernel_impl/add4.cpp", "kernel_impl/add3.cpp",
                                               "kernel_impl/module.cpp"], backend="Ascend",
                                              op_def=["ops_yaml/add4.yaml",
                                                      "ops_yaml/add3.yaml"], build_dir=tmpdirname).load()

            def construct(self, x, y):
                out = self.my_ops.add4(x, y, 1)
                return self.my_ops.add3(out, y, 1)

        x = np.array([1, 2, 3], dtype=np.float16)
        y = np.array([4, 5, 6], dtype=np.float16)
        output = MyNet()(ms.Tensor(x), ms.Tensor(y))
        expect = x + y + y
        print(output.asnumpy())
        assert np.allclose(output.asnumpy(), expect)


def test_graphmode_add_offline():
    """
    Feature: Offline-compiled custom operator.
    Description: Use a pre-built add4 operator imported as a Python module inside a JIT graph.
    Expectation: Execution succeeds and outputs equal x + y.
    """
    ms.set_device("Ascend")
    ms.set_context(mode=ms.GRAPH_MODE, save_graphs=False, save_graphs_path="./graphs")
    import custom_ops

    @ms.jit()
    def add_net(x, y):
        out = custom_ops.add4(x, y)
        return custom_ops.add4(out, y)

    x = np.array([1, 2, 3], dtype=np.float16)
    y = np.array([4, 5, 6], dtype=np.float16)
    output = add_net(ms.Tensor(x), ms.Tensor(y))
    expect = x + y + y
    assert np.allclose(output.asnumpy(), expect)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_CustomOpBuilder_exception_1():
    """
    Feature: CustomOpBuilder error handling in PyNative mode.
    Description: Attempt to run a custom operator built for GRAPH mode under PyNative mode.
    Expectation: RuntimeError containing 'not support PyNative mode' is raised.
    """
    ms.set_device("Ascend")
    ms.set_context(mode=ms.PYNATIVE_MODE, save_graphs=False, save_graphs_path="./graphs")
    with tempfile.TemporaryDirectory() as tmpdirname:
        class MyNet(ms.nn.Cell):
            def __init__(self):
                super(MyNet, self).__init__()
                self.my_ops = CustomOpBuilder("exception_1",
                                              ["kernel_impl/add4.cpp",
                                               'kernel_impl/pyboost_aclnn_sum.cpp',
                                               "kernel_impl/module.cpp"],
                                              backend="Ascend", op_def=["ops_yaml/add4.yaml"],
                                              op_doc=["ops_yaml/add4_doc.yaml"],
                                              build_dir=tmpdirname).load()

            def construct(self, x, y):
                return self.my_ops.add4(x, y, 1)

        x = np.array([1, 2, 3], dtype=np.float16)
        y = np.array([4, 5, 6], dtype=np.float16)
        try:
            MyNet()(ms.Tensor(x), ms.Tensor(y))
        except RuntimeError as e:
            assert "not support PyNative mode" in str(e)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_CustomOpBuilder_exception_2():
    """
    Feature: CustomOpBuilder unsupported GRAPH mode operator.
    Description: Attempt to invoke an operator that explicitly does not support GRAPH mode.
    Expectation: AttributeError containing 'does not support GRAPH mode' is raised.
    """
    ms.set_device("Ascend")
    ms.set_context(mode=ms.GRAPH_MODE, save_graphs=False, save_graphs_path="./graphs")
    with tempfile.TemporaryDirectory() as tmpdirname:
        class MyNet(ms.nn.Cell):
            def __init__(self):
                super(MyNet, self).__init__()
                self.my_ops = CustomOpBuilder("exception_2",
                                              ["kernel_impl/add4.cpp",
                                               'kernel_impl/pyboost_aclnn_sum.cpp',
                                               "kernel_impl/module.cpp"],
                                              backend="Ascend", op_def=["ops_yaml/add4.yaml"],
                                              op_doc=["ops_yaml/add4_doc.yaml"],
                                              build_dir=tmpdirname).load()

            def construct(self, x, y, z):
                return self.my_ops.npu_abs_reduce_sum(x, y, z)

        x = np.random.rand(4, 5, 6).astype(np.float32)
        try:
            MyNet()(ms.Tensor(x), (1,), True)
        except AttributeError as e:
            assert "does not support GRAPH mode" in str(e)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_CustomOpBuilder_exception_3():
    """
    Feature: CustomOpBuilder missing operator lookup.
    Description: Attempt to call an operator not present in either function or shared-object modules.
    Expectation: AttributeError containing 'neither in func_module nor in so_module' is raised.
    """
    ms.set_device("Ascend")
    ms.set_context(mode=ms.PYNATIVE_MODE, save_graphs=False, save_graphs_path="./graphs")
    with tempfile.TemporaryDirectory() as tmpdirname:
        class MyNet(ms.nn.Cell):
            def __init__(self):
                super(MyNet, self).__init__()
                self.my_ops = CustomOpBuilder("exception_3",
                                              ["kernel_impl/add4.cpp",
                                               'kernel_impl/pyboost_aclnn_sum.cpp',
                                               "kernel_impl/module.cpp"],
                                              backend="Ascend", op_def=["ops_yaml/add4.yaml"],
                                              op_doc=["ops_yaml/add4_doc.yaml"],
                                              build_dir=tmpdirname).load()

            def construct(self, x, y, z):
                return self.my_ops.add3(x, y, z)

        x = np.random.rand(4, 5, 6).astype(np.float32)
        try:
            MyNet()(ms.Tensor(x), (1,), True)
        except AttributeError as e:
            assert "neither in func_module nor in so_module" in str(e)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=4.0)
def test_aclnn_batch_norm():
    """
    Feature: CustomOpBuilder.
    Description: Custom aclnn op.
    Expectation: success.
    """

    ms.set_device("Ascend")
    ms.set_context(save_graphs=False, save_graphs_path="./graphs")

    with tempfile.TemporaryDirectory() as tmpdirname:
        my_ops = CustomOpBuilder("aclnn_op_1",
                                 ["kernel_impl/batch_norm.cpp",
                                  "kernel_impl/module.cpp"],
                                 backend="Ascend", op_def=["ops_yaml/batch_norm.yaml"],
                                 build_dir=tmpdirname).load()

        @ms.jit()
        def func(x, scale, bias, mean, variance):
            return my_ops.batch_norm(x, scale, bias, mean, variance, False, 0.1, 1e-5)

        x = ms.Tensor((3 * np.ones(16)).reshape(2, 2, 1, 4).astype(np.float32))
        scale = ms.Tensor(np.ones(2).astype(np.float32))
        bias = ms.Tensor(np.ones(2).astype(np.float32))
        mean = ms.Tensor(np.ones(2).astype(np.float32))
        variance = ms.Tensor(np.ones(2).astype(np.float32))

        expect = np.array([2.99999]).repeat(16, axis=0).astype(np.float32).reshape((2, 2, 1, 4))
        output = func(x, scale, bias, mean, variance)[0]
        assert np.allclose(output.asnumpy(), expect, 1e-3, 1e-3)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=4.0)
def test_aclnn_inplace_add():
    """
    Feature: CustomOpBuilder.
    Description: Custom aclnn op.
    Expectation: success.
    """

    ms.set_device("Ascend")
    ms.set_context(save_graphs=False, save_graphs_path="./graphs")

    with tempfile.TemporaryDirectory() as tmpdirname:
        my_ops = CustomOpBuilder("aclnn_op_2",
                                 ["kernel_impl/inplace_add.cpp",
                                  "kernel_impl/module.cpp"],
                                 backend="Ascend", op_def=["ops_yaml/inplace_add.yaml"],
                                 build_dir=tmpdirname).load()

        @ms.jit()
        def func(x, y):
            return my_ops.inplace_add(x, y)

        x = np.array([1, 2, 3], dtype=np.float16)
        y = np.array([4, 5, 6], dtype=np.float16)
        input_x = ms.Tensor(x)
        input_y = ms.Tensor(y)
        func(input_x, input_y)
        expect = x + y
        assert np.allclose(input_x.asnumpy(), expect, 1e-3, 1e-3)


if __name__ == "__main__":
    test_custom_single_operator()
