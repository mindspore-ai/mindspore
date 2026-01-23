# Copyright 2024-2025 Huawei Technologies Co., Ltd
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
# ==============================================================================
""" test_tensor_inplace_view.py """
import pytest
import os
import re
import shutil
import numpy as np
import mindspore as ms
from mindspore import nn
from mindspore import context, Tensor
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype
from mindspore.ops.auto_generate.gen_ops_prim import split_tensor_view_op
from tests.mark_utils import arg_mark

context.set_context(mode=ms.GRAPH_MODE)


@pytest.mark.skip(reason="The operation is Unsupported")
@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard',
        essential_mark='essential')
def test_tensor_view_inplace_reshape():
    """
    Feature: Support tensor inplace view.
    Description: Support tensor inplace view.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x, y):
            x = x + 2
            reshape_x = P.Reshape()(x, (2,))
            P.AssignAdd()(reshape_x, y)
            z = x + 1
            return z

    input_x = ms.Tensor([[2, 2]], dtype=ms.int32)
    input_y = ms.Tensor([3, 3], dtype=ms.int32)
    net = Net()
    out = net(input_x, input_y)
    assert (out.asnumpy() == [8, 8]).all()


@pytest.mark.skip(reason="The operation is Unsupported")
@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_tensor_view_inplace_split():
    """
    Feature: Support tensor inplace view.
    Description: Support tensor inplace view.
    Expectation: Run success.
    """
    class TensorSplitNet(nn.Cell):
        def __init__(self, axis=0, output_num=2):
            super(TensorSplitNet, self).__init__()
            self.split = P.Split(axis, output_num)

        def construct(self, x):
            x1, x2 = self.split(x)
            P.AssignAdd()(x1, x2)
            P.AssignSub()(x2, x1)
            y = x * 2
            return y

    np_x = np.array(np.arange(20).reshape((10, 2)), dtype=np.float32)
    x = ms.Tensor(np_x, dtype=ms.float32)
    out = TensorSplitNet(0, 2)(x)
    print("out:", out)


class ViewOut(nn.Cell):
    def __init__(self):
        super(ViewOut, self).__init__()
        self.transpose = P.TransposeView()
        self.assign = P.Assign()

    @ms.jit
    def construct(self, x):
        self.transpose(x, (0, 1, 2))
        self.assign(x, x * 2)
        return x * 3


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=4.0)
def test_graph_view_out():
    """
    Feature: Runtime view graph mode.
    Description: view op as graph output.
    Expectation: pass.
    """
    context.set_context(mode=context.GRAPH_MODE, jit_config={"jit_level": "O0"})
    x1 = ms.Tensor(np.array([[[1, 0, 0, 0], [0, 0, 0, 0], [-1, -1, 0, -1]],
                             [[0, -1, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0]]]), ms.int32)
    net = ViewOut()
    out_graph = net(x1)
    x2 = ms.Tensor(np.array([[[1, 0, 0, 0], [0, 0, 0, 0], [-1, -1, 0, -1]],
                             [[0, -1, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0]]]), ms.int32)
    x2.transpose((0, 1, 2))
    x2 += x2
    z = x2 * 3
    assert np.allclose(out_graph.asnumpy(), z.asnumpy(), rtol=10e-4, atol=10e-4)


class ViewOut2(nn.Cell):
    def __init__(self):
        super(ViewOut2, self).__init__()
        self.transpose = P.Transpose()
        self.assign = P.Assign()

    @ms.jit
    def construct(self, x, y):
        self.transpose(x, (0, 1, 2))
        self.assign(y, x * 2)
        return x * 3 + y


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=4.0)
def test_graph_view_out_tensormove():
    """
    Feature: Runtime view graph mode.
    Description: view op as graph output.
    Expectation: pass.
    """
    context.set_context(jit_config={"jit_level": "O0"})
    context.set_context(mode=context.GRAPH_MODE)
    x1 = ms.Tensor(np.array([[[1, 0, 0, 0], [0, 0, 0, 0], [-1, -1, 0, -1]],
                             [[0, -1, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0]]]), ms.int32)
    y1 = ms.Tensor(np.array([[[1, 0, 0, 0], [0, 0, 0, 0], [-1, -1, 0, -1]],
                             [[0, -1, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0]]]), ms.int32)
    net = ViewOut2()
    out_graph = net(x1, y1)
    x2 = ms.Tensor(np.array([[[1, 0, 0, 0], [0, 0, 0, 0], [-1, -1, 0, -1]],
                             [[0, -1, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0]]]), ms.int32)
    x2.transpose((0, 1, 2))
    y2 = x2 * 2
    z = x2 * 3 + y2
    assert np.allclose(out_graph.asnumpy(), z.asnumpy(), rtol=10e-4, atol=10e-4)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_setitem_simple_case():
    """
    Feature: Support tensor slice forward execute.
    Description: Support tensor slice forward execute.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.n = 2

        def construct(self, x, y):
            x[y.shape[0]:y.shape[1]] = x[0:y.shape[0]]
            out = x
            return out * self.n

    try:
        os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'
        context.set_context(mode=1)
        np_x = np.random.rand(6, 3, 4)
        np_y = np.random.rand(2, 4)
        # Run in pynative mode
        net = Net()
        out_expect = net(Tensor(np_x, dtype=ms.float32), Tensor(np_y, dtype=ms.int32))
        # Run in jit execute mode
        net.construct = ms.jit(net.construct, backend="ms_backend")
        d = Tensor(None, dtype=ms.float32)
        dy = Tensor(shape=[None, None], dtype=ms.int32)
        net.set_inputs(d, dy)
        out = net(Tensor(np_x, dtype=ms.float32), Tensor(np_y, dtype=ms.int32))
        assert np.allclose(out_expect.asnumpy(), out.asnumpy())
    finally:
        del os.environ["MS_DEV_TENSOR_INDEX_BOOST"]


def clean_all_ir_files(folder_path):
    if os.path.exists(folder_path):
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.ir') or file_name.endswith('.dot') or \
                    file_name.endswith('.dat') or file_name.endswith('.pb'):
                os.remove(os.path.join(folder_path, file_name))


def find_newest_validateir_file(folder_path):
    ckpt_files = map(lambda f: os.path.join(folder_path, f),
                     filter(lambda f: re.match(r'\d+_validate_\d+.ir', f),
                            os.listdir(folder_path)))
    return max(ckpt_files, key=os.path.getctime)


def read_file(save_path):
    filename = find_newest_validateir_file(save_path)
    with open((os.path.join(filename)), 'r', encoding="utf-8") as f:
        content = f.read()
    clean_all_ir_files(save_path)
    return content


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=2048.0)
def test_split_tensor_view_op():
    """
    Feature: View feature.
    Description: View feature.
    Expectation: The input of the view operator needs to be updated to ref_key.
    """
    class Split(nn.Cell):
        def construct(self, tensor, split_size_or_sections, dim):
            out = split_tensor_view_op(tensor, split_size_or_sections, dim)
            return out

    save_path = "./test_split_tensor_view_op"
    context.set_context(jit_config={"jit_level": "O0"}, save_graphs=True, save_graphs_path=save_path)
    tensor_input = Tensor(np.random.randn(3,), mstype.float32)
    net = Split()
    net.construct = ms.jit(net.construct, backend="ms_backend")
    net(tensor_input, 2, -1)
    content = read_file(save_path)
    ref_key_set = re.findall('ref_key', content)
    try:
        shutil.rmtree(save_path)
    except FileNotFoundError:
        pass
    assert len(ref_key_set) == 2
