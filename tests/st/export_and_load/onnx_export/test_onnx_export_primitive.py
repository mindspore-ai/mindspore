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
"""Test onnx export."""
import os
import numpy as np
import pytest
import onnxruntime as ort
from tests.mark_utils import arg_mark

import mindspore as ms
from mindspore import nn
from mindspore import context
from mindspore import Tensor, mint, ops
from mindspore.ops import operations as P
from mindspore.onnx import export
from mindspore.ops.auto_generate import SliceExt
from mindspore.ops.function.nn_func import batch_norm_ext
from mindspore.mint.nn.functional import conv2d
from mindspore.ops import auto_generate as gen
from mindspore.ops.auto_generate.gen_ops_prim import split_tensor_op


class InputNetTopK(nn.Cell):
    def construct(self, x):
        output = mint.topk(input=x, k=2)
        return output


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
def test_convert_model_with_topk():
    """
    Feature: Convert mindir to onnx and infer by onnx
    Description: Test topk between ms and onnx
    Expectation: success
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    net = InputNetTopK()
    x = [[0.5368, 0.2447, 0.4302, 0.9673],
         [0.4388, 0.6525, 0.4685, 0.1868],
         [0.3563, 0.5152, 0.9675, 0.8230]]
    ms_x = Tensor(x, dtype=ms.float32)
    onnx_name = "topk_net.onnx"
    export(net, ms_x, file_name=onnx_name)
    assert os.path.exists(onnx_name)
    if os.path.isfile("./topk_net.onnx"):
        np_x = np.array(x, dtype=np.float32)
        session = ort.InferenceSession("./topk_net.onnx")
        output = session.run(None, {'x': np_x})

        expected = (np.array([[9.67299998e-01, 5.36800027e-01],
                              [6.52499974e-01, 4.68499988e-01],
                              [9.67499971e-01, 8.23000014e-01]], dtype=np.float32),
                    np.array([[3, 0], [1, 2], [2, 3]], dtype=np.int32))
        assert np.array_equal(output, expected), "topk not equal, please check"
        os.remove("./topk_net.onnx")


class InputNetAtan(nn.Cell):
    def construct(self, x):
        output = mint.atan(x)
        return output


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='essential', mem_peak=0.01)
def test_convert_model_with_atan():
    """
    Feature: Convert mindir to onnx and infer by onnx
    Description: Test atan between ms and onnx
    Expectation: success
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    net = InputNetAtan()
    x = [1.0, 0.0]
    ms_x = Tensor(x, dtype=ms.float32)
    onnx_name = "atan_net.onnx"
    export(net, ms_x, file_name=onnx_name)
    assert os.path.exists(onnx_name)
    if os.path.isfile("./atan_net.onnx"):
        np_x = np.array(x, dtype=np.float32)
        session = ort.InferenceSession("./atan_net.onnx")
        output = session.run(None, {'x': np_x})

        expected = np.array([0.7853982, 0.0], dtype=np.float32)
        assert np.array_equal(output[0], expected), "atan not equal, please check"
        os.remove("./atan_net.onnx")


class InputNetArgMax(nn.Cell):
    def construct(self, x):
        output = mint.argmax(x, dim=-1, keepdim=False)
        return output


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
def test_convert_model_with_argmax():
    """
    Feature: Convert mindir to onnx and infer by onnx
    Description: Test argmax between ms and onnx
    Expectation: success
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    net = InputNetArgMax()
    x = [[1, 20, 5], [67, 8, 9], [130, 24, 15]]
    ms_x = Tensor(x, dtype=ms.float32)
    onnx_name = "argmax_net.onnx"
    export(net, ms_x, file_name=onnx_name)
    assert os.path.exists(onnx_name)
    if os.path.isfile("./argmax_net.onnx"):
        np_x = np.array(x, dtype=np.float32)
        session = ort.InferenceSession("./argmax_net.onnx")
        output = session.run(None, {'x': np_x})

        expected = np.array([1, 0, 0], dtype=np.int32)
        assert np.array_equal(output[0], expected), "argmax not equal, please check"
        os.remove("./argmax_net.onnx")


class InputNetClipScalar(nn.Cell):
    def construct(self, x, min_val, max_val):
        output = mint.clip(x, min=min_val, max=max_val)
        return output


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='essential', mem_peak=0.01)
def test_convert_model_with_clip_scalar():
    """
    Feature: Convert mindir to onnx and infer by onnx
    Description: Test clip scalar between ms and onnx
    Expectation: success
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    net = InputNetClipScalar()
    x = [[1., 25., 5., 7.], [4., 11., 6., 21.]]
    ms_x = Tensor(x, dtype=ms.float32)
    min_value = 5.0
    max_value = 20.0
    onnx_name = "clip_scalar_net.onnx"
    export(net, ms_x, min_value, max_value, file_name=onnx_name)
    assert os.path.exists(onnx_name)
    if os.path.isfile("./clip_scalar_net.onnx"):
        np_x = np.array(x, dtype=np.float32)
        session = ort.InferenceSession("./clip_scalar_net.onnx")
        output = session.run(None, {'x': np_x})

        expected = np.array([[5., 20., 5., 7.], [5., 11., 6., 20.]], dtype=np.float32)
        assert np.array_equal(output[0], expected), "clip scalar not equal, please check"
        os.remove("./clip_scalar_net.onnx")


class InputNetClipTensor(nn.Cell):
    def construct(self, x, min_val, max_val):
        output = mint.clip(x, min=min_val, max=max_val)
        return output


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='essential', mem_peak=0.01)
def test_convert_model_with_clip_tensor():
    """
    Feature: Convert mindir to onnx and infer by onnx
    Description: Test clip tensor between ms and onnx
    Expectation: success
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    net = InputNetClipScalar()
    x = [[1., 25., 5., 7.], [4., 11., 6., 21.]]
    ms_x = Tensor(x, dtype=ms.float32)
    min_value = Tensor(5.0, dtype=ms.float32)
    max_value = Tensor(20.0, dtype=ms.float32)
    onnx_name = "clip_tensor_net.onnx"
    export(net, ms_x, min_value, max_value, file_name=onnx_name)
    assert os.path.exists(onnx_name)
    if os.path.isfile("./clip_tensor_net.onnx"):
        np_x = np.array(x, dtype=np.float32)
        session = ort.InferenceSession("./clip_tensor_net.onnx")
        output = session.run(None, {'x': np_x, 'min_val': np.array(
            [5.0], dtype=np.float32), 'max_val': np.array([20.0], dtype=np.float32)})

        expected = np.array([[5., 20., 5., 7.], [5., 11., 6., 20.]], dtype=np.float32)
        assert np.array_equal(output[0], expected), "clip tensor not equal, please check"
        os.remove("./clip_tensor_net.onnx")


class InputNetPad(nn.Cell):
    def construct(self, x):
        output = mint.nn.functional.pad(x, [2, 0, 0, 0], mode='constant', value=0.0)
        return output


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
def test_convert_model_with_pad():
    """
    Feature: Convert mindir to onnx and infer by onnx
    Description: Test pad between ms and onnx
    Expectation: success
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    net = InputNetPad()
    x = [[1.0, 1.2], [2.3, 3.4], [4.5, 5.7]]
    ms_x = Tensor(x, dtype=ms.float32)
    onnx_name = "pad_net.onnx"
    export(net, ms_x, file_name=onnx_name)
    assert os.path.exists(onnx_name)
    if os.path.isfile("./pad_net.onnx"):
        np_x = np.array(x, dtype=np.float32)
        session = ort.InferenceSession("./pad_net.onnx")
        output = session.run(None, {'x': np_x})

        expected = np.array([[0., 0., 1., 1.2],
                             [0., 0., 2.3, 3.4],
                             [0., 0., 4.5, 5.7]], dtype=np.float32)
        assert np.array_equal(output[0], expected), "pad not equal, please check"
        os.remove("./pad_net.onnx")


class InputNetSub(nn.Cell):
    def construct(self, x, y):
        output = mint.sub(x, y, alpha=1)
        return output


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_convert_model_with_sub():
    """
    Feature: Convert mindir to onnx and infer by onnx
    Description: Test sub between ms and onnx
    Expectation: success
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    net = InputNetSub()
    x = [4, 5, 6]
    ms_x = Tensor(x, dtype=ms.float32)
    y = 1
    ms_y = Tensor(y, dtype=ms.int32)
    onnx_name = "sub_net.onnx"
    export(net, ms_x, ms_y, file_name=onnx_name)
    assert os.path.exists(onnx_name)
    if os.path.isfile("./sub_net.onnx"):
        np_x = np.array(x, dtype=np.float32)
        np_y = np.array(y, dtype=np.int32)
        session = ort.InferenceSession("./sub_net.onnx")
        output = session.run(None, {'x': np_x, 'y': np_y})

        expected = np.array([3., 4., 5.], dtype=np.float32)
        assert np.array_equal(output[0], expected), "sub not equal, please check"
        os.remove("./sub_net.onnx")


class InputNetMatMul(nn.Cell):
    def construct(self, x, y):
        output = mint.matmul(x, y)
        return output


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
def test_convert_model_with_matmul_and_input_output_names():
    """
    Feature: Convert mindir to onnx and infer by onnx and change input output node names
    Description: Test matmul between ms and onnx
    Expectation: success
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    net = InputNetMatMul()
    x = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    ms_x = Tensor(x, dtype=ms.float32)
    y = np.arange(4 * 5).reshape(4, 5)
    ms_y = Tensor(y, dtype=ms.float32)
    onnx_name = "matmul_net.onnx"
    export(net, ms_x, ms_y, file_name=onnx_name, input_names=["input_x", "input_y"], output_names=["output"])
    assert os.path.exists(onnx_name)
    if os.path.isfile("./matmul_net.onnx"):
        np_x = np.array(x, dtype=np.float32)
        np_y = np.array(y, dtype=np.float32)
        session = ort.InferenceSession("./matmul_net.onnx")
        output = session.run(None, {'input_x': np_x, 'input_y': np_y})

        expected = np.array([[[70., 76., 82., 88., 94.],
                              [190., 212., 234., 256., 278.],
                              [310., 348., 386., 424., 462.]],
                             [[430., 484., 538., 592., 646.],
                              [550., 620., 690., 760., 830.],
                              [670., 756., 842., 928., 1014.]]], dtype=np.float32)
        assert np.array_equal(output[0], expected), "matmul not equal, please check"
        os.remove("./matmul_net.onnx")


class InputNetTranspose(nn.Cell):
    def construct(self, x, dim1, dim2):
        output = mint.transpose(x, dim1, dim2)
        return output


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_convert_model_with_transpose():
    """
    Feature: Convert mindir to onnx and infer by onnx
    Description: Test transpose between ms and onnx
    Expectation: success
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    net = InputNetTranspose()
    x = [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]
    ms_x = Tensor(x, dtype=ms.float32)
    dim1 = 2
    dim2 = 1
    onnx_name = "transpose_net.onnx"
    export(net, ms_x, dim1, dim2, file_name=onnx_name)
    assert os.path.exists(onnx_name)
    if os.path.isfile("./transpose_net.onnx"):
        np_x = np.array(x, dtype=np.float32)
        session = ort.InferenceSession("./transpose_net.onnx")
        output = session.run(None, {'x': np_x})

        expected = np.array([[[1, 4], [2, 5], [3, 6]], [[7, 10], [8, 11], [9, 12]]], dtype=np.float32)
        assert np.array_equal(output[0], expected), "transpose not equal, please check"
        os.remove("./transpose_net.onnx")


class InputNetSplitTensor(nn.Cell):
    def construct(self, x):
        output = split_tensor_op(x, 3)
        return output


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_convert_model_with_split_tensor():
    """
    Feature: Convert mindir to onnx and infer by onnx
    Description: Test split tensor between ms and onnx
    Expectation: success
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    net = InputNetSplitTensor()
    x = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    ms_x = Tensor(x, dtype=ms.float32)
    onnx_name = "split_net.onnx"
    export(net, ms_x, file_name=onnx_name)
    assert os.path.exists(onnx_name)
    if os.path.isfile("./split_net.onnx"):
        np_x = np.array(x, dtype=np.float32)
        session = ort.InferenceSession("./split_net.onnx")
        output = session.run(None, {'x': np_x})

        expected = [np.array([0, 1, 2], dtype=np.float32), np.array(
            [3, 4, 5], dtype=np.float32), np.array([6, 7, 8], dtype=np.float32)]
        assert np.array_equal(output, expected), "split not equal, please check"
        os.remove("./split_net.onnx")


class InputNetAddExt(nn.Cell):
    def construct(self, x, y):
        output = mint.add(x, y, alpha=1)
        return output


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
def test_convert_model_with_add_ext():
    """
    Feature: Convert mindir to onnx and infer by onnx
    Description: Test add_ext between ms and onnx
    Expectation: success
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    net = InputNetAddExt()
    x = [0, 1, 2]
    ms_x = Tensor(x, dtype=ms.float32)
    y = [1, 2, 3]
    ms_y = Tensor(y, dtype=ms.float32)
    onnx_name = "add_ext_net.onnx"
    export(net, ms_x, ms_y, file_name=onnx_name)
    assert os.path.exists(onnx_name)
    if os.path.isfile("./add_ext_net.onnx"):
        np_x = np.array(x, dtype=np.float32)
        np_y = np.array(y, dtype=np.float32)
        session = ort.InferenceSession("./add_ext_net.onnx")
        output = session.run(None, {'x': np_x, 'y': np_y})

        expected = np.array([1, 3, 5], dtype=np.float32)
        assert np.array_equal(output[0], expected), "add_ext not equal, please check"
        os.remove("./add_ext_net.onnx")


class InputNetSliceExt(nn.Cell):
    def __init__(self):
        super().__init__()
        self.slice_ext = SliceExt()

    def construct(self, x):
        output = self.slice_ext(x, 0, 0, 2, 1)
        return output


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
def test_convert_model_with_slice_ext():
    """
    Feature: Convert mindir to onnx and infer by onnx
    Description: Test SliceExt between ms and onnx
    Expectation: success
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    net = InputNetSliceExt()
    x = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    ms_x = Tensor(x, dtype=ms.float32)
    onnx_name = "slice_ext_net.onnx"
    export(net, ms_x, file_name=onnx_name)
    assert os.path.exists(onnx_name)
    if os.path.isfile("./slice_ext_net.onnx"):
        np_x = np.array(x, dtype=np.float32)
        session = ort.InferenceSession("./slice_ext_net.onnx")
        output = session.run(None, {'x': np_x})

        expected = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        assert np.array_equal(output[0], expected), "slice_ext not equal, please check"
        os.remove("./slice_ext_net.onnx")


class InputNetBatchNormExt(nn.Cell):
    def construct(self, x):
        scale = Tensor(np.ones([2]), ms.float32)
        bias = Tensor(np.ones([2]), ms.float32)
        mean = Tensor(np.ones([2]), ms.float32)
        variance = Tensor(np.ones([2]), ms.float32)
        output = batch_norm_ext(x, mean, variance, scale, bias, False, 0.1, 1e-5)
        return output


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='essential', mem_peak=1024.00)
def test_convert_model_with_batch_norm_ext():
    """
    Feature: Convert mindir to onnx and infer by onnx
    Description: Test BatchNormExt between ms and onnx
    Expectation: success
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    net = InputNetBatchNormExt()
    x = (3 * np.ones(16)).reshape(2, 2, 1, 4).astype(np.float32)
    ms_x = Tensor(x, dtype=ms.float32)
    onnx_name = "batchnorm_ext_net.onnx"
    export(net, ms_x, file_name=onnx_name)
    assert os.path.exists(onnx_name)
    if os.path.isfile("./batchnorm_ext_net.onnx"):
        np_x = np.array(x, dtype=np.float32)
        session = ort.InferenceSession("./batchnorm_ext_net.onnx")
        output = session.run(None, {'x': np_x})

        expect = np.array([2.99999])
        expected = expect.repeat(16).astype(np.float32).reshape((2, 2, 1, 4))
        assert np.allclose(output[0], expected, rtol=1e-4, atol=1e-4), "batchnorm_ext not equal, please check"
        os.remove("./batchnorm_ext_net.onnx")


class InputNetBatchNorm(nn.Cell):
    def __init__(self):
        super().__init__()
        self.batch_norm = ops.BatchNorm()

    def construct(self, x):
        scale = Tensor(np.ones([2]), ms.float32)
        bias = Tensor(np.ones([2]), ms.float32)
        mean = Tensor(np.ones([2]), ms.float32)
        variance = Tensor(np.ones([2]), ms.float32)
        output = self.batch_norm(x, scale, bias, mean, variance)[0]
        return output


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='essential', mem_peak=0.01)
def test_convert_model_with_batch_norm():
    """
    Feature: Convert mindir to onnx and infer by onnx
    Description: Test ExportMergeBatchNorm function with BatchNorm[0] between ms and onnx
    Expectation: success
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    net = InputNetBatchNorm()
    x = (3 * np.ones(16)).reshape(2, 2, 1, 4).astype(np.float32)
    ms_x = Tensor(x, dtype=ms.float32)
    onnx_name = "batchnorm_net.onnx"
    export(net, ms_x, file_name=onnx_name)
    assert os.path.exists(onnx_name)
    if os.path.isfile("./batchnorm_net.onnx"):
        np_x = np.array(x, dtype=np.float32)
        session = ort.InferenceSession("./batchnorm_net.onnx")
        output = session.run(None, {'x': np_x})

        expect = np.array([2.99999])
        expected = expect.repeat(16).astype(np.float32).reshape((2, 2, 1, 4))
        assert np.allclose(output[0], expected, rtol=1e-4, atol=1e-4), "batchnorm not equal, please check"
        os.remove("./batchnorm_net.onnx")


class BatchMatmulExtNet(nn.Cell):
    def construct(self, a, b):
        return gen.BatchMatMulExt()(a, b)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0',
        card_mark='onecard', essential_mark='essential', mem_peak=1024.01)
def test_export_batchmatmulext():
    """
    Feature: Export ops.BatchMatMulExt to onnx
    Description: Export ops.BatchMatMulExt to onnx
    Expectation: success
    """
    np_a = np.ones(shape=[2, 3, 4]).astype(np.float32)
    np_b = np.ones(shape=[2, 4, 5]).astype(np.float32)
    a = Tensor(np_a)
    b = Tensor(np_b)
    net = BatchMatmulExtNet()
    y = net(a, b)
    export(net, a, b, file_name='./batchmatmulext_onnx')
    if os.path.isfile("./batchmatmulext_onnx.onnx"):
        session = ort.InferenceSession("./batchmatmulext_onnx.onnx")
        output = session.run(None, {"a": np_a, "b": np_b})[0]
        assert np.array_equal(output, y.asnumpy())
        os.remove("./batchmatmulext_onnx.onnx")
    else:
        raise RuntimeError("Export operator BatchMatMulExt to ONNX failed!")


class IdentityNet(nn.Cell):
    def construct(self, x):
        return gen.Identity()(x)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=1024.01)
def test_export_identity():
    """
    Feature: Export ops.Identity to onnx
    Description: Export ops.Identity to onnx
    Expectation: success
    """
    np_x = np.ones(shape=[2, 3, 4]).astype(np.float32)
    x = Tensor(np_x)
    net = IdentityNet()
    y = net(x)
    export(net, x, file_name='./identity_onnx')
    if os.path.isfile("./identity_onnx.onnx"):
        session = ort.InferenceSession("./identity_onnx.onnx")
        output = session.run(None, {"x": np_x})[0]
        assert np.array_equal(output, y.asnumpy())
        os.remove("./identity_onnx.onnx")
    else:
        raise RuntimeError("Export operator Identity to ONNX failed!")


class ConvolutionNet(nn.Cell):
    def construct(self, x, weight, bias, stride, padding, dilation, transposed, output_padding, groups):
        return gen.Convolution()(x, weight, bias, stride, padding, dilation, transposed, output_padding, groups)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.0)
@pytest.mark.parametrize('bias, stride, padding, dilation, transposed, output_padding, groups',
                         [(0, 1, 0, 1, False, 0, 1), ([1], [2, 2], [1, 1], [2, 2], False, [0, 0], 1)])
def test_export_convolution(bias, stride, padding, dilation, transposed, output_padding, groups):
    """
    Feature: Export ops.Convolution to onnx
    Description: Export ops.Convolution to onnx
    Expectation: success
    """
    context.set_context(jit_config={"jit_level": "O0"}, mode=context.GRAPH_MODE, device_target="Ascend")
    np_x = np.ones(shape=[1, 1, 8, 8]).astype(np.float32)
    np_weight = np.ones(shape=[1, 1, 2, 2]).astype(np.float32)
    x = Tensor(np_x)
    weight = Tensor(np_weight)
    ms_bias = Tensor(bias, ms.float32) if bias else None
    net = ConvolutionNet()
    y = net(x, weight, ms_bias, stride, padding, dilation, transposed, output_padding, groups)
    export(net, x, weight, ms_bias, stride, padding, dilation,
           transposed, output_padding, groups, file_name='./convolution_onnx')
    if os.path.isfile("./convolution_onnx.onnx"):
        session = ort.InferenceSession("./convolution_onnx.onnx")
        inputs = {"x": np_x, "weight": np_weight}
        if bias:
            inputs["bias"] = bias
        output = session.run(None, inputs)[0]
        assert np.array_equal(output, y.asnumpy())
        os.remove("./convolution_onnx.onnx")
    else:
        raise RuntimeError("Export operator Convolution to ONNX failed!")


class Conv2DExtNet(nn.Cell):
    def construct(self, x, weight, bias, stride, padding, dilation, groups):
        return conv2d(x, weight, bias, stride, padding, dilation, groups)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0',
        card_mark='onecard', essential_mark='essential', mem_peak=2048.01)
@pytest.mark.parametrize('bias, stride, padding, dilation, groups',
                         [(0, 1, 0, 1, 1), ([1], [2, 2], [1, 1], [2, 2], 1)])
def test_export_conv2dext(bias, stride, padding, dilation, groups):
    """
    Feature: Export ops.Conv2DExt to onnx
    Description: Export ops.Conv2DExt to onnx
    Expectation: success
    """
    context.set_context(jit_config={"jit_level": "O0"}, mode=context.GRAPH_MODE, device_target="Ascend")
    np_x = np.ones(shape=[1, 1, 8, 8]).astype(np.float32)
    np_weight = np.ones(shape=[1, 1, 2, 2]).astype(np.float32)
    x = Tensor(np_x)
    weight = Tensor(np_weight)
    ms_bias = Tensor(bias, ms.float32) if bias else None
    net = Conv2DExtNet()
    y = net(x, weight, ms_bias, stride, padding, dilation, groups)
    export(net, x, weight, ms_bias, stride, padding, dilation, groups, file_name='./conv2dext_onnx')
    if os.path.isfile("./conv2dext_onnx.onnx"):
        session = ort.InferenceSession("./conv2dext_onnx.onnx")
        inputs = {"x": np_x, "weight": np_weight}
        if bias:
            inputs["bias"] = bias
        output = session.run(None, inputs)[0]
        assert np.array_equal(output, y.asnumpy())
        os.remove("./conv2dext_onnx.onnx")
    else:
        raise RuntimeError("Export operator Conv2DExt to ONNX failed!")


class ConvTranspose2DNet(nn.Cell):
    def construct(self, x, weight, bias, stride, padding, output_padding, groups, dilation):
        return gen.ConvTranspose2D()(x, weight, bias, stride, padding, output_padding, groups, dilation)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=1024.0)
@pytest.mark.parametrize('bias, stride, padding, output_padding, groups, dilation',
                         [(0, 2, 0, 1, 1, 2), ([1], [2, 2], [1, 1], [1, 1], 1, [2, 2])])
def test_export_convtranspose2d(bias, stride, padding, output_padding, groups, dilation):
    """
    Feature: Export ops.ConvTranspose2D to onnx
    Description: Export ops.ConvTranspose2D to onnx
    Expectation: success
    """
    np_x = np.ones(shape=[1, 1, 8, 8]).astype(np.float32)
    np_weight = np.ones(shape=[1, 1, 2, 2]).astype(np.float32)
    x = Tensor(np_x)
    weight = Tensor(np_weight)
    ms_bias = Tensor(bias, ms.float32) if bias else None
    net = ConvTranspose2DNet()
    y = net(x, weight, ms_bias, stride, padding, output_padding, groups, dilation)
    export(net, x, weight, ms_bias, stride, padding, output_padding,
           groups, dilation, file_name='./convtranspose2d_onnx')
    if os.path.isfile("./convtranspose2d_onnx.onnx"):
        session = ort.InferenceSession("./convtranspose2d_onnx.onnx")
        inputs = {"x": np_x, "weight": np_weight}
        if bias:
            inputs["bias"] = bias
        output = session.run(None, inputs)[0]
        assert np.array_equal(output, y.asnumpy())
        os.remove("./convtranspose2d_onnx.onnx")
    else:
        raise RuntimeError("Export operator ConvTranspose2D to ONNX failed!")


class UpsampleNearest2DNet(nn.Cell):
    def construct(self, x, output_size, scales):
        return gen.UpsampleNearest2D()(x, output_size, scales)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0',
        card_mark='onecard', essential_mark='essential', mem_peak=0.01)
@pytest.mark.parametrize('output_size, scales', [([8, 8], None), (None, [1.5, 1.5])])
def test_export_upsamplenearest2d(output_size, scales):
    """
    Feature: Export ops.UpsampleNearest2D to onnx
    Description: Export ops.UpsampleNearest2D to onnx
    Expectation: success
    """
    np_x = np.ones(shape=[1, 1, 4, 4]).astype(np.float32)
    x = Tensor(np_x)
    net = UpsampleNearest2DNet()
    y = net(x, output_size, scales)
    export(net, x, output_size, scales, file_name='./upsamplenearest2d_onnx')
    if os.path.isfile("./upsamplenearest2d_onnx.onnx"):
        session = ort.InferenceSession("./upsamplenearest2d_onnx.onnx")
        inputs = {"x": np_x}
        output = session.run(None, inputs)[0]
        assert np.array_equal(output, y.asnumpy())
        os.remove("./upsamplenearest2d_onnx.onnx")
    else:
        raise RuntimeError("Export operator UpsampleNearest2D to ONNX failed!")


class MeshgridNet(nn.Cell):
    def __init__(self, indexing):
        super().__init__()
        self.meshgrid_op = gen.Meshgrid(indexing)

    def construct(self, inputs):
        x, y, z = inputs
        x = x + 1
        a = (x, y, z)
        return self.meshgrid_op(a)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=1024.0)
@pytest.mark.parametrize('indexing', ['ij'])
def test_export_meshgrid(indexing):
    """
    Feature: Export ops.Meshgrid to onnx
    Description: Export ops.Meshgrid to onnx
    Expectation: success
    """
    np_x = np.array([1, 2, 3, 4]).astype(np.int32)
    np_y = np.array([5, 6, 7]).astype(np.int32)
    np_z = np.array([8, 9, 0, 1, 2]).astype(np.int32)
    x = Tensor(np_x)
    y = Tensor(np_y)
    z = Tensor(np_z)
    inputs = ms.mutable((x, y, z))
    net = MeshgridNet(indexing)
    ms_outputs = net(inputs)
    onnx_file = './meshgrid_onnx_' + indexing
    export(net, inputs, file_name=onnx_file)
    onnx_file = onnx_file + '.onnx'
    if os.path.isfile(onnx_file):
        session = ort.InferenceSession(onnx_file)
        inputs = {"inputs_0": np_x, "inputs_1": np_y, "inputs_2": np_z}
        outputs = session.run(None, inputs)
        for i, output in zip(ms_outputs, outputs):
            assert np.array_equal(i.asnumpy(), output), f" ms:{i.shape}, onnx:{o.shape}"
        os.remove(onnx_file)
    else:
        raise RuntimeError("Export operator Meshgrid to ONNX failed!")


class StackExtNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.stack_ext_op = gen.StackExt()

    def construct(self, x, y, z):
        return self.stack_ext_op((x, y, z))


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=1024.0)
def test_export_stackext():
    """
    Feature: Export ops.StackExt to onnx
    Description: Export ops.StackExt to onnx
    Expectation: success
    """
    np_x = np.array([1, 2, 3]).astype(np.int32)
    np_y = np.array([4, 5, 6]).astype(np.int32)
    np_z = np.array([7, 8, 9]).astype(np.int32)
    x = Tensor(np_x)
    y = Tensor(np_y)
    z = Tensor(np_z)
    net = StackExtNet()
    ms_outputs = net(x, y, z)
    onnx_file = './stackext_onnx'
    export(net, x, y, z, file_name=onnx_file)
    onnx_file = onnx_file + '.onnx'
    if os.path.isfile(onnx_file):
        session = ort.InferenceSession(onnx_file)
        inputs = {"x": np_x, "y": np_y, "z": np_z}
        outputs = session.run(None, inputs)[0]
        for i, output in zip(ms_outputs, outputs):
            assert np.array_equal(i.asnumpy(), output), f" ms:{i.shape}, onnx:{output.shape}"
        os.remove(onnx_file)
    else:
        raise RuntimeError("Export operator Stack to ONNX failed!")


class DenseNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.dense_op = gen.Dense()

    def construct(self, x, w, b):
        return self.dense_op(x, w, b)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0',
        card_mark='onecard', essential_mark='essential', mem_peak=1024.01)
def test_export_dense():
    """
    Feature: Export ops.Dense to onnx
    Description: Export ops.Dense to onnx
    Expectation: success
    """
    np_x = np.arange(100 * 16 * 10).reshape(100, 16, 10).astype(np.float32)
    x = Tensor(np_x)
    np_w = np.arange(24 * 10).reshape(24, 10).astype(np.float32)
    w = Tensor(np_w)
    np_b = np.array(5).astype(np.float32)
    b = Tensor(np_b)
    net = DenseNet()
    ms_output = net(x, w, b)
    onnx_file = './dense_onnx'
    export(net, x, w, b, file_name=onnx_file)
    onnx_file = onnx_file + '.onnx'
    if os.path.isfile(onnx_file):
        session = ort.InferenceSession(onnx_file)
        inputs = {"x": np_x, "w": np_w, "b": np_b}
        output = session.run(None, inputs)[0]
        assert np.allclose(ms_output.asnumpy(), output, 1e-3, 1e-3), f" ms:{ms_output}, onnx:{output}"
        os.remove(onnx_file)
    else:
        raise RuntimeError("Export operator Dense to ONNX failed!")


class ArgMaxWithValueNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.argmaxwithvalue_op = ops.ArgMaxWithValue(keep_dims=True)

    def construct(self, x, flag):
        if flag == 0:
            return self.argmaxwithvalue_op(x)
        arg, value = self.argmaxwithvalue_op(x)
        arg = arg + 1
        value = value + 1
        if flag == 1:
            return arg, value
        if flag == 2:
            return arg
        if flag == 3:
            return value
        raise RuntimeError("When call ArgMaxWithValueNet construct, the flag should be less than 3!")


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=1024.0)
def test_export_argmaxwithvalue():
    """
    Feature: Export ops.ArgMaxWithValue to onnx
    Description: Export ops.ArgMaxWithValue to onnx
    Expectation: success
    """
    np_x = np.array([0.0, 0.4, 0.6, 0.7, 0.1]).astype(np.float32)
    x = Tensor(np_x)
    net = ArgMaxWithValueNet()
    for i in range(4):
        ms_output = net(x, i)
        onnx_file = './argmaxwithvalue_onnx'
        export(net, x, i, file_name=onnx_file, output_names=["out1"])
        onnx_file = onnx_file + '.onnx'
        if os.path.isfile(onnx_file):
            session = ort.InferenceSession(onnx_file)
            inputs = {"x": np_x}
            if i == 0 or i == 1:
                output = session.run(None, inputs)
                assert np.allclose(ms_output[0].asnumpy(),
                                   output[0], 1e-3, 1e-3), f" ms:{ms_output[0]}, onnx:{output[1]}"
                assert np.allclose(ms_output[1].asnumpy(),
                                   output[1], 1e-3, 1e-3), f" ms:{ms_output[1]}, onnx:{output[1]}"
            else:
                output = session.run(None, inputs)[0]
                assert np.allclose(ms_output.asnumpy(), output, 1e-3, 1e-3), f" ms:{ms_output}, onnx:{output}"
            os.remove(onnx_file)
        else:
            raise RuntimeError("Export operator ArgMaxWithValue to ONNX failed!")


class StridedSliceNet(nn.Cell):
    def construct(self, x):
        return P.StridedSlice()(x, (0, 0), (1, 1), (1, 1))


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=1024.0)
def test_export_stridedslice():
    """
    Feature: Export ops.StridedSlice to onnx
    Description: Export ops.StridedSlice to onnx
    Expectation: success
    """
    np_x = np.array([[0, 1], [2, 3]]).astype(np.float32)
    x = Tensor(np_x)
    net = StridedSliceNet()
    ms_output = net(x)
    onnx_file = './strided_slice_onnx.onnx'
    export(net, x, file_name=onnx_file)
    if os.path.isfile(onnx_file):
        session = ort.InferenceSession(onnx_file)
        inputs = {"x": np_x}
        output = session.run(None, inputs)[0]
        assert np.array_equal(ms_output.asnumpy(), output), f" ms:{ms_output}, onnx:{output}"
        os.remove(onnx_file)
    else:
        raise RuntimeError("Export operator StridedSlice to ONNX failed!")


class SumExtNet(nn.Cell):
    def construct(self, x):
        return mint.sum(x, dim=2, keepdim=True)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=1024.0)
def test_export_sumext():
    """
    Feature: Export mint.sum to onnx
    Description: Export mint.sum to onnx
    Expectation: success
    """
    np_x = np.array([[[1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2], [3, 3, 3, 3, 3, 3]],
                     [[4, 4, 4, 4, 4, 4], [5, 5, 5, 5, 5, 5], [6, 6, 6, 6, 6, 6]],
                     [[7, 7, 7, 7, 7, 7], [8, 8, 8, 8, 8, 8], [9, 9, 9, 9, 9, 9]]]).astype(np.float32)
    x = Tensor(np_x)
    net = SumExtNet()
    ms_output = net(x)
    onnx_file = './sum_ext_onnx.onnx'
    export(net, x, file_name=onnx_file)
    if os.path.isfile(onnx_file):
        session = ort.InferenceSession(onnx_file)
        inputs = {"x": np_x}
        output = session.run(None, inputs)[0]
        assert np.array_equal(ms_output.asnumpy(), output), f" ms:{ms_output}, onnx:{output}"
        os.remove(onnx_file)
    else:
        raise RuntimeError("Export operator SumExt to ONNX failed!")


class SquareNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.square = ops.Square()

    def construct(self, x):
        return self.square(x)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=1024.0)
def test_export_square():
    """
    Feature: Export ops.Square to onnx
    Description: Export ops.Square to onnx
    Expectation: success
    """
    np_x = np.random.randn(2, 4, 8, 16, 8, 4).astype(np.float16)
    x = Tensor(np_x)
    net = SquareNet()
    ms_output = net(x)
    onnx_file = './square_onnx.onnx'
    export(net, x, file_name=onnx_file)
    if os.path.isfile(onnx_file):
        session = ort.InferenceSession(onnx_file)
        inputs = {"x": np_x}
        output = session.run(None, inputs)[0]
        assert np.array_equal(ms_output.asnumpy(), output), f" ms:{ms_output}, onnx:{output}"
        os.remove(onnx_file)
    else:
        raise RuntimeError("Export operator SiLU to ONNX failed!")


class SiLUNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.silu = nn.SiLU()

    def construct(self, x):
        return self.silu(x)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=1024.01)
def test_export_silu():
    """
    Feature: Export nn.SiLU to onnx
    Description: Export nn.SiLU to onnx
    Expectation: success
    """
    np_x = np.ones(shape=[2, 3, 4]).astype(np.float32)
    x = Tensor(np_x)
    net = SiLUNet()
    ms_output = net(x)
    onnx_file = './silu_onnx.onnx'
    export(net, x, file_name=onnx_file)
    if os.path.isfile(onnx_file):
        session = ort.InferenceSession(onnx_file)
        inputs = {"x": np_x}
        output = session.run(None, inputs)[0]
        assert np.array_equal(ms_output.asnumpy(), output), f" ms:{ms_output}, onnx:{output}"
        os.remove(onnx_file)
    else:
        raise RuntimeError("Export operator SiLU to ONNX failed!")


class ModNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.mod = ops.Mod()

    def construct(self, x, y):
        return self.mod(x, y)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0',
        card_mark='onecard', essential_mark='essential', mem_peak=1024.01)
def test_export_mod():
    """
    Feature: Export ops.Mod to onnx
    Description: Export ops.Mod to onnx
    Expectation: success
    """
    np_x = np.array([-4.0, 5.0, 6.0]).astype(np.float16)
    np_y = np.array([3.0, 2.0, 3.0]).astype(np.float16)
    x = Tensor(np_x)
    y = Tensor(np_y)
    net = ModNet()
    ms_output = net(x, y)
    onnx_file = './mod_onnx.onnx'
    export(net, x, y, file_name=onnx_file)
    if os.path.isfile(onnx_file):
        session = ort.InferenceSession(onnx_file)
        inputs = {"x": np_x, "y": np_y}
        output = session.run(None, inputs)[0]
        assert np.array_equal(ms_output.asnumpy(), output), f" ms:{ms_output}, onnx:{output}"
        os.remove(onnx_file)
    else:
        raise RuntimeError("Export operator SiLU to ONNX failed!")


class AvgPoolNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.avgpool = ops.AvgPool(pad_mode='VALID', kernel_size=2, strides=1)

    def construct(self, x):
        return self.avgpool(x)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0',
        card_mark='onecard', essential_mark='essential', mem_peak=2048.01)
def test_export_avgpool():
    """
    Feature: Export ops.AvgPool to onnx
    Description: Export ops.AvgPool to onnx
    Expectation: success
    """
    np_x = np.arange(1 * 3 * 3 * 4).reshape(1, 3, 3, 4).astype(np.float32)
    x = Tensor(np_x)
    net = AvgPoolNet()
    ms_output = net(x)
    onnx_file = './avgpool_onnx.onnx'
    export(net, x, file_name=onnx_file)
    if os.path.isfile(onnx_file):
        session = ort.InferenceSession(onnx_file)
        inputs = {"x": np_x}
        output = session.run(None, inputs)[0]
        assert np.array_equal(ms_output.asnumpy(), output), f" ms:{ms_output}, onnx:{output}"
        os.remove(onnx_file)
    else:
        raise RuntimeError("Export operator SiLU to ONNX failed!")


class SoftmaxNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(axis=1)

    def construct(self, x):
        return self.softmax(x)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=1024.0)
def test_export_softmax():
    """
    Feature: Export nn.Softmax to onnx
    Description: Export nn.Softmax to onnx
    Expectation: success
    """
    np_x = np.random.uniform(low=0, high=1, size=(1, 3, 640)).astype(np.float32)
    x = Tensor(np_x)
    net = SoftmaxNet()
    ms_output = net(x)
    onnx_file = './softmax_onnx.onnx'
    export(net, x, file_name=onnx_file)
    if os.path.isfile(onnx_file):
        session = ort.InferenceSession(onnx_file)
        inputs = {"x": np_x}
        output = session.run(None, inputs)[0]
        assert np.allclose(ms_output.asnumpy(), output, rtol=1e-4, atol=1e-4), f" ms:{ms_output}, onnx:{output}"
        os.remove(onnx_file)
    else:
        raise RuntimeError("Export operator Softmax to ONNX failed!")


class SqueezeNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.squeeze = ops.Squeeze(axis=(1, 2))

    def construct(self, x):
        return self.squeeze(x)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=1024.0)
def test_export_squeeze():
    """
    Feature: Export ops.Squeeze to onnx
    Description: Export ops.Squeeze to onnx
    Expectation: success
    """
    np_x = np.random.uniform(low=0, high=1, size=(1, 3, 4, 5)).astype(np.float32)
    x = Tensor(np_x)
    net = SqueezeNet()
    ms_output = net(x)
    onnx_file = './squeeze_onnx.onnx'
    export(net, x, file_name=onnx_file)
    if os.path.isfile(onnx_file):
        session = ort.InferenceSession(onnx_file)
        inputs = {"x": np_x}
        output = session.run(None, inputs)[0]
        assert np.array_equal(ms_output.asnumpy(), output), f" ms:{ms_output}, onnx:{output}"
        os.remove(onnx_file)
    else:
        raise RuntimeError("Export operator Squeeze to ONNX failed!")


class MulsNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.y = 3.0

    def construct(self, x):
        z = x * self.y
        return z


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=1024.01)
def test_export_muls():
    """
    Feature: Export Muls to onnx
    Description: Export Muls to onnx
    Expectation: success
    """
    np_x = np.random.uniform(low=0, high=1, size=(1, 3, 640)).astype(np.float32)
    x = Tensor(np_x)
    net = MulsNet()
    ms_output = net(x)
    onnx_file = './muls_onnx.onnx'
    export(net, x, file_name=onnx_file)
    if os.path.isfile(onnx_file):
        session = ort.InferenceSession(onnx_file)
        inputs = {"x": np_x}
        output = session.run(None, inputs)[0]
        assert np.array_equal(ms_output.asnumpy(), output), f" ms:{ms_output}, onnx:{output}"
        os.remove(onnx_file)
    else:
        raise RuntimeError("Export operator Muls to ONNX failed!")


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0',
        card_mark='onecard', essential_mark='essential', mem_peak=1024.01)
def test_export_muls_with_saved_dirs():
    """
    Feature: Export Muls to onnx and set dirs
    Description: Export Muls to onnx
    Expectation: success
    """
    np_x = np.random.uniform(low=0, high=1, size=(1, 3, 640)).astype(np.float32)
    x = Tensor(np_x)
    net = MulsNet()
    ms_output = net(x)
    onnx_file = './muls_test/test1/test2/muls_onnx.onnx'
    export(net, x, file_name=onnx_file)
    if os.path.isfile(onnx_file):
        session = ort.InferenceSession(onnx_file)
        inputs = {"x": np_x}
        output = session.run(None, inputs)[0]
        assert np.array_equal(ms_output.asnumpy(), output), f" ms:{ms_output}, onnx:{output}"
        os.remove(onnx_file)
    else:
        raise RuntimeError("Export operator Muls to ONNX failed!")


class ConvReluModel(nn.Cell):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.SequentialCell(
            nn.Conv2d(3, 5, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=1, stride=1),
        )

    def construct(self, x):
        x = self.conv_layers(x)
        return x


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.0)
def test_export_param_keep_initializers():
    """
    Feature: Export model with export_params=False and keep_initializers=True.
    Description: Export net to onnx.
    Expectation: export success, runtime success.
    """
    np_x = np.random.uniform(low=0, high=1, size=(1, 3, 5, 5)).astype(np.float32)
    x = Tensor(np_x)
    net = ConvReluModel()
    ms_output = net(x)
    onnx_file_origin_weight = './origin_weight.onnx'
    onnx_file_export_params_false = './export_false.onnx'
    onnx_file_keep_true = './keep_true.onnx'
    export(net, x, file_name=onnx_file_origin_weight)
    export(net, x, file_name=onnx_file_export_params_false, export_params=False)
    export(net, x, file_name=onnx_file_keep_true, keep_initializers_as_inputs=True)
    if not os.path.isfile(onnx_file_origin_weight):
        raise RuntimeError("Export operator ConvReluModel to ONNX failed!")
    if not os.path.isfile(onnx_file_export_params_false):
        raise RuntimeError("Export operator ConvReluModel to ONNX with export_params=False failed!")
    if not os.path.isfile(onnx_file_keep_true):
        raise RuntimeError("Export operator ConvReluModel to ONNX with keep_initializers_as_inputs=True failed!")
    session1 = ort.InferenceSession(onnx_file_origin_weight)
    session2 = ort.InferenceSession(onnx_file_export_params_false)
    session3 = ort.InferenceSession(onnx_file_keep_true)
    np_weight = np.random.uniform(low=0, high=1, size=(5, 3, 1, 1)).astype(np.float32)
    inputs = {"x": np_x, "conv_layers.0.weight": np_weight}
    output1 = session1.run(None, {"x": np_x})[0]
    assert np.allclose(ms_output, output1, rtol=1e-3, atol=1e-3), "origin weight not equal, please check"
    output2 = session2.run(None, inputs)[0]
    output3 = session3.run(None, inputs)[0]
    assert np.array_equal(output2,
                          output3), "Result of export_params=False and keep_initializers_as_inputs=True are not equal"

    os.remove(onnx_file_origin_weight)
    os.remove(onnx_file_export_params_false)
    os.remove(onnx_file_keep_true)


class DynamicNetwork(nn.Cell):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 2, kernel_size=3, pad_mode='same')
        self.bn1 = nn.BatchNorm2d(2)
        self.relu1 = nn.ReLU()

    def construct(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        return x


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0',
        card_mark='onecard', essential_mark='essential', mem_peak=0.01)
def test_dynamic_axes():
    """
    Feature: Export model with dynamic_axes and different input shape for export and onnxruntime
    Description: Export net to onnx.
    Expectation: export success, runtime success.
    """
    ms.set_context(mode=ms.GRAPH_MODE)
    input_tensor = ms.Tensor(shape=(8, 3, 2, 2), dtype=ms.float32, init=ms.common.initializer.One())
    input_tensor1 = ms.Tensor(shape=(6, 3, 2, 2), dtype=ms.float32, init=ms.common.initializer.One())
    net = DynamicNetwork()
    ms_output = net(input_tensor)
    onnx_file = './dynamic_batch_size.onnx'
    export(net, input_tensor1, file_name=onnx_file, input_names=['input0'],
           output_names=["out0"], dynamic_axes={"input0": {0: "batch_size"}})
    session = ort.InferenceSession(onnx_file)
    inputs = {"input0": input_tensor.asnumpy()}
    onnxruntime_output = session.run(None, inputs)[0]
    assert np.allclose(ms_output.asnumpy(), onnxruntime_output, rtol=1e-3, atol=1e-3)
    os.remove(onnx_file)
