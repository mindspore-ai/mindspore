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
""" test_dequant_swiglu_quant.py """
import numpy as np
import pytest
import torch
import mindspore as ms
from mindspore import nn
from mindspore import Tensor, context, ops
from tests.mark_utils import arg_mark

class DequantSwigluQuantNet(nn.Cell):
    def __init__(self):
        super(DequantSwigluQuantNet, self).__init__()
        self.dequant_swiglu_quant = ops.auto_generate.DequantSwigluQuant()

    def construct(self, x, weight_scale, activation_scale, bias, quant_scale, quant_offset, group_index,
                  activate_left=False, quant_mode="static"):
        y_out, scale_out = self.dequant_swiglu_quant(
            x, weight_scale, activation_scale, bias, quant_scale, quant_offset, group_index, activate_left,
            quant_mode)
        return y_out, scale_out

def golden_dequant_swiglu_quant_torch(
        x,
        weight_scale,
        activation_scale,
        bias,
        quant_scale,
        quant_offset,
        group_num,
        activate_left,
        quant_mode,
    ):
    x = x.to(torch.float32)
    weight_scale = weight_scale.to(torch.float32)
    activation_scale = activation_scale.to(torch.float32)
    res = torch.mul(x, weight_scale)
    res = torch.mul(res, activation_scale)
    out = torch.chunk(res, 2, dim=-1)
    if activate_left:
        self_tensor = out[0]
        other = out[1]
    else:
        self_tensor = out[1]
        other = out[0]
    output = torch.nn.functional.silu(self_tensor) * other
    if quant_scale is not None:
        output = torch.mul(output, quant_scale)
    scale_out = torch.zeros([x.shape[0]], dtype=torch.float32)
    if quant_mode == "dynamic":
        absolute = torch.abs(output)
        max_values = torch.amax(absolute, dim=-1)
        scale_out = max_values / 127.0
        max_values = 127.0 / max_values
        output = output * max_values.unsqueeze(1)
    output = torch.clamp(output, -128, 127)
    output = torch.round(output)
    return output.to(torch.int8).cpu().numpy(), scale_out.cpu().numpy()

def golden_dequant_swiglu_quant_torch_group(
        x,
        weight_scale,
        activation_scale,
        bias,
        quant_scale,
        quant_offset,
        group_num,
        activate_left,
        quant_mode,
    ):
    x = x.to(torch.float32)
    weight_scale = weight_scale.to(torch.float32)
    activation_scale = activation_scale.to(torch.float32)

    y_golden = torch.zeros((x.shape[0], x.shape[1] // 2))
    scale_golden = torch.zeros((x.shape[0]))
    start = 0
    for i, end in enumerate(group_num):
        if end == 0:
            continue
        x_group = x[start:start + end, :]
        weight_scale_group = weight_scale[i, :].unsqueeze(0)
        res_group = torch.mul(x_group, weight_scale_group)
        res_group = torch.mul(res_group, activation_scale[start:start + end, :])
        out_group = torch.chunk(res_group, 2, dim=-1)
        if activate_left:
            self_tensor = out_group[0]
            other = out_group[1]
        else:
            self_tensor = out_group[1]
            other = out_group[0]
        output = torch.nn.functional.silu(self_tensor) * other
        scale_out = torch.zeros([x_group.shape[0]], dtype=torch.float32)
        if quant_mode == "dynamic":
            absolute = torch.abs(output)
            max_values = torch.amax(absolute, dim=-1)
            scale_out = max_values / 127.0
            max_values = 127.0 / max_values
            output = output * max_values.unsqueeze(1)
        output = torch.clamp(output, -128, 127)
        output = torch.round(output)
        y_golden[start:start + end, :] = output
        scale_golden[start:start + end] = scale_out
        start = start + end

    return y_golden.to(torch.int8).cpu().numpy(), scale_golden.cpu().numpy()


def custom_compare(output, expect, mstype):
    if mstype == ms.float16:
        limit = 0.004
    elif mstype == ms.bfloat16:
        limit = 0.03
    elif mstype == ms.float32:
        limit = 0.004
    else:
        limit = 0.004

    out_flatten = output.flatten()
    expect_flatten = expect.flatten()

    err_cnt = 0
    size = len(out_flatten)
    err_cnt = np.sum(np.abs(out_flatten - expect_flatten) /
                     np.abs(expect_flatten) > limit).astype(np.int32)
    limit_cnt = int(size * limit)
    if err_cnt > limit_cnt:
        print("[FAILED]", "err_cnt = ", err_cnt, "/", limit_cnt)
        return False

    print("[SUCCESS]", "err_cnt = ", err_cnt, "/", limit_cnt)
    return True

def DequantSwigluQuantNetTest(net, x, weight_scale, activation_scale, bias, quant_scale, quant_offset,
                              group_index, activate_left, quant_mode):
    valid_data_length = x.shape[0] if group_index is None else group_index.sum()
    bias_tensor = None if bias is None else Tensor(bias)
    quant_scale_tensor = None if quant_scale is None else Tensor(quant_scale)
    quant_offset_tensor = None if quant_offset is None else Tensor(quant_offset)
    group_index_tensor = None if group_index is None else Tensor(group_index)
    y_out, scale_out = net(
        Tensor(x),
        Tensor(weight_scale),
        Tensor(activation_scale),
        bias_tensor,
        quant_scale_tensor,
        quant_offset_tensor,
        group_index_tensor,
        activate_left,
        quant_mode)

    x_torch = torch.from_numpy(x)
    weight_scale_torch = torch.from_numpy(weight_scale)
    activatition_scale_torch = torch.from_numpy(activation_scale)
    bias_torch = None if bias is None else torch.from_numpy(bias)
    quant_scale_torch = None if quant_scale is None else torch.from_numpy(quant_scale)
    quant_offset_torch = None if quant_offset is None else torch.from_numpy(quant_offset)
    group_index_torch = None if group_index is None else torch.from_numpy(group_index)
    if group_index_torch is None:
        y_out_golden, scale_out_golden = golden_dequant_swiglu_quant_torch(
            x_torch,
            weight_scale_torch,
            activatition_scale_torch,
            bias_torch,
            quant_scale_torch,
            quant_offset_torch,
            group_index_torch,
            activate_left,
            quant_mode)
    else:
        y_out_golden, scale_out_golden = golden_dequant_swiglu_quant_torch_group(
            x_torch,
            weight_scale_torch,
            activatition_scale_torch,
            bias_torch,
            quant_scale_torch,
            quant_offset_torch,
            group_index_torch,
            activate_left,
            quant_mode)

    y_diff = y_out.asnumpy()[:valid_data_length, :].flatten() - y_out_golden[:valid_data_length, :].flatten()
    y_max_diff = np.max(np.abs(y_diff))
    y_compare_result = y_max_diff <= 1
    scale_compare_result = custom_compare(scale_out.asnumpy(), scale_out_golden, ms.float32)
    assert y_compare_result and scale_compare_result, "dequant_swiglu_quant compare failed"

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='unessential', mem_peak=60.00)
@pytest.mark.parametrize('activate_left', [True, False])
@pytest.mark.parametrize('context_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_dequant_swiglu_quant_non_group_index(activate_left, context_mode):
    '''
    Feature:aclnnDequantSwigluQuant kernel.
    Description: test for aclnnDequantSwigluQuant ops.
    Expectation:should pass for all testcases.
    '''
    ms.set_context(device_target="Ascend", mode=context_mode)
    ms.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})
    # ms.set_context(save_graphs=True, save_graphs_path="./dequant_swiglu_quant_ir")

    tokensNum = 4608
    H = 1024
    x = np.random.randint(-10, 10, size=(tokensNum, 2 * H), dtype=np.int32)
    weight_scale = np.random.randn(2 * H).astype(np.float32)
    activation_scale = np.random.randn(tokensNum, 1).astype(np.float32)
    bias = None
    quant_scale = np.random.randn(1, H).astype(np.float32)
    quant_offset = None
    group_index = None
    quant_mode = "dynamic"

    net = DequantSwigluQuantNet()
    DequantSwigluQuantNetTest(net, x, weight_scale, activation_scale, bias, quant_scale, quant_offset,
                              group_index, activate_left, quant_mode)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='unessential', mem_peak=28.00)
@pytest.mark.parametrize('param', [[128, 4608]])
@pytest.mark.parametrize('activate_left', [True, False])
@pytest.mark.parametrize('context_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_dequant_swiglu_quant_group_x_weight_activation(param, activate_left, context_mode):
    '''
    Feature:aclnnDequantSwigluQuant kernel.
    Description: test for aclnnDequantSwigluQuant ops.
    Expectation:should pass for all testcases.
    '''
    ms.set_context(device_target="Ascend", mode=context_mode)
    ms.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})
    # ms.set_context(save_graphs=True, save_graphs_path="./dequant_swiglu_quant_ir")

    tokensNum, H = param
    x = np.random.randint(-10, 10, size=(tokensNum, 2 * H), dtype=np.int32)
    weight_scale = np.random.randn(2 * H).astype(np.float32)
    activation_scale = np.random.randn(tokensNum, 1).astype(np.float32)
    bias = None
    quant_scale = None
    quant_offset = None
    group_index = None
    quant_mode = "static"

    net = DequantSwigluQuantNet()
    DequantSwigluQuantNetTest(net, x, weight_scale, activation_scale, bias, quant_scale, quant_offset,
                              group_index, activate_left, quant_mode)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='unessential', mem_peak=148.00)
@pytest.mark.parametrize('activate_left', [True, False])
@pytest.mark.parametrize('context_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_dequant_swiglu_quant_group(activate_left, context_mode):
    '''
    Feature:aclnnDequantSwigluQuant kernel.
    Description: test for aclnnDequantSwigluQuant ops.
    Expectation:should pass for all testcases.
    '''
    ms.set_context(device_target="Ascend", mode=context_mode)
    ms.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})
    # ms.set_context(save_graphs=True, save_graphs_path="./dequant_swiglu_quant_ir")

    tokensNum, H, groupNum = [8192, 2048, 8]
    x = np.random.randint(-10, 10, size=(tokensNum, 2 * H), dtype=np.int32)
    weight_scale = np.random.randn(groupNum, 2 * H).astype(np.float32)
    activation_scale = np.random.randn(tokensNum, 1).astype(np.float32)
    bias = None
    quant_scale = None
    quant_offset = None
    group_index = np.random.randint(1, 100, size=(groupNum,))
    current_sum = group_index.sum()
    scale = tokensNum / current_sum
    group_index = np.floor(group_index * scale).astype(np.int64)
    assert group_index.sum() <= tokensNum, "the sum of group_index cannot be bigger than tokensNum"
    quant_mode = "dynamic"

    net = DequantSwigluQuantNet()
    DequantSwigluQuantNetTest(net, x, weight_scale, activation_scale, bias, quant_scale, quant_offset,
                              group_index, activate_left, quant_mode)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='unessential', mem_peak=148.00)
@pytest.mark.parametrize('context_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_dequant_swiglu_quant_group_dynamic_shape(context_mode):
    '''
    Feature:aclnnDequantSwigluQuant kernel.
    Description: test for aclnnDequantSwigluQuant ops.
    Expectation:should pass for all testcases.
    '''
    ms.set_context(device_target="Ascend", mode=context_mode)
    ms.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})
    # ms.set_context(save_graphs=True, save_graphs_path="./dequant_swiglu_quant_ir")

    tokensNum, H, groupNum = [8192, 2048, 8]
    x = np.random.randint(-10, 10, size=(tokensNum, 2 * H), dtype=np.int32)
    weight_scale = np.random.randn(groupNum, 2 * H).astype(np.float32)
    activation_scale = np.random.randn(tokensNum, 1).astype(np.float32)
    bias = None
    quant_scale = None
    quant_offset = None
    group_index = np.random.randint(1, 100, size=(groupNum,))
    current_sum = group_index.sum()
    scale = tokensNum / current_sum
    group_index = np.floor(group_index * scale).astype(np.int64)
    assert group_index.sum() <= tokensNum, "the sum of group_index cannot be bigger than tokensNum"
    activate_left = False
    quant_mode = "dynamic"

    x_dyn = ms.Tensor(shape=[None, None], dtype=ms.int32)
    weight_scale_dyn = ms.Tensor(shape=[None, None], dtype=ms.float32)
    activation_scale_dyn = ms.Tensor(shape=[None, None], dtype=ms.float32)
    group_index_dyn = ms.Tensor(shape=[None], dtype=ms.int64)

    net = DequantSwigluQuantNet()
    net.set_inputs(x=x_dyn, weight_scale=weight_scale_dyn, activation_scale=activation_scale_dyn,
                   group_index=group_index_dyn)
    DequantSwigluQuantNetTest(net, x, weight_scale, activation_scale, bias, quant_scale, quant_offset,
                              group_index, activate_left, quant_mode)

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='unessential', mem_peak=20.00)
@pytest.mark.parametrize('context_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_dequant_swiglu_quant_dynamic_shape(context_mode):
    '''
    Feature:aclnnDequantSwigluQuant kernel.
    Description: test for aclnnDequantSwigluQuant ops.
    Expectation:should pass for all testcases.
    '''
    ms.set_context(device_target="Ascend", mode=context_mode)
    ms.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})
    # ms.set_context(save_graphs=True, save_graphs_path="./dequant_swiglu_quant_ir")

    tokensNum, H, _ = [32, 64, 2]
    x = np.random.randint(-10, 10, size=(tokensNum, 2 * H), dtype=np.int32)
    weight_scale = np.random.randn(2 * H).astype(np.float32)
    activation_scale = np.random.randn(tokensNum, 1).astype(np.float32)
    bias = None
    quant_scale = None
    quant_offset = None
    group_index = None
    activate_left = False
    quant_mode = "static"

    x_dyn = ms.Tensor(shape=[None, None], dtype=ms.int32)
    weight_scale_dyn = ms.Tensor(shape=[None], dtype=ms.float32)
    activation_scale_dyn = ms.Tensor(shape=[None, None], dtype=ms.float32)

    net = DequantSwigluQuantNet()
    net.set_inputs(x=x_dyn, weight_scale=weight_scale_dyn, activation_scale=activation_scale_dyn)
    DequantSwigluQuantNetTest(net, x, weight_scale, activation_scale, bias, quant_scale, quant_offset,
                              group_index, activate_left, quant_mode)
