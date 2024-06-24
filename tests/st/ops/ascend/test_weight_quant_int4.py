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

import numpy as np
import pytest
from mindspore.common import dtype as mstype
from mindspore.nn import Cell, GraphCell
from mindspore.ops.auto_generate import WeightQuantBatchMatmul

import mindspore as ms
from mindspore import Parameter, Tensor, export, JitConfig


class WeightQuantBatchMatmulNet(Cell):
    """
    WeightQuantBatchMatmulNet.
    """

    def __init__(self, weight=None, transpose_x=False, transpose_weight=False, antiquant_group_size=0):
        super().__init__()
        self.wqbmm = WeightQuantBatchMatmul(transpose_x, transpose_weight, antiquant_group_size)
        self.weight = weight

    def construct(self, x, antiquant_scale, antiquant_offset, quant_scale, quant_offset, bias):
        out = self.wqbmm(x, self.weight, antiquant_scale, antiquant_offset, quant_scale, quant_offset, bias)
        return out


def np_antiquant(np_data, scale=1.0, offset=0.):
    """mindspore implemented antiquant"""
    np_antiquant_data = np_data.astype(np.float16)
    if offset is None:
        offset = 0
    np_antiquant_data = scale * (np_antiquant_data - offset)
    np_antiquant_data = np_antiquant_data.astype(np.float16)
    return np_antiquant_data


def np_antiquant_pergroup(np_data, scale, offset, group_num):
    """mindspore implemented antiquant"""
    weight_shape = np_data.shape
    assert len(weight_shape) == 2
    k_size, _ = weight_shape
    group_size = k_size // group_num

    np_antiquant_data = np_data.astype(np.float32)
    for i in range(group_num):
        np_antiquant_data[i * group_size : (i+1) * group_size, :] = \
            (np_antiquant_data[i * group_size : (i+1) * group_size, :] +
             offset[i, :].astype(np.float16)) * scale[i, :].astype(np.float32)
    return np_antiquant_data


def np_int4data_pack_to_int8(np_data):
    """pack int4(represented in int8) data to int8(int4*2)"""
    np_data = np_data.astype(np.int8)
    np_data &= 0x000F
    np_data[::, 0::2] <<= 0
    np_data[::, 1::2] <<= 4
    np_int4_data = np_data[::, 0::2] | np_data[::, 1::2]
    return np_int4_data


def np_quant_int4(np_data, scale=1.0, offset=0.0):
    """quant data to int4 data"""
    np_quant_int8_data = np.round(np_data / scale + offset).astype(np.int8)
    np_quant_int8_data = np.clip(np_quant_int8_data, -8, 7).astype(np.int8)
    np_quant_int4_data = np_int4data_pack_to_int8(np_quant_int8_data)
    return np_quant_int8_data, np_quant_int4_data


def np_quant_int4_pergroup(np_data, scale=1.0, offset=0.0, group_size=2):
    """quant data to int4 data"""
    weight_shape = np_data.shape
    assert len(weight_shape) == 2
    k_size, _ = weight_shape

    for i in range(k_size):
        group_size_dim = i // group_size
        np_data[i, :] = (np_data[i, :] / scale[group_size_dim, :].astype(np.float16)) \
                        + offset[group_size_dim, :].astype(np.float16)
    np_quant_int8_data = np.clip(np_data, -8, 7).astype(np.int8)
    np_quant_int4_data = np_int4data_pack_to_int8(np_quant_int8_data)
    return np_quant_int8_data, np_quant_int4_data


def np_gen_int4_data(scale, offset=0.):
    """
    gen fp16_activation and int4_weight for test
    :param scale: scale for quant
    :return: activation with dtype fp16, weight width dtype int4
    """
    np_x = np.random.rand(8, 8).astype(np.float16)
    np_weight = np.linspace(-0.64, 0.64, 64).astype(np.float16).reshape((8, 8))
    np_quant_int8_data, np_quant_int4_data = np_quant_int4(np_weight, scale, offset)
    return np_x, np_quant_int8_data, np_quant_int4_data


def np_gen_int4_data_perchannel():
    """
    gen fp16_activation and int4_weight for test
    :param scale: scale for quant

    :return: activation with dtype fp16, weight width dtype int4
    """
    np_x = np.random.rand(8, 8).astype(np.float16)
    np_weight = np.linspace(-0.64, 0.64, 64).astype(np.float16).reshape((8, 8))
    scale = np.random.rand(8).astype(np.float16)
    offset = np.random.rand(8).astype(np.float16)
    np_quant_int8_data, np_quant_int4_data = np_quant_int4(np_weight, scale, offset)
    return np_x, np_quant_int8_data, np_quant_int4_data, scale, offset


def np_gen_int4_data_pergroup(batch_size=4,
                              channel_in=8,
                              channel_out=8,
                              group_size=2):
    """
    gen fp16_activation and int4_weight for test
    :param scale: scale for quant

    :return: activation with dtype fp16, weight width dtype int4
    """
    np_x = np.random.rand(batch_size, channel_in).astype(np.float16)
    np_weight = np.linspace(-0.64, 0.64, channel_in * channel_out).astype(np.float16).reshape((channel_in, channel_out))
    scale = np.random.rand(channel_in // group_size, channel_out).astype(np.float16)
    offset = np.random.rand(channel_in // group_size, channel_out).astype(np.float16)
    np_quant_int8_data, np_quant_int4_data = np_quant_int4_pergroup(np_weight, scale, offset, group_size)
    return np_x, np_quant_int8_data, np_quant_int4_data, scale, offset


def np_quant_int4_pergroup_data_gen(channel_in, channel_out, group_num):
    """quant data to int4 data"""
    np_quant_int8_data = np.random.randint(-8, 7, size=(channel_in, channel_out)).astype(np.int8)
    np_quant_int4_data = np_int4data_pack_to_int8(np_quant_int8_data)
    antiquant_scale = np.random.rand(group_num, channel_out).astype(np.float16)
    antiquant_offset = np.random.rand(group_num, channel_out).astype(np.float16)
    np_quant_fp16_data = np_antiquant_pergroup(np_quant_int8_data, antiquant_scale, antiquant_offset, group_num)
    return np_quant_fp16_data, np_quant_int4_data, antiquant_scale, antiquant_offset


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['GE', 'KBK', 'pynative'])
def test_ms_int4_weight_quant_1p(mode):
    """
    feature: test int4 dtype parameter save and load
    Description: test antiquant using weight quant bmm cell
    Expectation: accuracy in tolerance
    """
    if mode == 'pynative':
        ms.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'GE':
        ms.context.set_context(mode=ms.GRAPH_MODE)
        ms.set_context(jit_level='O2')
    elif mode == 'KBK':
        ms.set_context(mode=ms.GRAPH_MODE)

    scale = 0.1
    offset = 4
    np_x, np_int8_data, np_int4_weight = np_gen_int4_data(scale, offset)

    np_anti_weight = np_antiquant(np_int8_data, scale, offset)
    expect = np.matmul(np_x, np_anti_weight)
    ms_int4_weight = Parameter(Tensor(np_int4_weight, dtype=mstype.qint4x2))

    antiquant_scale = Tensor([scale], dtype=mstype.float16)
    antiquant_offset = Tensor([-offset], dtype=mstype.float16)
    quant_scale = None
    quant_offset = None
    bias = None
    wqbm_net = WeightQuantBatchMatmulNet(ms_int4_weight)
    x = Tensor(np_x, dtype=mstype.float16)
    fact = wqbm_net(x, antiquant_scale, antiquant_offset,
                    quant_scale, quant_offset, bias).asnumpy()
    np.testing.assert_allclose(fact, expect, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['GE', 'KBK', 'pynative'])
def test_ms_int4_weight_quant_perchannel_1p(mode):
    """
    feature: test int4 dtype parameter save and load
    Description: test antiquant using weight quant bmm cell
    Expectation: accuracy in tolerance
    """
    if mode == 'pynative':
        ms.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'GE':
        ms.context.set_context(mode=ms.GRAPH_MODE)
        ms.set_context(jit_level='O2')
    elif mode == 'KBK':
        ms.set_context(mode=ms.GRAPH_MODE)

    ms.set_seed(666)
    np_x, np_int8_data, np_int4_weight, scale, offset = np_gen_int4_data_perchannel()

    np_anti_weight = np_antiquant(np_int8_data, scale, offset)
    expect = np.matmul(np_x, np_anti_weight)
    ms_int4_weight = Parameter(Tensor(np_int4_weight, dtype=mstype.qint4x2))

    antiquant_scale = Tensor(scale, dtype=mstype.float16)
    antiquant_offset = Tensor(offset * -1.0, dtype=mstype.float16)
    quant_scale = None
    quant_offset = None
    bias = None
    wqbm_net = WeightQuantBatchMatmulNet(ms_int4_weight)
    x = Tensor(np_x, dtype=mstype.float16)
    fact = wqbm_net(x, antiquant_scale, antiquant_offset,
                    quant_scale, quant_offset, bias).asnumpy()
    np.testing.assert_allclose(fact, expect, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['GE', 'KBK', 'pynative'])
def test_ms_int4_ckpt_save_and_load(mode):
    """
    feature: test weight quant int4 net save ckpt and load
    Description: test int4 ckpt save and load procedure
    Expectation: save and load successful with the same inference results
    """
    if mode == 'pynative':
        ms.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'GE':
        ms.context.set_context(mode=ms.GRAPH_MODE)
        ms.set_context(jit_level='O2')
    elif mode == 'KBK':
        ms.set_context(mode=ms.GRAPH_MODE)

    scale = 0.1
    np_x, _, np_int4_weight = np_gen_int4_data(scale)

    ms_int4_weight = Parameter(Tensor(np_int4_weight, dtype=mstype.qint4x2))

    antiquant_scale = Tensor([scale], dtype=mstype.float16)
    antiquant_offset = Tensor([0], dtype=mstype.float16)
    quant_scale = None
    quant_offset = None
    bias = None
    wqbm_net = WeightQuantBatchMatmulNet(ms_int4_weight)
    x = Tensor(np_x, dtype=mstype.float16)
    expect = wqbm_net(x, antiquant_scale, antiquant_offset,
                      quant_scale, quant_offset, bias).asnumpy()

    # save ms ckpt
    ckpt_file_name = "int4.ckpt"
    ms.save_checkpoint(wqbm_net, ckpt_file_name)

    # reload ckpt and inference
    np_weight_tensor = np.ones(shape=(8, 4))
    weight_tensor = Parameter(Tensor(np_weight_tensor, dtype=mstype.qint4x2))
    new_net = WeightQuantBatchMatmulNet(weight_tensor)
    new_net.set_jit_config(JitConfig(jit_level='O0'))
    ms.load_checkpoint(ckpt_file_name, new_net)
    fact = new_net(x, antiquant_scale, antiquant_offset, quant_scale, quant_offset, bias).asnumpy()
    np.testing.assert_equal(expect, fact)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['GE', 'KBK'])
def test_ms_int4_mindir_save_and_load(mode):
    """
    feature: test weight quant int4 net save as mindir and load
    Description: test int4 mindir save and load procedure
    Expectation: save and load successful with the same inference results
    """
    if mode == 'pynative':
        ms.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'GE':
        ms.context.set_context(mode=ms.GRAPH_MODE)
        ms.set_context(jit_level='O2')
    elif mode == 'KBK':
        ms.set_context(mode=ms.GRAPH_MODE)

    scale = 0.1
    offset = 4.0
    ms.set_seed(666)
    np_x, _, np_int4_weight = np_gen_int4_data(scale, offset)

    ms_int4_weight = Parameter(Tensor(np_int4_weight, dtype=mstype.qint4x2))

    antiquant_scale = Tensor([scale], dtype=mstype.float16)
    antiquant_offset = Tensor([offset * -1.0], dtype=mstype.float16)
    quant_scale = None
    quant_offset = None
    bias = None
    wqbm_net = WeightQuantBatchMatmulNet(ms_int4_weight)
    x = Tensor(np_x, dtype=mstype.float16)
    expect = wqbm_net(x, antiquant_scale, antiquant_offset,
                      quant_scale, quant_offset, bias).asnumpy()

    # save ms ckpt
    ckpt_file_name = "int4.mindir"

    export(wqbm_net, x, antiquant_scale, antiquant_offset,
           quant_scale, quant_offset, bias, file_name=ckpt_file_name, file_format="MINDIR")

    graph = ms.load(ckpt_file_name)
    net = GraphCell(graph)
    fact = net(x, antiquant_scale, antiquant_offset).asnumpy()
    np.testing.assert_equal(expect, fact)


@pytest.mark.parametrize('mode', ['pynative', 'GE', 'KBK'])
def test_ms_int4_weight_quant_pergroup_1p_GE(mode):
    """
    feature: test int4 dtype parameter save and load
    Description: test antiquant using weight quant bmm cell
    Expectation: accuracy in tolerance
    """
    ms.set_context(device_target='Ascend', mode=ms.GRAPH_MODE)
    ms.set_seed(666)
    group_num = 4
    batch_seq = 4
    channel_in = 256
    channel_out = 128
    group_size = channel_in // group_num

    np_x, np_int8_data, np_int4_weight, scale, offset = np_gen_int4_data_pergroup(batch_seq,
                                                                                  channel_in,
                                                                                  channel_out,
                                                                                  group_size)
    np_anti_weight = np_antiquant_pergroup(np_int8_data, scale, offset, group_num)
    expect = np.matmul(np_x, np_anti_weight)
    ms_int4_weight = Parameter(Tensor(np_int4_weight, dtype=mstype.qint4x2))
    quant_scale = None
    quant_offset = None
    antiquant_scale = Tensor(scale)
    antiquant_offset = Tensor(offset)
    bias = None
    wqbm_net = WeightQuantBatchMatmulNet(ms_int4_weight, antiquant_group_size=group_size)
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'GE':
        ms.context.set_context(mode=ms.GRAPH_MODE)
    elif mode == 'KBK':
        ms.context.set_context(mode=ms.GRAPH_MODE)
        wqbm_net.set_jit_config(JitConfig(jit_level='O0'))
    x = Tensor(np_x, dtype=mstype.float16)
    fact = wqbm_net(x, antiquant_scale, antiquant_offset,
                    quant_scale, quant_offset, bias).asnumpy()
    np.testing.assert_allclose(fact, expect, rtol=3e-3)
