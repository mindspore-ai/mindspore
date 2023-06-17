# Copyright 2023 Huawei Technologies Co., Ltd
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
import os.path

import numpy as np
import pytest

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import ops
from mindspore.common import set_seed

from test_combine_momentum import clear_files, find_files

set_seed(1)


class BnSiluFusedNet(nn.Cell):
    def __init__(self, c=24, momentum=0.97, eps=1e-3):
        super(BnSiluFusedNet, self).__init__()
        self.bn = nn.BatchNorm2d(c, momentum=momentum, eps=eps, use_batch_statistics=True, data_format="NHWC")
        self.act = nn.SiLU()

    def construct(self, x):
        x = self.bn(x)
        x = self.act(x)
        return x


class GradNet(nn.Cell):
    def __init__(self, net):
        super(GradNet, self).__init__()
        self.net = net
        self.grad_op = ops.GradOperation(get_all=True)

    def construct(self, x):
        gradient_function = self.grad_op(self.net)
        return gradient_function(x)


SAVE_GRAPHS_PATH = "./ir_path"


def ir_fusion_bn_silu(mode, check_ir):
    ms.set_context(mode=mode, device_target="GPU", save_graphs=True, save_graphs_path=SAVE_GRAPHS_PATH)
    clear_files(SAVE_GRAPHS_PATH + "/verbose_ir_files/*")
    net = BnSiluFusedNet()
    input_x = Tensor(np.arange(1, 24 * 24 * 24 + 1, 1).reshape(1, 24, 24, 24).astype(np.float32))
    out = net(input_x)
    if check_ir:
        result = find_files('BatchNormWithActivation', os.path.join(SAVE_GRAPHS_PATH + "/verbose_ir_files",
                                                                    'hwopt*silu_fusion*ir'))
        assert result == '2'
    return out


def ir_fusion_bn_silu_grad(mode, check_ir):
    ms.set_context(mode=mode, device_target="GPU", save_graphs=True, save_graphs_path=SAVE_GRAPHS_PATH)
    clear_files(SAVE_GRAPHS_PATH + "/verbose_ir_files/*")
    net = GradNet(BnSiluFusedNet())
    input_x = Tensor(np.arange(1, 24 * 24 * 24 + 1, 1).reshape(1, 24, 24, 24).astype(np.float32))
    out = net(input_x)
    if check_ir:
        result = find_files('BatchNormGradWithActivation', os.path.join(SAVE_GRAPHS_PATH + "/verbose_ir_files",
                                                                        'hwopt*silu_grad_fusion*ir'))
        assert result == '2'
    return out


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_ir_fusion_bn_silu_wrap():
    """"
    Feature: Test batch norm and silu fusion
    Description: Test batch norm and silu fusion
    Expectation: The results are as expected
    """
    out1 = ir_fusion_bn_silu(ms.GRAPH_MODE, True)
    out2 = ir_fusion_bn_silu(ms.PYNATIVE_MODE, False)
    assert np.all(out1.asnumpy() == out2.asnumpy())


@pytest.mark.level2
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_ir_fusion_bn_silu_grad_wrap():
    """"
    Feature: Test batch norm and silu grad fusion
    Description: Test batch norm and silu grad fusion
    Expectation: The results are as expected
    """
    out1 = ir_fusion_bn_silu_grad(ms.GRAPH_MODE, True)
    out2 = ir_fusion_bn_silu_grad(ms.PYNATIVE_MODE, False)
    if len(out1) != len(out2):
        assert False
    for i, _ in enumerate(out1):
        assert np.all(out1[i].asnumpy() == out2[i].asnumpy())
