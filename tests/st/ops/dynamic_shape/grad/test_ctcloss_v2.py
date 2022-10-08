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

import numpy as np
import pytest
from scipy.special import log_softmax
from mindspore import nn, context, Tensor
from mindspore.ops.operations import CTCLossV2
from .test_grad_of_dynamic import TestDynamicGrad


class Net(nn.Cell):
    def __init__(self, blank, reduction):
        super(Net, self).__init__()
        self.loss = CTCLossV2(blank=blank, reduction=reduction)

    def construct(self, input_matrix, target, input_lengths, target_lengths):
        x, _ = self.loss(input_matrix, target, input_lengths, target_lengths)
        return x


def grad_dyn_case(is_dynamic_rank):
    test_dynamic = TestDynamicGrad(Net(blank=0, reduction='none'))
    input_sequences = 40
    input_sequences_min = 35
    classes = 20
    target_sequences = 20  # Target sequence length of longest target in batch (padding length)
    target_sequences_min = 10  # Minimum target length, for demonstration purposes
    batch = 1
    data_type = np.float32

    # Initialize random batch of input vectors, for *size = (input_sequences,N,classes)
    input_matrix = log_softmax(np.random.randn(input_sequences, batch, classes), 2).astype(data_type)
    # Initialize random batch of targets (0 = blank, 1:classes = classes)
    target = np.random.randint(low=1, high=classes, size=(batch, target_sequences), dtype=np.int64)

    input_lengths = np.random.randint(low=input_sequences_min, high=input_sequences, size=(batch,), dtype=np.int64)
    target_lengths = np.random.randint(low=target_sequences_min, high=target_sequences, size=(batch,), dtype=np.int64)

    input_matrix = Tensor(input_matrix)
    target = Tensor(target)
    input_lengths = Tensor(input_lengths)
    target_lengths = Tensor(target_lengths)

    test_dynamic.test_dynamic_grad_net([input_matrix, target, input_lengths, target_lengths], is_dynamic_rank)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_grad_dynamic_shape():
    """
    Feature: test CTCLossV2 grad dynamic shape on GPU and CPU.
    Description: input is dynamic shape.
    Expectation: the result match with static shape
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    grad_dyn_case(False)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_grad_dynamic_rank():
    """
    Feature: test CTCLossV2 grad dynamic rank on GPU and CPU.
    Description: input is dynamic rank.
    Expectation: the result match with static shape
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    grad_dyn_case(True)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gpu_grad_dynamic_shape():
    """
    Feature: test CTCLossV2 grad dynamic shape on GPU.
    Description: input is dynamic shape.
    Expectation: the result match with static shape
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    grad_dyn_case(False)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gpu_grad_dynamic_rank():
    """
    Feature: test CTCLossV2 grad dynamic shape on GPU.
    Description: input is dynamic shape.
    Expectation: the result match with static shape
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    grad_dyn_case(True)
