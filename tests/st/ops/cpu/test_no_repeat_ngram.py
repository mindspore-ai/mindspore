# Copyright 2020 Huawei Technologies Co., Ltd
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
import mindspore.nn as nn
import mindspore.context as context
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops.functional import vmap
from mindspore.common.api import jit
from mindspore import dtype as mstype

FLT_MAX = 3.4028235e+38


class CpuNet(nn.Cell):
    def __init__(self, ngram_size):
        super(CpuNet, self).__init__()
        self.no_repeat_ngram = P.NoRepeatNGram(ngram_size)

    def construct(self, state_seq, log_probs):
        return self.no_repeat_ngram(state_seq, log_probs)


base_state_seq = np.array([[[1, 2, 1, 2, 5, 1, 2],
                            [9, 3, 9, 5, 4, 1, 5],
                            [4, 7, 9, 1, 9, 6, 1],
                            [7, 6, 4, 2, 9, 1, 5],
                            [7, 5, 8, 9, 9, 3, 9]],
                           [[7, 7, 2, 7, 9, 9, 4],
                            [3, 4, 7, 4, 7, 6, 8],
                            [1, 9, 5, 7, 6, 9, 3],
                            [4, 8, 6, 4, 5, 6, 4],
                            [4, 8, 8, 4, 3, 4, 8]]], dtype=np.int32)
base_log_probs = np.random.random((2, 5, 10)).astype(np.float32)
base_expect_log_probs = base_log_probs.copy()
base_expect_log_probs[0, 0, 1] = -FLT_MAX
base_expect_log_probs[0, 0, 5] = -FLT_MAX
base_expect_log_probs[1, 3, 5] = -FLT_MAX
base_expect_log_probs[1, 4, 8] = -FLT_MAX


def test_net():
    """
    Feature: test NoRepeatNGram on CPU.
    Description: inputs with batch.
    Expectation: the result match with expect.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    state_seq = Tensor(base_state_seq)
    log_probs = Tensor(base_log_probs)
    expect_log_probs = base_expect_log_probs

    net = CpuNet(ngram_size=3)
    output = net(state_seq, log_probs)
    assert np.array_equal(expect_log_probs, output.asnumpy())


def test_net_dynamic_shape():
    """
    Feature: test NoRepeatNGram dynamic shape on CPU.
    Description: inputs with batch.
    Expectation: the result match with expect.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    state_seq = Tensor(base_state_seq)
    log_probs = Tensor(base_log_probs)
    expect_log_probs = base_expect_log_probs

    net = CpuNet(ngram_size=3)
    place_holder_x = Tensor(shape=[None, 5, 7], dtype=mstype.int32)
    place_holder_v = Tensor(shape=[None, 5, 10], dtype=mstype.float32)
    net.set_inputs(place_holder_x, place_holder_v)
    output = net(state_seq, log_probs)
    assert np.array_equal(expect_log_probs, output.asnumpy())


def vmap_case():
    class Net(nn.Cell):
        def __init__(self, ngram_size):
            super(Net, self).__init__()
            self.no_repeat_ngram = P.NoRepeatNGram(ngram_size)

        def construct(self, seq, log):
            return self.no_repeat_ngram(seq, log)

    class VmapNet(nn.Cell):
        def __init__(self, net, in_axes, out_axes):
            super(VmapNet, self).__init__()
            self.net = net
            self.in_axes = in_axes
            self.out_axes = out_axes

        def construct(self, state_seq, log_probs):
            return vmap(self.net, self.in_axes, self.out_axes)(state_seq, log_probs)

    @jit
    def for_net(state_seq, log_probs, ngram_size):
        # split and concat along dimension 0
        output = []
        for i in range(state_seq.shape[0]):
            out = P.NoRepeatNGram(ngram_size)(state_seq[i], log_probs[i])
            output.append(out)
        return F.stack(output)

    state_seq = Tensor(np.tile(base_state_seq, (2, 1, 1, 1)))
    log_probs = Tensor(np.tile(base_log_probs, (2, 1, 1, 1)))

    output = VmapNet(Net(3), (0, 0), 0)(state_seq, log_probs)
    fornet_output = for_net(state_seq, log_probs, 3)
    np.testing.assert_allclose(output.asnumpy(), fornet_output.asnumpy(), rtol=1e-6)


def vmap_nested_case():
    class Net(nn.Cell):
        def __init__(self, ngram_size):
            super(Net, self).__init__()
            self.no_repeat_ngram = P.NoRepeatNGram(ngram_size)

        def construct(self, seq, log):
            return self.no_repeat_ngram(seq, log)

    class WrapNet(nn.Cell):
        def __init__(self, net, inin_axes, inout_axes, outin_axes, outout_axes):
            super(WrapNet, self).__init__()
            self.net = net
            self.ii = inin_axes
            self.io = inout_axes
            self.oi = outin_axes
            self.oo = outout_axes

        def construct(self, state_seq, log_probs):
            return vmap(vmap(self.net, self.ii, self.io), self.oi, self.oo)(state_seq, log_probs)

    @jit
    def for_net(state_seq, log_probs, ngram_size):
        # split and concat along dimension 0 and 1
        output = []
        for i in range(state_seq.shape[0]):
            inner_output = []
            for j in range(state_seq.shape[1]):
                out = P.NoRepeatNGram(ngram_size)(state_seq[i][j], log_probs[i][j])
                inner_output.append(out)
            output.append(F.stack(inner_output))
        return F.stack(output)

    state_seq = Tensor(np.tile(base_state_seq, (2, 2, 1, 1, 1)))
    log_probs = Tensor(np.tile(base_log_probs, (2, 2, 1, 1, 1)))

    output = WrapNet(Net(3), (0, 0), 0, (0, 0), 0)(state_seq, log_probs)
    fornet_output = for_net(state_seq, log_probs, 3)
    np.testing.assert_allclose(output.asnumpy(), fornet_output.asnumpy(), rtol=1e-6)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_vmap_cpu():
    """
    Feature: test NoRepeatNGram vmap on CPU.
    Description: inputs with batch.
    Expectation: the result match with expect.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    vmap_case()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_vmap_cpu_nested():
    """
    Feature: test nested NoRepeatNGram vmap on CPU.
    Description: inputs with batch.
    Expectation: the result match with expect.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    vmap_nested_case()
