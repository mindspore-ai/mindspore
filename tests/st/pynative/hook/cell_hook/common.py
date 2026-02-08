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

import numpy as np
import mindspore as ms
from mindspore import Tensor
from mindspore.common import ParameterTuple
from mindspore.graph.api import jit
from mindspore.common._stub_tensor import StubTensor
from tests.st.pynative.utils import GradOfAllInputs

def assert_equal(out1, out2):
    if isinstance(out1, (Tensor, StubTensor)):
        assert isinstance(out2, (Tensor, StubTensor))
    else:
        assert type(out1) is type(out2)

    if isinstance(out1, (list, tuple)):
        assert len(out1) == len(out2)
        for e1, e2 in zip(out1, out2):
            assert_equal(e1, e2)
    elif isinstance(out1, dict):
        assert len(out1) == len(out2)
        for k, v in out1.items():
            assert k in out2
            assert_equal(v, out2[k])
    if isinstance(out1, Tensor):
        assert np.allclose(out1.asnumpy(), out2.asnumpy(), 0.000001, 0.000001)

def assert_jit_net(net, py_out, *args):
    assert net.construct.__code__.co_name == "construct"
    old_construct = net.construct
    net.construct = jit(net.construct)
    assert net.construct.__code__.co_name == "staging_specialize"
    jit_out = net(*args)
    assert_equal(jit_out, py_out)
    net.construct = old_construct
    assert net.construct.__code__.co_name == "construct"

def assert_jit_grad_net_by_grad_op(grad_op, net, py_out, is_grad_param, *args, **kwargs):
    assert net.construct.__code__.co_name == "construct"
    old_construct = net.construct
    net.construct = jit(net.construct)
    assert net.construct.__code__.co_name == "staging_specialize"
    if is_grad_param:
        grad_net = grad_op(net, ParameterTuple(net.trainable_params()))
    else:
        grad_net = grad_op(net)
    jit_out = grad_net(*args, **kwargs)
    assert_equal(jit_out, py_out)
    net.construct = old_construct
    assert net.construct.__code__.co_name == "construct"

def assert_jit_grad_net_by_ms_grad(net, grad_position, py_out, *args, **kwargs):
    assert net.construct.__code__.co_name == "construct"
    old_construct = net.construct
    net.construct = jit(net.construct)
    grad_net = ms.grad(net, grad_position=grad_position)
    assert net.construct.__code__.co_name == "staging_specialize"
    jit_out = grad_net(*args, *kwargs)
    assert_equal(jit_out, py_out)
    net.construct = old_construct
    assert net.construct.__code__.co_name == "construct"

def assert_jit_grad_net_by_grad_of_all_inputs(net, py_out, *args, **kwargs):
    assert net.construct.__code__.co_name == "construct"
    old_construct = net.construct
    net.construct = jit(net.construct)
    assert net.construct.__code__.co_name == "staging_specialize"
    grad_net = GradOfAllInputs(net)
    jit_out = grad_net(*args, **kwargs)
    assert_equal(jit_out, py_out)
    net.construct = old_construct
    assert net.construct.__code__.co_name == "construct"
