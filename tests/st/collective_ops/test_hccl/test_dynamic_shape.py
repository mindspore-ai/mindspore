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
"""
The tests of mindspore, used to test communication ops of AllGatherV.
"""
import numpy as np
from mindspore import Tensor, context
from mindspore.nn import Cell
from mindspore.common import dtype, ParameterTuple
from mindspore.communication.management import init
from mindspore.communication.management import get_rank
from mindspore.communication.management import GlobalComm
from mindspore.ops.composite import GradOperation
from mindspore.ops.operations import AlltoAll, GeLU
from tests.st.pi_jit.share.utils import allclose_nparray
np.random.seed(1)
context.set_context(jit_level='O0')

class _Grad(Cell):
    """
    class _Grad for base AllGatherV
    """
    def __init__(self, grad, network, wrt_params=False, real_inputs_count=None):
        """
        define init for base AllGatherV
        """
        super().__init__()
        self.network = network
        self.grad = grad
        self.sens_param = self.grad.sens_param
        self.wrt_params = wrt_params
        self.real_inputs_count = real_inputs_count
        if self.wrt_params:
            self.params = ParameterTuple(self.network.trainable_params())

    def __call__(self, *inputs):
        """
        define call for base AllGatherV
        """
        if self.sens_param and self._dynamic_shape_inputs is not None:
            # not support dynamic shape sens
            if self.real_inputs_count is None:
                dyn_inputs = self._dynamic_shape_inputs[:-1]
                real_sens = inputs[-1:]
            else:
                idx = self.real_inputs_count
                dyn_inputs = self._dynamic_shape_inputs[:idx]
                real_sens = inputs[idx:]
            static_sens = list(dyn_inputs) + list(real_sens)
            super().set_inputs(*static_sens)

        out = super().__call__(*inputs)
        return out

    def construct(self, *inputs):
        """
        define construct for base AllGatherV
        """
        if self.wrt_params:
            if self.real_inputs_count is None or self.sens_param is False:
                return self.grad(self.network, self.params)(*inputs)
            real_inputs = inputs[:self.real_inputs_count]
            sense_param_inputs = inputs[self.real_inputs_count:]
            return self.grad(self.network, self.params)(*real_inputs, sense_param_inputs)
        if self.real_inputs_count is None or self.sens_param is False:
            return self.grad(self.network)(*inputs)
        real_inputs = inputs[:self.real_inputs_count]
        sense_param_inputs = inputs[self.real_inputs_count:]
        return self.grad(self.network)(*real_inputs, sense_param_inputs)


class GradOfFirstInput(_Grad):
    """
    get grad of first input
    """
    def __init__(self, network, sens_param=True, real_inputs_count=None):
        super().__init__(grad=GradOperation(sens_param=sens_param),
                         network=network, real_inputs_count=real_inputs_count)


class MetaFactory:
    """
    define class MetaFactory
    """
    def __init__(self):
        """
        define init for MetaFactory
        """
        self.device_target = context.get_context('device_target')
        self.rank_size = None
        self.device_id = None
        self.global_rank_id = None


class OpsFactory(MetaFactory):
    """
    define class OpsFactory
    """
    def __init__(self, dtype=np.float16): # pylint: disable=redefined-outer-name
        """
        define init for OpsFactory
        """
        super().__init__()
        self.dtype = dtype
        if self.dtype == np.float16:
            self.loss = 1e-3
        elif self.dtype == np.float32:
            self.loss = 1e-4
        elif self.dtype == np.float64:
            self.loss = 1e-5
        else:
            self.loss = 0


class AllToAllGeluNet(Cell):
    """
    define class AllToAllGeluNet
    """
    def __init__(self, split_count, split_dim, concat_dim, group=GlobalComm.WORLD_COMM_GROUP):
        """
        define init for base AllGatherV
        """
        super(AllToAllGeluNet, self).__init__() # pylint: disable=super-with-arguments
        self.gelu = GeLU()
        self.alltoallv = AlltoAll(split_count=split_count, split_dim=split_dim,
                                  concat_dim=concat_dim, group=group)

    def construct(self, x):
        """
        define construct for base AllToAllGeluNet
        """
        x = self.gelu(x)
        x = self.alltoallv(x)
        return x


class AllToAllFactory(OpsFactory):
    """
    define class AllToAllFactory
    """
    def __init__(self, input_shape=(4, 3, 224, 224), input_dtype=np.float32,
                 group=GlobalComm.WORLD_COMM_GROUP, input_random_seed=5):
        """
        define init for base AllToAllFactory
        """
        super().__init__(dtype=input_dtype)
        self.group_rank_id = get_rank(group=group)
        np.random.seed(input_random_seed)
        self.input_np = np.random.randn(*input_shape).astype(dtype=input_dtype)


def test_alltoall_8p_dynamic_shape():
    """
    Feature: test 'AllToAll' communication operation.
    Description: test dynamic shape communication operation.
    Expectation: expect correct result.
    """
    init()
    fact = AllToAllFactory()
    net_dy = AllToAllGeluNet(split_count=8, split_dim=2,
                             concat_dim=3, group=GlobalComm.WORLD_COMM_GROUP)
    input_x_dyn = Tensor(shape=[None, 3, None, 224], dtype=dtype.float32)
    net_dy.set_inputs(input_x_dyn)
    forward_dy = net_dy(Tensor(fact.input_np))
    net_dy_grad = GradOfFirstInput(net_dy)
    net_dy_grad.set_train()
    grad_dy = net_dy_grad(Tensor(fact.input_np), forward_dy)

    net = AllToAllGeluNet(split_count=8, split_dim=2,
                          concat_dim=3, group=GlobalComm.WORLD_COMM_GROUP)
    forward = net(Tensor(fact.input_np))
    net_grad = GradOfFirstInput(net)
    net_grad.set_train()
    grad = net_grad(Tensor(fact.input_np), forward)

    allclose_nparray(forward_dy.asnumpy(), forward.asnumpy(), 0.001, 0.001)
    allclose_nparray(grad_dy.asnumpy(), grad.asnumpy(), 0.0001, 0.0001)
