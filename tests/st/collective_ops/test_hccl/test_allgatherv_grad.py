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
from mindspore.common import ParameterTuple
from mindspore.communication.management import init
from mindspore.communication.management import get_rank
from mindspore.communication.management import GlobalComm
from mindspore.train.model import Model
from mindspore.ops.composite import GradOperation
from mindspore.ops.operations import AllGatherV
from tests.st.pi_jit.share.utils import tensor_to_numpy, allclose_nparray_recursive
context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")


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
    def __init__(self, dtype=np.float16):
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


class AllGatherFactory(OpsFactory):
    """
    define class AllGatherFactory
    """
    def __init__(self, input_shape_list=None, input_dtype=np.float32, output_split_sizes=None,
                 group=GlobalComm.WORLD_COMM_GROUP, input_random_seed=5, grad_lab="one"):
        """
        define init for AllGatherFactory
        """
        super().__init__(dtype=input_dtype)
        self.group_rank_id = get_rank(group=group)
        self.output_split_sizes = output_split_sizes
        np.random.seed(input_random_seed)
        self.input_np_list = [np.random.randn(*s).astype(dtype=input_dtype) for s in input_shape_list]
        input_np_list = []
        for index, input_np in enumerate(self.input_np_list):
            input_np_list.append(input_np[0][:self.output_split_sizes[index]])
        self.output_grad_np = np.concatenate(input_np_list, axis=0)
        if grad_lab == "two":
            self.output_grad_tensor = (Tensor(self.output_grad_np), Tensor(self.output_grad_np))
        else:
            self.output_grad_tensor = Tensor(self.output_grad_np)
        self.input_tensor_list = None

    def forward_mindspore_impl(self):
        """
        define forward impl  OneAllGatherV
        """
        self.input_tensor_list = [Tensor(input_np) for input_np in self.input_np_list]
        model = Model(self)
        output = model.predict(self.input_tensor_list[self.group_rank_id], Tensor(self.output_split_sizes))
        return tensor_to_numpy(output)

    def forward_cmp(self):
        """
        define forward cmp OneAllGatherV
        """
        output_mindspore = self.forward_mindspore_impl()
        output_numpy = self.forward_numpy_impl()
        allclose_nparray_recursive(
            output_numpy,
            output_mindspore,
            self.loss,
            self.loss)

    def grad_mindspore_impl(self):
        """
        define grad impl  OneAllGatherV
        """
        self.input_tensor_list = [Tensor(input_np)
                                  for input_np in self.input_np_list]
        net_me = GradOfFirstInput(self)
        net_me.set_train()
        output = net_me(self.input_tensor_list[self.group_rank_id],
                        Tensor(self.output_split_sizes), self.output_grad_tensor)
        return tensor_to_numpy(output)

    def grad_cmp(self):
        """
        define grad cmp OneAllGatherV
        """
        input_grad_mindspore = self.grad_mindspore_impl()
        input_grad_numpy = self.grad_numpy_impl()
        allclose_nparray_recursive(input_grad_numpy[self.group_rank_id], input_grad_mindspore,
                                   self.loss, self.loss)


class OneAllGatherV(Cell, AllGatherFactory):
    """
    define class OneAllGatherV
    """
    def __init__(self, input_shape_list, output_split_sizes, input_dtype=np.float32,
                 group=GlobalComm.WORLD_COMM_GROUP, input_random_seed=5):
        """
        define init for OneAllGatherV
        """
        AllGatherFactory.__init__(self, input_shape_list=input_shape_list, input_dtype=input_dtype,
                                  output_split_sizes=output_split_sizes, group=group,
                                  input_random_seed=input_random_seed)
        super().__init__()
        self.all_gather_v = AllGatherV(group=group)

    def construct(self, x, output_split_sizes):
        x = self.all_gather_v(x, output_split_sizes)
        return x

    def forward_numpy_impl(self):
        """
        forward numpy impl of OneAllGatherV
        """
        input_np_list = []
        for index, input_np in enumerate(self.input_np_list):
            input_np_list.append(input_np[0][:self.output_split_sizes[index]])
        return np.concatenate(input_np_list, axis=0)

    def grad_numpy_impl(self):
        """
        grad numpy impl of OneAllGatherV
        """
        reduce_grad_np = 0
        for _ in self.input_np_list:
            reduce_grad_np += self.output_grad_np
        rank = []
        for index, i in enumerate(self.output_split_sizes):
            if index == 0:
                rank.append(np.array(reduce_grad_np[:i]))
            else:
                rank.append(np.array(
                    reduce_grad_np[self.output_split_sizes[index - 1]:self.output_split_sizes[index - 1] + i]))
        return rank


#msrun --worker_num=2 --local_worker_num=2 --join=True pytest -sv  test_allgatherv_dts.py::test_allgatherv_001
def test_allgatherv_001():
    """
    Feature: test 'AllGatherV' communication operation.
    Description: https://e.gitee.com/mind_spore/dashboard?issue=IC9HKA.
    Expectation: expect correct result.
    """
    init()
    input_shape_list = [(1, 3), (1, 5)]
    output_split_sizes = [3, 5]
    fact = OneAllGatherV(input_shape_list, output_split_sizes)
    fact.forward_cmp()
    fact.grad_cmp()
