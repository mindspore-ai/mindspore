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
"""adasum"""
from __future__ import absolute_import

import copy
import hashlib
import math

import mindspore.nn as nn
import mindspore.log as logger
from mindspore import context
from mindspore._checkparam import Validator as validator
from mindspore.nn.cell import Cell
from mindspore.common.parameter import ParameterTuple, Parameter
from mindspore.parallel._utils import _get_global_rank, _get_stage_device_num
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.ops.operations._inner_ops import Send, Receive
from mindspore.common.tensor import Tensor
from mindspore.common import dtype as mstype
from mindspore.communication.management import create_group

__all__ = ["AdaSumByDeltaWeightWrapCell", "AdaSumByGradWrapCell"]

MAX_NUM_HASH = 2 ** 31

_update_parameters = C.MultitypeFuncGraph("update_parameters")
_reshape_grads = C.MultitypeFuncGraph("reshape_grads")


@_update_parameters.register("Tensor", "Tensor", "Tensor", "Tensor", "Function")
def _update_parameters_adasum(delta_weight, update_delta_weight, parameter, old_parameter, reshape):
    shape = F.shape(delta_weight)
    update_delta_weight = reshape(update_delta_weight, shape)
    new_parameter = old_parameter - update_delta_weight
    P.Assign()(parameter, new_parameter)
    return parameter


@_reshape_grads.register("Tensor", "Tensor", "Function")
def reshape_grads_adasum(grads, update_grads, reshape):
    """
    Reshape gradient.
    """
    shape = F.shape(grads)
    update_grads = reshape(update_grads, shape)
    return update_grads


def _send_before_receive(send_part, send, recv):
    send_ok = send(send_part)
    return recv(send_ok)


def _receive_before_send(send_part, send, recv):
    receive_ok = recv(send_part)
    send_part = F.depend(send_part, receive_ok)
    return F.depend(receive_ok, send(send_part))


def _send_recv_res(left_send, recv_part, local_part, allreduce, parameter_divisibility, allreduce_node_num):
    """send result and receive result."""
    if parameter_divisibility:
        recv_part = P.Squeeze()(recv_part)
        if F.shape(recv_part) is None:
            recv_part = Tensor([recv_part])
        local_part = F.depend(local_part, recv_part)
        eps = 1e-12
        scale_value = P.ReduceMax()(local_part) + eps
        local_part_scale = local_part / scale_value
        recv_part_scale = recv_part / scale_value
        recv_part_scale = F.depend(recv_part_scale, local_part_scale)
        value_0 = P.ReduceSum()(local_part_scale * recv_part_scale) + eps
        if left_send:
            value_1 = P.ReduceSum()(local_part_scale * local_part_scale) + eps
            value_2 = P.ReduceSum()(recv_part_scale * recv_part_scale) + eps
        else:
            value_1 = P.ReduceSum()(recv_part_scale * recv_part_scale) + eps
            value_2 = P.ReduceSum()(local_part_scale * local_part_scale) + eps
        value_0 = allreduce(value_0)
        value_1 = F.depend(allreduce(value_1), value_0)
        value_2 = F.depend(allreduce(value_2), value_1)
        if left_send:
            res = (1 - (value_0 / (2 * value_1))) * local_part + (1 - (value_0 / (2 * value_2))) * recv_part
        else:
            res = (1 - (value_0 / (2 * value_1))) * recv_part + (1 - (value_0 / (2 * value_2))) * local_part
    else:
        res = allreduce(local_part)
        res = res / allreduce_node_num
    return res


_adasum_opt_forward = C.MultitypeFuncGraph("adasum_opt_forward")
_adasum_opt_rollback = C.MultitypeFuncGraph("adasum_opt_rollback")


@_adasum_opt_forward.register("Bool", "Function", "Bool", "Int64", "Function", "Function", "Tensor")
def _adasum_opt_forward_process(left_send, allreduce, parameter_divisibility, allreduce_node_num, send, recv, delta_w):
    """adasum optimizer process."""
    if parameter_divisibility:
        delta_w = P.Squeeze()(delta_w)
        ori_len = F.shape(delta_w)[0]
        divide_len = ori_len // 2
        left_part = delta_w[:divide_len]
        right_part = delta_w[divide_len:]
    else:
        left_part = delta_w
        right_part = delta_w

    if left_send:
        if parameter_divisibility:
            recv_part = _send_before_receive(left_part, send, recv)
        else:
            recv_part = right_part
        update_delta_w = _send_recv_res(left_send, recv_part, right_part, allreduce, parameter_divisibility,
                                        allreduce_node_num)
    else:
        if parameter_divisibility:
            recv_part = _receive_before_send(right_part, send, recv)
        else:
            recv_part = left_part
        update_delta_w = _send_recv_res(left_send, recv_part, left_part, allreduce, parameter_divisibility,
                                        allreduce_node_num)
    return update_delta_w


@_adasum_opt_rollback.register("Bool", "Bool", "Tensor", "Function", "Function")
def _adasum_opt_rollback_process(left_send, parameter_divisibility, delta_w, send, recv):
    """adasum optimizer rollback process."""
    if parameter_divisibility:
        if left_send:
            recv_part = _send_before_receive(delta_w, send, recv)
        else:
            recv_part = _receive_before_send(delta_w, send, recv)

        recv_part = P.Squeeze()(recv_part)
        if F.shape(recv_part) is None:
            recv_part = Tensor([recv_part])
        if F.shape(delta_w) is None:
            delta_w = Tensor([delta_w])
        recv_part = P.Reshape()(recv_part, (-1,))
        delta_w = P.Reshape()(delta_w, (-1,))

        if left_send:
            res = P.Concat()((recv_part, delta_w))
        else:
            res = P.Concat()((delta_w, recv_part))
    else:
        res = delta_w
    return res


class _AdaSum(Cell):
    r"""
    The Adaptive Summation, or AdaSum, is a novel algorithm for improving distributed data
    parallel training of Deep Learning models.

    Inputs:
        - **delta_weights** (Tuple(Tensor)) - Tuple of gradients.
        - **parameters** (Tuple(Parameter)) - Tuple of current parameters.
        - **old_parameters** (Tuple(Parameter)) - Tuple of last parameters.

    Outputs:
        - **adasum_parameters** (Tuple(Tensor)) - Tuple of parameters after adasum process.
    """
    def __init__(self, rank, device_number, group_number, parameter_tuple):
        super(_AdaSum, self).__init__()
        self.rank = rank
        self.device_number = device_number
        self.group_number = group_number
        self.parameter_tuple = parameter_tuple
        self.calc_times = int(math.log(self.group_number, 2))
        self.send_node = []
        self.send_list_forward = []
        self.recv_list_forward = []
        self.send_list_rollback = []
        self.recv_list_rollback = []
        self.allreduce_list = []
        self.parameter_divisibility_list = []
        self.allreduce_node_num_list = []
        self._generate_communication_op()
        self.hyper_map = C.HyperMap()
        self.update_reshape_list = []
        for parameter in self.parameter_tuple:
            reshape = P.Reshape().add_prim_attr("target_param", "adasum_delta_weight." + parameter.name)
            self.update_reshape_list.append(reshape)

    @staticmethod
    def _hash(step, target, weights_index):
        target = "tag" + str(step) + str(target) + str(weights_index)
        target_hash = hashlib.sha1(target.encode()).hexdigest()
        hash_res = int(int(target_hash, 16) % MAX_NUM_HASH)
        return hash_res

    def construct(self, delta_weights, parameters, old_parameters):
        forward_weights = [delta_weights]
        for i in range(self.calc_times):
            process_weights = self.hyper_map(F.partial(_adasum_opt_forward, self.send_node[i]), self.allreduce_list[i],
                                             self.parameter_divisibility_list[i], self.allreduce_node_num_list[i],
                                             self.send_list_forward[i], self.recv_list_forward[i], forward_weights[-1])
            forward_weights.append(process_weights)
        for i in range(self.calc_times):
            j = self.calc_times - i - 1
            process_weights = self.hyper_map(F.partial(_adasum_opt_rollback, self.send_node[j]),
                                             self.parameter_divisibility_list[j], forward_weights[j + 1],
                                             self.send_list_rollback[j], self.recv_list_rollback[j])
            forward_weights[j] = process_weights
        adasum_parameters = self.hyper_map(F.partial(_update_parameters), delta_weights, forward_weights[0],
                                           parameters, old_parameters, self.update_reshape_list)
        return adasum_parameters

    def _generate_communication_op(self):
        """generate communication op."""
        last_delta_weights = []
        fusion_attr = "origin_fusion"
        if context.get_auto_parallel_context("parallel_mode") in ["data_parallel", "hybrid_parallel"]:
            fusion_attr = "fusion"
        for step in range(self.calc_times):
            current_group = self.device_number * (2 ** step)
            if (self.rank // current_group) % 2 == 0:
                dest_target = self.rank + current_group
                self.send_node.append(True)
            else:
                dest_target = self.rank - current_group
                self.send_node.append(False)
            send_left = []
            send_right = []
            recv_left = []
            recv_right = []
            allreduce_node_num = ()
            left_delta_weights, right_delta_weights, delta_weights_divisibility = \
                self._get_delta_weights_info(last_delta_weights)
            self.parameter_divisibility_list.append(delta_weights_divisibility)
            weights_index = 0
            fusion_id = (step + 1) * 3
            for shape, dtype, name in left_delta_weights:
                send_tag = self._hash(step, self.rank, weights_index)
                send = Send(sr_tag=send_tag, dest_rank=dest_target, group="hccl_world_group")
                send.add_prim_attr(fusion_attr, fusion_id)
                send.add_prim_attr("opposite_rank", dest_target)
                send.add_prim_attr("target_param", name)
                recv_tag = self._hash(step, dest_target, weights_index)
                recv = Receive(sr_tag=recv_tag, src_rank=dest_target, shape=shape, dtype=dtype,
                               group="hccl_world_group")
                recv.add_prim_attr(fusion_attr, fusion_id)
                recv.add_prim_attr("opposite_rank", dest_target)
                recv.add_prim_attr("target_param", name)
                send_left.append(send)
                recv_left.append(recv)
                weights_index += 1
            for shape, dtype, name in right_delta_weights:
                send_tag = self._hash(step, self.rank, weights_index)
                send = Send(sr_tag=send_tag, dest_rank=dest_target, group="hccl_world_group")
                send.add_prim_attr(fusion_attr, fusion_id + 1)
                send.add_prim_attr("opposite_rank", dest_target)
                send.add_prim_attr("target_param", name)
                recv_tag = self._hash(step, dest_target, weights_index)
                recv = Receive(sr_tag=recv_tag, src_rank=dest_target, shape=shape, dtype=dtype,
                               group="hccl_world_group")
                recv.add_prim_attr(fusion_attr, fusion_id + 1)
                recv.add_prim_attr("opposite_rank", dest_target)
                recv.add_prim_attr("target_param", name)
                send_right.append(send)
                recv_right.append(recv)
                weights_index += 1
            if self.send_node and self.send_node[-1]:
                self.send_list_forward.append(send_left)
                self.send_list_rollback.append(send_right)
                self.recv_list_forward.append(recv_right)
                self.recv_list_rollback.append(recv_left)
                last_delta_weights = right_delta_weights
            else:
                self.send_list_forward.append(send_right)
                self.send_list_rollback.append(send_left)
                self.recv_list_forward.append(recv_left)
                self.recv_list_rollback.append(recv_right)
                last_delta_weights = left_delta_weights
            param_allreduce_list = []
            neighbor_ids = []
            rank_ids = []
            for index in range(2 ** (step + 1)):
                node_rank = self.rank // self.device_number
                double_d = 2 ** (step + 1)
                neighbor_id = (node_rank // double_d * double_d + index) * self.device_number + \
                              self.rank % self.device_number
                neighbor_ids.append(str(neighbor_id))
                rank_ids.append(neighbor_id)
            group_name = "-".join(neighbor_ids)
            if context.get_auto_parallel_context("parallel_mode") in ["data_parallel", "hybrid_parallel"]:
                create_group(group_name, rank_ids)
            for parameter in self.parameter_tuple:
                allreduce = P.AllReduce("sum", group_name)
                allreduce.add_prim_attr("target_param", "adasum_delta_weight." + parameter.name)
                allreduce.add_prim_attr(fusion_attr, fusion_id + 2)
                allreduce.add_prim_attr("step", step)
                param_allreduce_list.append(allreduce)
            self.allreduce_list.append(param_allreduce_list)
            for param_divisibility in delta_weights_divisibility:
                if param_divisibility:
                    allreduce_node_num += (0,)
                else:
                    allreduce_node_num += (2 ** (step + 1),)
            self.allreduce_node_num_list.append(allreduce_node_num)

    def _get_delta_weights_info(self, last_delta_weights):
        """get delta weights info."""
        half_delta_weights = []
        if last_delta_weights:
            half_delta_weights = last_delta_weights
        else:
            for parameter in self.parameter_tuple:
                new_shape = [int(x) for x in parameter.shape]
                half_delta_weights.append((new_shape, parameter.dtype, "adasum_delta_weight." + parameter.name))
        left_delta_weights = []
        right_delta_weights = []
        delta_weights_divisibility = ()
        for shape, dtype, name in half_delta_weights:
            left_shape = copy.deepcopy(shape)
            right_shape = copy.deepcopy(shape)
            divisibility_flag = False
            for i, value in enumerate(shape):
                if value > 1:
                    left_shape[i] = int(value // 2)
                    right_shape[i] = value - int(value // 2)
                    divisibility_flag = True
                    break
            left_delta_weights.append((left_shape, dtype, name))
            right_delta_weights.append((right_shape, dtype, name))
            delta_weights_divisibility += (divisibility_flag,)
        return left_delta_weights, right_delta_weights, delta_weights_divisibility


class _AdaSumByGrad(_AdaSum):
    """Apply adasum by gradients"""
    def construct(self, grads):
        forward_grads = [grads]
        for i in range(self.calc_times):
            process_weights = self.hyper_map(F.partial(_adasum_opt_forward, self.send_node[i]), self.allreduce_list[i],
                                             self.parameter_divisibility_list[i], self.allreduce_node_num_list[i],
                                             self.send_list_forward[i], self.recv_list_forward[i], forward_grads[-1])
            forward_grads.append(process_weights)
        for i in range(self.calc_times):
            j = self.calc_times - i - 1
            process_weights = self.hyper_map(F.partial(_adasum_opt_rollback, self.send_node[j]),
                                             self.parameter_divisibility_list[j], forward_grads[j + 1],
                                             self.send_list_rollback[j], self.recv_list_rollback[j])
            forward_grads[j] = process_weights
        update_grads = self.hyper_map(F.partial(_reshape_grads), grads, forward_grads[0],
                                      self.update_reshape_list)
        return update_grads


_get_delta_weight = C.MultitypeFuncGraph("_get_delta_weight")
_save_weight = C.MultitypeFuncGraph("_save_weight")
scale_mul = P.Mul().add_prim_attr("keep_alive", True)
_clone_weight = C.MultitypeFuncGraph("_clone_weight")


@_get_delta_weight.register("Tensor", "Tensor")
def _get_delta_weight_process(new_parameter, old_parameter):
    delta_w = old_parameter - new_parameter
    return delta_w


@_save_weight.register("Tensor", "Tensor")
def _save_weight_process(new_parameter, old_parameter):
    P.Assign()(new_parameter, old_parameter)
    return new_parameter


@_clone_weight.register("Tensor", "Tensor")
def _clone_weight_process(scale, weight):
    return scale_mul(weight, scale)


def _parallel_check():
    """Parallel infos checking"""
    if context.get_auto_parallel_context("parallel_mode") == "stand_alone":
        raise RuntimeError("Stand alone mode is not supported to apply adasum.")
    if context.get_auto_parallel_context("parallel_mode") in ["data_parallel", "hybrid_parallel"]:
        logger.warning("For data parallel mode or hybrid parallel mode, "
                       "it is recommended to using mindspore.boost to enable adasum.")
    if context.get_auto_parallel_context("enable_parallel_optimizer"):
        raise RuntimeError("Currently, the optimizer shard is not supported with applying adasum.")
    if context.get_auto_parallel_context("pipeline_stages") > 1:
        raise RuntimeError("Currently, the pipeline parallel is not supported with applying adasum.")
    stage_device_num = _get_stage_device_num()
    if stage_device_num < 16 or (stage_device_num & (stage_device_num - 1) != 0):
        raise RuntimeError("The device_num must be at least 16 and must be the power of 2 when applying adasum.")


class AdaSumByGradWrapCell(Cell):
    r"""
    Enable the adasum in "auto_parallel/semi_auto_parallel" mode.
    The implementation of the Adaptive Summation (AdaSum) algorithm is calculated by gradients.
    See the paper `AdaSum: Scaling Distributed Training with Adaptive Summation <https://arxiv.org/abs/2006.02924>`_.

    .. math::
        \begin{array}{ll}
          w_{t+1}=w_{t} - \alpha \cdot Adasum(g_{1}, g_{2})  \\
          w_{t+1}=w_{t} - \alpha \cdot [(1 - \frac{g_2^{T}\cdot g_1}{2\cdot \left \| g_1 \right \|^2 })\cdot g_1 + (1 -
          \frac{g_1^{T}\cdot g_2}{2\cdot \left \| g_2 \right \|^2 })\cdot g_2]  \\
        \end{array}

    In this implementation, :math:`g` represents the gradient of the weights,
    and the subscripts represent different devices in the data-parallel dimension.

    Note:
        When using AdaSum, the number of traning cards needs to be a power of 2 and at least 16 cards are required.
        Currently, the optimizer sharding and pipeline parallel is not supported when using AdaSum.
        It is recommended to using AdaSumByGradWrapCell in semi auto parallel/auto parallel mode. In data parallel
        mode, we recommend to using mindspore.boost to applying AdaSum.

    Args:
        optimizer (Union[Cell]): Optimizer for updating the weights. The construct function of the optimizer
            requires only one input.

    Inputs:
        - **grads** (Tuple(Tensor)) - Tuple of gradients, same with the input of passed optimizer.

    Raises:
        RuntimeError: If `parallel_mode` uses `stand_alone` mode, AdaSum only supports use in distributed scenarios.
        RuntimeError: If the optimizer parallel is used when using AdaSum.
        RuntimeError: If the pipeline parallel is used when using AdaSum.
        RuntimeError: If `device_num` is not a power of 2, or less than 16.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> from mindspore import nn
        >>> from mindspore.nn import AdaSumByGradWrapCell
        >>> net = Net()
        >>> optim = AdaSumByGradWrapCell(nn.Momentum(params=net.trainable_params(), learning_rate=0.1, momentum=0.9))
        >>> loss = nn.SoftmaxCrossEntropyWithLogits()
        >>> model = Model(net, loss_fn=loss, optimizer=optim, metrics=None)
    """
    def __init__(self, optimizer):
        super(AdaSumByGradWrapCell, self).__init__(auto_prefix=False)
        _device_number = 8
        _parallel_check()
        self.optimizer = optimizer
        validator.check_value_type('optimizer', optimizer, (nn.Optimizer,))
        self.parameters = optimizer.parameters
        self.hyper_map = C.HyperMap()
        group_number = _get_stage_device_num() // _device_number
        self.grad_clone = ParameterTuple(self.parameters)
        self.adasum = _AdaSumByGrad(_get_global_rank(), _device_number, group_number, self.grad_clone)
        self.sync_tensor = Parameter(Tensor(0, dtype=mstype.int32))

    def construct(self, grads):
        adasum_res = self.adasum(grads)
        sync_tensor = F.depend(self.sync_tensor, adasum_res)
        sync_flag = P.AllReduce()(sync_tensor)
        return F.depend(self.optimizer(adasum_res), sync_flag)


class AdaSumByDeltaWeightWrapCell(Cell):
    r"""
    Enable the adasum in "auto_parallel/semi_auto_parallel" mode.
    The implementation of the Adaptive Summation (AdaSum) algorithm is calculated based on the difference of weights
    before and after the updating of optimizer.
    See the paper `AdaSum: Scaling Distributed Training with Adaptive Summation <https://arxiv.org/abs/2006.02924>`_.

    .. math::
        \begin{array}{ll}
          w_{t+1}=w_{t} - \alpha \cdot Adasum(g_{1}, g_{2})  \\
          w_{t+1}=w_{t} - \alpha \cdot [(1 - \frac{g_2^{T}\cdot g_1}{2\cdot \left \| g_1 \right \|^2 })\cdot g_1 + (1 -
          \frac{g_1^{T}\cdot g_2}{2\cdot \left \| g_2 \right \|^2 })\cdot g_2]  \\
        \end{array}

    In this implementation, :math:`g` represents the weight difference before and after the updating of optimizer,
    and the subscripts represent different devices in the data parallel dimension.

    Note:
        When using AdaSum, the number of traning cards needs to be a power of 2 and at least 16 cards are required.
        Currently, the optimizer sharding and pipeline parallel is not supported when using AdaSum.
        It is recommended to using AdaSumByDeltaWeightWrapCell in semi auto parallel/auto parallel mode.
        In data parallel mode, we recommend to using mindspore.boost to applying AdaSum.

    Args:
        optimizer (Union[Cell]): Optimizer for updating the weights. The construct function of the optimizer
            requires only one input.

    Inputs:
        - **grads** (Tuple(Tensor)) - Tuple of gradients, same with the input of passed optimizer.

    Raises:
        RuntimeError: If `parallel_mode` uses `stand_alone` mode, AdaSum only supports use in distributed scenarios.
        RuntimeError: If the optimizer parallel is used when using AdaSum.
        RuntimeError: If the pipeline parallel is used when using AdaSum.
        RuntimeError: If `device_num` is not a power of 2, or less than 16.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> from mindspore import nn
        >>> from mindspore.nn import AdaSumByDeltaWeightWrapCell
        >>> net = Net()
        >>> optim = AdaSumByDeltaWeightWrapCell(nn.Momentum(params=net.trainable_params(),
        ...                                                 learning_rate=0.1, momentum=0.9))
        >>> loss = nn.SoftmaxCrossEntropyWithLogits()
        >>> model = Model(net, loss_fn=loss, optimizer=optim, metrics=None)
    """
    def __init__(self, optimizer):
        super(AdaSumByDeltaWeightWrapCell, self).__init__(auto_prefix=False)
        _parallel_check()
        self.optimizer = optimizer
        validator.check_value_type('optimizer', optimizer, (nn.Optimizer,))
        self.parameters = optimizer.parameters
        self.hyper_map = C.HyperMap()
        _device_number = 8
        group_number = _get_stage_device_num() // _device_number
        self.grad_clone = ParameterTuple(self.parameters)
        self.adasum = _AdaSum(_get_global_rank(), _device_number, group_number, self.grad_clone)
        self.sync_tensor = Parameter(Tensor(0, dtype=mstype.int32))
        self.scale = Tensor(1.0, dtype=mstype.float32)

    def construct(self, grads):
        grad_clone = self.hyper_map(F.partial(_clone_weight, self.scale), self.parameters)
        grads = F.depend(grads, grad_clone)
        opt_result = self.optimizer(grads)
        parameters = F.depend(self.parameters, opt_result)
        delta_w = self.hyper_map(F.partial(_get_delta_weight), parameters, grad_clone)
        adasum_res = self.adasum(delta_w, parameters, grad_clone)
        sync_tensor = F.depend(self.sync_tensor, adasum_res)
        sync_flag = P.AllReduce()(sync_tensor)
        updated_weights = F.depend(parameters, sync_flag)
        return updated_weights
