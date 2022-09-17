# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
from mindspore.nn.cell import Cell
from mindspore.communication.management import create_group
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.ops.operations._inner_ops import Send, Receive


__all__ = ["AdaSum"]


_update_parameters = C.MultitypeFuncGraph("update_parameters")


@_update_parameters.register("Tensor", "Tensor", "Tensor", "Tensor")
def _update_parameters_after_broadcast(delta_weight, update_delta_weight, parameter, old_parameter):
    shape = F.shape(delta_weight)
    update_delta_weight = P.Reshape()(update_delta_weight, shape)
    new_parameter = old_parameter - update_delta_weight
    P.Assign()(parameter, new_parameter)
    return parameter


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
        local_part = F.depend(local_part, recv_part)
        eps = 1e-12
        value_0 = P.ReduceSum()(local_part * recv_part) + eps
        if left_send:
            value_1 = P.ReduceSum()(local_part * local_part) + eps
            value_2 = P.ReduceSum()(recv_part * recv_part) + eps
        else:
            value_1 = P.ReduceSum()(recv_part * recv_part) + eps
            value_2 = P.ReduceSum()(local_part * local_part) + eps
        value_0 = allreduce(value_0)
        value_1 = F.depend(allreduce(value_1), value_0)
        value_2 = F.depend(allreduce(value_2), value_1)
        if left_send:
            res = (1 - (value_0 / (2 * value_1))) * local_part + (1 - (value_0 / (2 * value_2))) * recv_part
        else:
            res = (1 - (value_0 / (2 * value_1))) * recv_part + (1 - (value_0 / (2 * value_2))) * local_part
    else:
        res = allreduce(local_part)
        res /= allreduce_node_num
    return res


_adasum_opt_forward = C.MultitypeFuncGraph("adasum_opt_forward")


@_adasum_opt_forward.register("Bool", "Function", "Bool", "Int64", "Function", "Function", "Tensor")
def _adasum_opt_forward_process(left_send, allreduce, parameter_divisibility, allreduce_node_num, send, recv, delta_w):
    """adasum optimizer process."""
    if parameter_divisibility:
        delta_w = P.Squeeze()(delta_w)
        ori_len = F.shape(delta_w)[0]
        divide_len = ori_len / 2
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


_adasum_opt_rollback = C.MultitypeFuncGraph("adasum_opt_rollback")


@_adasum_opt_rollback.register("Bool", "Bool", "Tensor", "Function", "Function")
def _adasum_opt_rollback_process(left_send, parameter_divisibility, delta_w, send, recv):
    """adasum optimizer rollback process."""
    if parameter_divisibility:
        if left_send:
            recv_part = _send_before_receive(delta_w, send, recv)
        else:
            recv_part = _receive_before_send(delta_w, send, recv)

        recv_part = P.Squeeze()(recv_part)
        recv_part = P.Reshape()(recv_part, (-1,))
        delta_w = P.Reshape()(delta_w, (-1,))

        if left_send:
            res = P.Concat()((recv_part, delta_w))
        else:
            res = P.Concat()((delta_w, recv_part))
    else:
        res = delta_w
    return res


class AdaSum(Cell):
    r"""
    The Adaptive Summation, or AdaSum, is a novel algorithm for improving distributed data
    parallel training of Deep Learning models.

    Args:
        rank (int): Rank number.
        device_number (int): Device number.
        group_number (int): Group number.
        parameter_tuple (Tuple(Parameter)): Tuple of parameters.

    Inputs:
        - **delta_weights** (Tuple(Tensor)) - Tuple of gradients.
        - **parameters** (Tuple(Parameter)) - Tuple of current parameters.
        - **old_parameters** (Tuple(Parameter)) - Tuple of last parameters.

    Outputs:
        - **adasum_parameters** (Tuple(Tensor)) - Tuple of parameters after adasum process.
    """
    def __init__(self, rank, device_number, group_number, parameter_tuple):
        super(AdaSum, self).__init__()
        self.rank = rank
        self.device_number = device_number
        self.group_number = group_number
        self.parameter_tuple = parameter_tuple
        self._generate_communication_op()
        self.hyper_map = C.HyperMap()

    @staticmethod
    def _hash(step, target, weights_index):
        target = "tag" + str(step) + str(target) + str(weights_index)
        target_hash = hashlib.sha1(target.encode()).hexdigest()
        max_num_hash = 2 ** 31
        hash_res = int(int(target_hash, 16) % max_num_hash)
        return hash_res

    def construct(self, delta_weights, parameters, old_parameters):
        forward_weights = [delta_weights]
        for i in range(self.calc_times):
            process_weights = self.hyper_map(F.partial(_adasum_opt_forward, self.send_node[i], self.allreduce_list[i]),
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
                                           parameters, old_parameters)
        return adasum_parameters

    def _generate_communication_op(self):
        """generate communication op."""
        self.calc_times = int(math.log(self.group_number, 2))
        self.send_node = []
        self.send_list_forward = []
        self.recv_list_forward = []
        self.send_list_rollback = []
        self.recv_list_rollback = []
        self.allreduce_list = []
        self.broadcast_list = []
        self.parameter_divisibility_list = []
        self.allreduce_node_num_list = []
        last_delta_weights = []
        group_start_rank = (self.rank // self.device_number) * self.device_number

        for step in range(self.calc_times):
            current_group = self.device_number * (2 ** step)
            sr_target = self.rank
            if (sr_target // current_group) % 2 == 0:
                dest_target = sr_target + current_group
                self.send_node.append(True)
            else:
                dest_target = sr_target - current_group
                self.send_node.append(False)

            neighbor_ids = []
            group_name_last = 0
            for index in range(2 ** (step + 1)):
                node_rank = self.rank // self.device_number
                double_d = 2 ** (step + 1)
                neighbor_id = (node_rank // double_d * double_d + index) * self.device_number + \
                              self.rank % self.device_number
                neighbor_ids.append(neighbor_id)
                group_name_last += neighbor_id
            group_name = "adasum_{}_{}".format(str(step), str(group_name_last))
            create_group(group_name, neighbor_ids)

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
            for shape, dtype in left_delta_weights:
                send_tag = AdaSum._hash(step, sr_target, weights_index)
                send = Send(sr_tag=send_tag, dest_rank=dest_target, group="hccl_world_group")
                send.add_prim_attr("fusion", fusion_id)
                recv_tag = AdaSum._hash(step, dest_target, weights_index)
                recv = Receive(sr_tag=recv_tag, src_rank=dest_target, shape=shape, dtype=dtype,
                               group="hccl_world_group")
                recv.add_prim_attr("fusion", fusion_id)
                send_left.append(send)
                recv_left.append(recv)
                weights_index += 1
            for shape, dtype in right_delta_weights:
                send_tag = AdaSum._hash(step, sr_target, weights_index)
                send = Send(sr_tag=send_tag, dest_rank=dest_target, group="hccl_world_group")
                send.add_prim_attr("fusion", fusion_id + 1)
                recv_tag = AdaSum._hash(step, dest_target, weights_index)
                recv = Receive(sr_tag=recv_tag, src_rank=dest_target, shape=shape, dtype=dtype,
                               group="hccl_world_group")
                recv.add_prim_attr("fusion", fusion_id + 1)
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

            server_all_reduce = P.AllReduce("sum", group_name)
            server_all_reduce.add_prim_attr("fusion", fusion_id + 2)
            self.allreduce_list.append(server_all_reduce)

            for param_divisibility in delta_weights_divisibility:
                if param_divisibility:
                    allreduce_node_num += (0,)
                else:
                    allreduce_node_num += (2 ** (step + 1),)
            self.allreduce_node_num_list.append(allreduce_node_num)

        broadcast_group = list(range(group_start_rank, group_start_rank + self.device_number))
        broadcast_group_name = "broadcast_group_" + str(group_start_rank)
        create_group(broadcast_group_name, broadcast_group)
        for b_rank in range(len(broadcast_group)):
            self.broadcast_list.append(P.Broadcast(b_rank, group=broadcast_group_name))
        self.sync_barrier = P.AllReduce("sum", group=broadcast_group_name)

    def _get_delta_weights_info(self, last_delta_weights):
        """get delta weights info."""
        half_delta_weights = []
        if last_delta_weights:
            half_delta_weights = last_delta_weights
        else:
            for parameter in self.parameter_tuple:
                new_shape = [int(x) for x in parameter.shape]
                half_delta_weights.append((new_shape, parameter.dtype))
        left_delta_weights = []
        right_delta_weights = []
        delta_weights_divisibility = ()
        for shape, dtype in half_delta_weights:
            left_shape = copy.deepcopy(shape)
            right_shape = copy.deepcopy(shape)
            divisibility_flag = False
            for i, _ in enumerate(shape):
                if shape[i] > 1:
                    left_shape[i] = int(shape[i] // 2)
                    right_shape[i] = shape[i] - int(shape[i] // 2)
                    divisibility_flag = True
                    break
            left_delta_weights.append((left_shape, dtype))
            right_delta_weights.append((right_shape, dtype))
            delta_weights_divisibility += (divisibility_flag,)
        return left_delta_weights, right_delta_weights, delta_weights_divisibility
