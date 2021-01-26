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

"""Generate bprop for comm ops"""
import mindspore.common.dtype as mstype
from mindspore.ops import functional as F
from mindspore.communication import get_rank, get_group_size
from .. import operations as P
from ...common.tensor import RowTensor
from ..composite.multitype_ops.zeros_like_impl import zeros_like
from ..operations.comm_ops import (AllGather, _MiniStepAllGather, _HostAllGather, AllReduce, _AlltoAll, Broadcast,
                                   _GetTensorSlice, _MirrorOperator, _MirrorMiniStepOperator, ReduceOp,
                                   ReduceScatter, _HostReduceScatter, _VirtualDiv, _VirtualAdd, AllSwap)
from .grad_base import bprop_getters
from ..operations._inner_ops import Send, Receive


@bprop_getters.register(AllReduce)
def get_bprop_all_reduce(self):
    """Generate bprop for AllReduce, do allreduce or allgather, allgather for sparse feature."""

    all_reduce_grad = AllReduce(ReduceOp.SUM, self.group)
    all_gather = AllGather(group=self.group)
    if self.instance_name:
        instance_name = "grad" + self.instance_name
        all_reduce_grad.set_prim_instance_name(instance_name)
    equal = P.Equal()
    cast = P.Cast()
    mul = P.Mul()
    div = P.RealDiv()
    dtype = P.DType()

    if self.op == ReduceOp.PROD:

        def bprop(x, out, dout):
            dy1 = mul(dout, out)
            dy2 = all_reduce_grad(dy1)
            dx = div(dy2, x)
            return (dx,)

    elif self.op == ReduceOp.SUM:

        def bprop(x, out, dout):
            if F.issubclass_(F.typeof(dout), mstype.tensor):
                dx = all_reduce_grad(dout)
            else:
                indices = all_gather(dout.indices)
                grad = all_gather(dout.values)
                dx = RowTensor(indices, grad, dout.dense_shape)
            return (dx,)
    else:

        def bprop(x, out, dout):
            if F.issubclass_(F.typeof(dout), mstype.tensor):
                dx = all_reduce_grad(dout)
                z = equal(x, out)
                z = cast(z, dtype(dx))
                dx = mul(dx, z)
            else:
                indices = all_gather(dout.indices)
                grad = all_gather(dout.values)
                z = equal(x, out)
                z = cast(z, dtype(grad))
                grad = mul(grad, z)
                dx = RowTensor(indices, grad, dout.dense_shape)
            return (dx,)
    return bprop


@bprop_getters.register(Send)
def get_bprop_send(self):
    """Generate bprop for Send."""
    shape = self.get_attr_dict()["shape"]
    dtype = self.get_attr_dict()["dtype"]
    send_grad = Receive(self.sr_tag, self.rank, shape, dtype, self.group)
    send_grad.add_prim_attr("backward", True)

    def bprop(x, out, dout):
        dx = send_grad()
        return (dx,)
    return bprop


@bprop_getters.register(Receive)
def get_bprop_receive(self):
    """Generate bprop for Receive."""
    receive_grad = Send(self.tag, self.rank, self.group)
    receive_grad.add_prim_attr("backward", True)
    depend = P.Depend()
    cast = P.Cast()

    def bprop(x, out, dout):
        send_out = receive_grad(dout)
        dx = depend(cast(zeros_like(x), F.dtype(x)), send_out)
        return (dx,)
    return bprop


@bprop_getters.register(_VirtualAdd)
def get_bprop_virtual_add(self):
    """Generate bprop for _VirtualAdd"""
    def bprop(x, grad_accu, out, dout):
        return (dout + grad_accu, zeros_like(grad_accu))
    return bprop


@bprop_getters.register(Broadcast)
def get_bprop_broad_cast(self):
    """Generate bprop for Broadcast."""

    def bprop(x, out, dout):
        return (dout,)
    return bprop


@bprop_getters.register(AllGather)
def get_bprop_all_gather(self):
    """Generate bprop for AllGather"""
    fusion = self.get_attr_dict()["fusion"]
    if fusion == 0:
        reduce_scatter = ReduceScatter(ReduceOp.SUM, self.group)
        if self.instance_name:
            instance_name = "grad_" + self.instance_name
            reduce_scatter.set_prim_instance_name(instance_name)
    else:
        all_reduce = AllReduce(ReduceOp.SUM, self.group).add_prim_attr("fusion", fusion)
        if self.instance_name:
            instance_name = "grad_" + self.instance_name
            all_reduce.set_prim_instance_name(instance_name)
        rank = get_rank(self.group)
        dev_num = get_group_size(self.group)
        split = P.Split(output_num=dev_num)
        mean_flag = self.get_attr_dict()["mean_flag"]
        scale = 1/self.rank_size

    def bprop(x, out, dout):
        if fusion == 0:
            dx = reduce_scatter(dout)
        else:
            grad = all_reduce(dout)
            dx = split(grad)[rank]
            if mean_flag:
                dx = F.tensor_mul(dx, scale)
        return (dx,)

    return bprop


@bprop_getters.register(_MiniStepAllGather)
def get_bprop_mini_step_all_gather(self):
    """Generate bprop for _MiniStepAllGather"""
    fusion = self.get_attr_dict()["fusion"]
    mean_flag = self.get_attr_dict()["mean_flag"]
    do_mirror = self.get_attr_dict()["do_mirror"]
    scale = 1 / self.rank_size
    all_reduce = AllReduce(ReduceOp.SUM, self.group).add_prim_attr("fusion", fusion)
    if self.instance_name:
        instance_name = "grad_" + self.instance_name
        all_reduce.set_prim_instance_name(instance_name)
    rank = get_rank(self.group)
    dev_num = get_group_size(self.group)
    split = P.Split(output_num=dev_num)

    def bprop(x, z, out, dout):
        if do_mirror:
            if mean_flag:
                z = F.depend(z, F.assign_add(z, dout))
                grad = all_reduce(z)
                dx = split(grad)[rank]
                dx = F.tensor_mul(dx, scale)
            else:
                z = F.depend(z, F.assign_add(z, dout))
                grad = all_reduce(z)
                dx = split(grad)[rank]
        else:
            dx = dout
        return (dx, zeros_like(z))

    return bprop


@bprop_getters.register(_HostAllGather)
def get_bprop_host_all_gather(self):
    """Generate bprop for _HostAllGather"""
    host_all_gather_grad = _HostReduceScatter(ReduceOp.SUM, self.group)
    if self.instance_name:
        instance_name = "grad" + self.instance_name
        host_all_gather_grad.set_prim_instance_name(instance_name)

    def bprop(x, out, dout):
        dx = host_all_gather_grad(dout)
        return (dx,)

    return bprop


@bprop_getters.register(ReduceScatter)
def get_bprop_reduce_scatter(self):
    """Generate bprop for ReduceScatter"""
    reduce_scatter_grad = AllGather(self.group)
    if self.instance_name:
        instance_name = "grad" + self.instance_name
        reduce_scatter_grad.set_prim_instance_name(instance_name)

    if self.op != ReduceOp.SUM:
        raise RuntimeError("The reducescatter bprop only support ReduceOp.SUM until now.")

    def bprop(x, out, dout):
        dx = reduce_scatter_grad(dout)
        return (dx,)

    return bprop


@bprop_getters.register(AllSwap)
def get_bprop_allswap(self):
    """Generate bprop for AllSwap."""
    all_swap_grad = AllSwap(self.group)
    if self.instance_name:
        instance_name = "grad" + self.instance_name
        all_swap_grad.set_prim_instance_name(instance_name)

    def bprop(x, send_size, recv_size, out, dout):
        dx = all_swap_grad(dout, recv_size, send_size)
        return (dx, zeros_like(send_size), zeros_like(recv_size))

    return bprop


@bprop_getters.register(_HostReduceScatter)
def get_bprop_host_reduce_scatter(self):
    """Generate bprop for _HostReduceScatter"""
    host_reduce_scatter_grad = _HostAllGather(self.group)
    if self.instance_name:
        instance_name = "grad" + self.instance_name
        host_reduce_scatter_grad.set_prim_instance_name(instance_name)

    if self.op != ReduceOp.SUM:
        raise RuntimeError("The hostreducescatter bprop only support ReduceOp.SUM until now.")

    def bprop(x, out, dout):
        dx = host_reduce_scatter_grad(dout)
        return (dx,)

    return bprop


@bprop_getters.register(_AlltoAll)
def get_bprop_all_to_all(self):
    """Generate bprop for AlltoAll."""
    all_to_all_grad = _AlltoAll(self.split_count, self.concat_dim, self.split_dim, self.group)
    if self.instance_name:
        instance_name = "grad" + self.instance_name
        all_to_all_grad.set_prim_instance_name(instance_name)

    def bprop(x, out, dout):
        dx = all_to_all_grad(dout)
        return (dx,)

    return bprop


@bprop_getters.register(_MirrorOperator)
def get_bprop_mirror_operator(self):
    """
    Backpropagator for _MirrorOperator, do allreduce or allgather for the devices in group(only for one group),
    allgather for sparse feature.
    """
    group = self.group
    dev_num = self.dev_num
    mean_flag = self.mean_flag

    all_reduce = AllReduce(group=group)
    all_gather = AllGather(group=group)
    mul = P.Mul()
    cast = P.Cast()

    fusion = self.get_attr_dict()["fusion"]
    all_reduce.add_prim_attr("fusion", fusion)
    if hasattr(self, 'parameter'):
        parameter = self.parameter
        all_reduce.add_prim_attr("parameter", parameter)

    if self.instance_name:
        instance_name = "grad_mirror" + self.instance_name
        all_reduce.set_prim_instance_name(instance_name)

    def bprop(x, out, dout):
        if mean_flag:
            if F.issubclass_(F.typeof(dout), mstype.tensor):
                dx = all_reduce(dout)
                float_one = F.scalar_cast(1.0, F.dtype(dx))
                num = F.scalar_cast(dev_num, F.dtype(dx))
                dx = mul(dx, cast(F.scalar_to_array(float_one/num), F.dtype(dx)))
            else:
                indices = all_gather(dout.indices)
                grad = all_gather(dout.values)
                float_one = F.scalar_cast(1.0, F.dtype(grad))
                num = F.scalar_cast(dev_num, F.dtype(grad))
                grad = mul(grad, cast(F.scalar_to_array(float_one/num), F.dtype(grad)))
                dx = RowTensor(indices, grad, dout.dense_shape)
        else:
            if F.issubclass_(F.typeof(dout), mstype.tensor):
                dx = all_reduce(dout)
            else:
                indices = all_gather(dout.indices)
                grad = all_gather(dout.values)
                dx = RowTensor(indices, grad, dout.dense_shape)

        return (dx,)
    return bprop


@bprop_getters.register(_MirrorMiniStepOperator)
def get_bprop_mirror_mini_step_operator(self):
    """
    Backpropagator for _MirrorMiniStepOperator, do allreduce or allgather for the devices in the group,
    allgather for sparse feature.
    """
    group = self.group
    dev_num = self.dev_num
    mean_flag = self.mean_flag

    all_reduce = AllReduce(group=group)
    mul = P.Mul()
    cast = P.Cast()

    fusion = self.get_attr_dict()["fusion"]
    all_reduce.add_prim_attr("fusion", fusion)
    if hasattr(self, 'parameter'):
        parameter = self.parameter
        all_reduce.add_prim_attr("parameter", parameter)

    if self.instance_name:
        instance_name = "grad_mirror" + self.instance_name
        all_reduce.set_prim_instance_name(instance_name)
    do_mirror = self.get_attr_dict()["do_mirror"]

    def bprop(x, z, out, dout):
        if mean_flag:
            if F.issubclass_(F.typeof(dout), mstype.tensor):
                if do_mirror:
                    z = F.depend(z, F.assign_add(z, dout))
                    real_grad = all_reduce(z)
                    dx = real_grad
                else:
                    dx = dout
                float_one = F.scalar_cast(1.0, F.dtype(dx))
                num = F.scalar_cast(dev_num, F.dtype(dx))
                dx = mul(dx, cast(F.scalar_to_array(float_one/num), F.dtype(dx)))
            else:
                dx = zeros_like(x)  # The grad accumulation do not support row tensor now
        else:
            if F.issubclass_(F.typeof(dout), mstype.tensor):
                if do_mirror:
                    z = F.depend(z, F.assign_add(z, dout))
                    real_grad = all_reduce(z)
                    dx = real_grad
                else:
                    dx = dout
            else:
                dx = zeros_like(x)  # The grad accumulation do not support row tensor now

        return (dx, zeros_like(z))
    return bprop


@bprop_getters.register(_VirtualDiv)
def get_bprop_virtual_div_operator(self):
    """Backpropagator for _VirtualDiv, do Div for the divisor."""
    divisor = self.divisor
    op = P.RealDiv()
    cast = P.Cast()
    dtype = P.DType()

    def bprop(x, out, dout):
        if F.issubclass_(F.typeof(dout), mstype.tensor):
            if F.issubclass_(F.dtype(dout), mstype.bool_) or F.issubclass_(F.dtype(dout), mstype.int32) \
                                     or F.issubclass_(F.dtype(dout), mstype.int16):
                return (dout,)
            dx = op(dout, cast(F.scalar_to_array(divisor), dtype(dout)))
            return (dx,)

        if F.issubclass_(F.typeof(dout), mstype.tuple_):
            dx = ()
            input_nums = F.tuple_len(dout)
            for i in range(input_nums):
                ele_grad = op(dout[i], cast(F.scalar_to_array(divisor), dtype(dout[i])))
                dx = dx + (ele_grad,)
            return (dx,)

        dx = []
        input_nums = F.list_len(dout)
        for i in range(input_nums):
            ele_grad = op(dout[i], cast(F.scalar_to_array(divisor), dtype(dout[i])))
            dx.append(ele_grad)
        return (dx,)
    return bprop


@bprop_getters.register(_GetTensorSlice)
def get_bprop_get_tensor_slice_operator(self):
    """Backpropagator for _GetTensorSlice"""

    def bprop(x, dev_mat, tensor_map, out, dout):
        return (zeros_like(x),)
    return bprop
