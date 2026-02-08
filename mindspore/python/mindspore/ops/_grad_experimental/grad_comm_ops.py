# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
from __future__ import division
from __future__ import absolute_import
import os
from mindspore import Tensor, Parameter
import mindspore.common.dtype as mstype
from mindspore.ops import functional as F
from mindspore.communication import get_rank, get_group_size
from mindspore.parallel._utils import _get_enable_parallel_optimizer, _get_grad_accumulation_shard
from mindspore.ops import operations as P
from mindspore.ops import Send, Receive
from mindspore.ops.operations._inner_ops import issubclass_
from mindspore.common.sparse_tensor import RowTensorInner
from mindspore.ops.composite.multitype_ops.zeros_like_impl import zeros_like
from mindspore.ops.operations.comm_ops import (AllGather, _MiniStepAllGather, _HostAllGather, AllReduce,
                                               NeighborExchange, AlltoAll, AlltoAllV, NeighborExchangeV2,
                                               Broadcast, AlltoAllVC, AllGatherV, ReduceScatterV,
                                               _GetTensorSlice, _MirrorOperator, _MirrorMiniStepOperator, ReduceOp,
                                               ReduceScatter, _HostReduceScatter, _VirtualDiv, _VirtualAdd, _AllSwap,
                                               _VirtualAssignAdd, _VirtualAccuGrad, _MirrorMicroStepOperator,
                                               _MicroStepAllGather, Reduce, CollectiveGather, CollectiveScatter,
                                               _VirtualAssignKvCache)
from mindspore.ops._grad_experimental.grad_base import bprop_getters
from mindspore.ops.operations import _grad_ops as G
from mindspore.tools.sdc_detect import _sdc_detector
import mindspore as ms

_squared_device_local_norm = None


def get_squared_device_local_norm_param():
    """
    Get Parameter `_squared_device_local_norm`.
    `_squared_device_local_norm` will accumulate squared local norm of each grad in bprop under GRAPH_MODE.
    User need to reset it to zero after network propagation each step.
    """
    global _squared_device_local_norm
    if _squared_device_local_norm is None:
        if ms.get_auto_parallel_context("dump_device_local_norm"):
            _squared_device_local_norm = Parameter(Tensor(0.0, mstype.float32), name="_squared_device_local_norm",
                                                   requires_grad=False)
        else:
            raise ValueError("The parallel config 'dump_device_local_norm' is False.")
    return _squared_device_local_norm


@bprop_getters.register(AllReduce)
def get_bprop_all_reduce(self):
    """Generate bprop for AllReduce, do allreduce or allgather, allgather for sparse feature."""

    all_reduce_grad = AllReduce(ReduceOp.SUM, self.group)
    all_gather = AllGather(group=self.group)
    if hasattr(self, "instance_name") and self.instance_name:
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
            if issubclass_(F.typeof(dout), mstype.tensor_type):
                dx = all_reduce_grad(dout)
            else:
                indices = all_gather(dout.indices)
                grad = all_gather(dout.values)
                dx = RowTensorInner(indices, grad, dout.dense_shape)
            return (dx,)
    else:

        def bprop(x, out, dout):
            if issubclass_(F.typeof(dout), mstype.tensor_type):
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
                dx = RowTensorInner(indices, grad, dout.dense_shape)
            return (dx,)
    return bprop


@bprop_getters.register(Send)
def get_bprop_send(self):
    """Generate bprop for Send."""
    shape = self.get_attr_dict()["shape"]
    dtype = self.get_attr_dict()["dtype"]
    tag = self.get_attr_dict()["sr_tag"]
    send_grad = Receive(tag, self.rank, shape, dtype, self.group_back)
    if "dst_global_rank" in self.get_attr_dict():
        dst_global_rank = self.get_attr_dict().get("dst_global_rank")
        send_grad.add_prim_attr("src_global_rank", dst_global_rank)
    if "RING_ATTENTION_INDEX" in self.get_attr_dict():
        ringattention = self.get_attr_dict().get("RING_ATTENTION_INDEX")
        send_grad.add_prim_attr("RING_ATTENTION_INDEX", ringattention)
    virtual_input = Tensor(0.0, dtype)

    def bprop(x, out, dout):
        dx = send_grad(virtual_input)
        return (dx,)

    return bprop


@bprop_getters.register(Receive)
def get_bprop_receive(self):
    """Generate bprop for Receive."""
    tag = self.get_attr_dict()["sr_tag"]
    flash_tag = self.get_attr_dict().get("flash_tag")
    receive_grad = Send(tag, self.rank, self.group_back)
    shape = self.get_attr_dict()["shape"]
    receive_grad.add_prim_attr("shape", shape)
    if "src_global_rank" in self.get_attr_dict():
        src_global_rank = self.get_attr_dict().get("src_global_rank")
        receive_grad.add_prim_attr("dst_global_rank", src_global_rank)
    if "RING_ATTENTION_INDEX" in self.get_attr_dict():
        ringattention = self.get_attr_dict().get("RING_ATTENTION_INDEX")
        receive_grad.add_prim_attr("RING_ATTENTION_INDEX", ringattention)
    depend = P.Depend()
    cast = P.Cast()
    out_tensor = Tensor(0.0, mstype.float16)
    is_opt_shard = _get_enable_parallel_optimizer()

    def bprop(x, out, dout):
        send_out = receive_grad(dout)
        if is_opt_shard or (flash_tag == "True"):
            dx = depend(F.zeros_like(x), send_out)
        else:
            dx = depend(cast(out_tensor, F.dtype(x)), send_out)
        return (dx,)

    return bprop


@bprop_getters.register(_VirtualAdd)
def get_bprop_virtual_add(self):
    """Generate bprop for _VirtualAdd"""

    def bprop(x, grad_accu, out, dout):
        return (dout + grad_accu, zeros_like(grad_accu))

    return bprop


@bprop_getters.register(_VirtualAssignAdd)
def get_bprop_virtual_assign_add(self):
    """Generate bprop for VirtualAssignAdd."""
    assign_add = P.AssignAdd()
    cast = P.Cast()
    dtype = P.DType()
    out_tensor = Tensor(0.0, mstype.float16)
    reduce_scatter = None
    group = self.get_attr_dict().get("group", None)
    fusion = self.get_attr_dict().get("fusion", 0)
    if group:
        reduce_scatter = ReduceScatter(ReduceOp.SUM, group).add_prim_attr("fusion", fusion)
        if self.instance_name:
            instance_name = "_grad_accumulation_shard_grad" + self.instance_name
            reduce_scatter.set_prim_instance_name(instance_name)
            # For pipeline training, as the fused communication will be visited later
            # this may make memory increase, so we need to add a tag to let the
            # fused communication not be effective.
            reduce_scatter.add_prim_attr("not_delay_fusion", True)

    def bprop(x, y, out, dout):
        if reduce_scatter:
            dout = reduce_scatter(cast(dout, dtype(y)))
        return F.depend((cast(out_tensor, dtype(x)), cast(out_tensor, dtype(y))), assign_add(y, dout))

    return bprop


@bprop_getters.register(_VirtualAssignKvCache)
def get_bprop_virtual_assign_kv_cache(self):
    """Generate bprop for VirtualAssignAdd."""
    assign = P.Assign()
    cast = P.Cast()
    dtype = P.DType()
    out_tensor = Tensor(0.0, mstype.float16)

    def bprop(x, y, seq_chunk, out, dout):
        dout_update = dout + y
        kv_equal = F.equal(seq_chunk, 0)
        update_kv = F.select(kv_equal, F.broadcast_to(cast(out_tensor, dtype(y)), F.shape(y)), dout_update)
        return F.depend((cast(dout_update, dtype(dout)), cast(out_tensor, dtype(y)),
                         cast(out_tensor, dtype(seq_chunk))), assign(y, update_kv))

    return bprop


@bprop_getters.register(_VirtualAccuGrad)
def get_bprop_virtual_accu_grad(self):
    """Generate bprop for VirtualAccuGrad."""
    cast = P.Cast()
    dtype = P.DType()
    out_tensor = Tensor(0.0, mstype.float16)

    def bprop(x, y, out, dout):
        return (F.depend(y, dout), cast(out_tensor, dtype(y)))

    return bprop


@bprop_getters.register(_MirrorMicroStepOperator)
def get_bprop_mirror_micro_step_operator(self):
    """
    Backpropagator for _MirrorMicroStepOperator, do allreduce or allgather for the devices in the group,
    allgather for sparse feature.
    """
    group = self.group
    dev_num = self.dev_num
    mean_flag = self.mean_flag
    param_name = " "
    if 'mirror_user_id' in self.get_attr_dict():
        param_name = self.get_attr_dict()['mirror_user_id']
    scale = 1 / dev_num

    all_reduce = AllReduce(group=group)
    if "segment" in self.get_attr_dict():
        all_reduce.add_prim_attr("segment", self.get_attr_dict()["segment"])
    fusion = self.get_attr_dict()["fusion"]
    all_reduce.add_prim_attr("fusion", fusion)
    if hasattr(self, 'parameter'):
        parameter = self.parameter
        all_reduce.add_prim_attr("parameter", parameter)
    if self.instance_name:
        instance_name = "grad_mirror" + self.instance_name
        all_reduce.set_prim_instance_name(instance_name)
    cast = P.Cast()
    dtype = P.DType()
    assign = P.Assign()
    if "parameter_micro" in self.get_attr_dict():
        assign.add_prim_attr("parameter_micro", 0)
    out_tensor = Tensor(1.0, mstype.float16)
    opt_shard = _get_enable_parallel_optimizer()
    ln_print = P.Print()
    tensor_dump = P.TensorDump()
    reduce_sum = P.ReduceSum(keep_dims=False)
    square = P.Square()
    sqrt = P.Sqrt()
    dump_local_norm = ms.get_auto_parallel_context("dump_local_norm")
    dump_local_norm_path = ms.get_auto_parallel_context("dump_local_norm_path")
    dump_device_local_norm = ms.get_auto_parallel_context("dump_device_local_norm")
    if dump_local_norm_path:
        global_rank = get_rank()
        file = os.path.join(dump_local_norm_path, "rank_" + str(global_rank), "local_norm__" + param_name)
    if dump_device_local_norm:
        # init _squared _squared_device_local_norm
        squared_device_local_norm = get_squared_device_local_norm_param()
    # feature value sampling for sdc detect
    feat_value_dump_name = _sdc_detector.get_dump_name(param_name) if _sdc_detector.need_sample() else None

    def bprop(x, z, out, dout):
        if dump_local_norm or dump_device_local_norm:
            squared_norm = reduce_sum(square((z)))
            if dump_local_norm:
                if dump_local_norm_path:
                    z = F.depend(z, tensor_dump(file, sqrt(squared_norm)))
                else:
                    z = F.depend(z, ln_print("dump local norm: ", param_name, sqrt(squared_norm)))
            if dump_device_local_norm:
                z = F.depend(z, F.assign_add(squared_device_local_norm,
                                             cast(squared_norm, squared_device_local_norm.dtype)))
        if feat_value_dump_name and z.ndim > 1:
            feat_value = square(F.max(F.abs(z))[0])
            z = F.depend(z, tensor_dump(feat_value_dump_name, feat_value))
        real_grad = z
        assign_out = dout
        if issubclass_(F.typeof(dout), mstype.tensor_type):
            z = F.depend(z, dout)
            if dev_num > 1:
                real_grad = all_reduce(z)
                if mean_flag:
                    real_grad = F.tensor_mul(real_grad, scale)
            else:
                real_grad = z
            if opt_shard:
                return (real_grad, cast(out_tensor, dtype(z)))
            return F.depend((cast(out_tensor, dtype(x)), cast(out_tensor, dtype(z))), assign(z, real_grad))
        return F.depend((cast(out_tensor, dtype(x)), cast(out_tensor, dtype(z))), assign_out)

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
    self.group = self.get_attr_dict()["group"]
    reduce_scatter = ReduceScatter(ReduceOp.SUM, self.group).add_prim_attr("fusion", fusion)
    if hasattr(self, "instance_name") and self.instance_name:
        instance_name = "grad_" + self.instance_name
        reduce_scatter.set_prim_instance_name(instance_name)
    mean_flag = self.get_attr_dict()["mean_flag"]
    self.rank_size = self.get_attr_dict()["rank_size"]
    if self.rank_size == 0:
        raise ValueError(f"The 'rank_size' can not be zero, but got {self.rank_size}.")
    scale = 1.0 / self.rank_size
    param_name = ""
    if 'mirror_user_id' in self.get_attr_dict():
        param_name = self.get_attr_dict()['mirror_user_id']
    # monitor local norm
    dump_local_norm = ms.get_auto_parallel_context("dump_local_norm")
    dump_local_norm_path = ms.get_auto_parallel_context("dump_local_norm_path")
    dump_device_local_norm = ms.get_auto_parallel_context("dump_device_local_norm")
    if param_name and (dump_local_norm or dump_device_local_norm):
        cast = P.Cast()
        ln_print = P.Print()
        tensor_dump = P.TensorDump()
        reduce_sum = P.ReduceSum(keep_dims=False)
        sqrt = P.Sqrt()
    square = P.Square()
    if dump_local_norm_path:
        global_rank = get_rank()
        file = os.path.join(dump_local_norm_path, "rank_" + str(global_rank), "local_norm__" + param_name)
    if dump_device_local_norm:
        # init _squared _squared_device_local_norm
        squared_device_local_norm = get_squared_device_local_norm_param()
    # feature value sampling for sdc detect
    feat_value_dump_name = _sdc_detector.get_dump_name(param_name) if _sdc_detector.need_sample() else None

    def bprop(x, out, dout):
        if param_name and (dump_local_norm or dump_device_local_norm):
            squared_norm = reduce_sum(square((dout)))
            if dump_local_norm:
                if dump_local_norm_path:
                    dout = F.depend(dout, tensor_dump(file, sqrt(squared_norm)))
                else:
                    dout = F.depend(dout, ln_print("dump local norm: ", param_name, sqrt(squared_norm)))
            if dump_device_local_norm:
                dout = F.depend(dout, F.assign_add(squared_device_local_norm,
                                                   cast(squared_norm, squared_device_local_norm.dtype)))
        if param_name and feat_value_dump_name and dout.ndim > 1:
            feat_value = square(F.max(F.abs(dout))[0])
            dout = F.depend(dout, tensor_dump(feat_value_dump_name, feat_value))

        dx = reduce_scatter(dout)
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
    add_accu = self.get_attr_dict().get("add_accu", False)
    gradient_shard = _get_grad_accumulation_shard()
    scale = 1 / self.rank_size
    all_reduce = AllReduce(ReduceOp.SUM, self.group).add_prim_attr("fusion", fusion)
    assign_add = P.AssignAdd()
    if hasattr(self, "instance_name") and self.instance_name:
        instance_name = "grad_" + self.instance_name
        all_reduce.set_prim_instance_name(instance_name)
    rank = get_rank(self.group)
    dev_num = get_group_size(self.group)
    split = P.Split(output_num=dev_num)

    def bprop(x, z, out, dout):
        if do_mirror:
            if not gradient_shard:
                z = F.depend(z, F.assign_add(z, dout))
                grad = all_reduce(z)
                dx = split(grad)[rank]
                if mean_flag:
                    dx = F.tensor_mul(dx, scale)
            else:
                dout = F.depend(dout, z)
                grad = all_reduce(dout)
                dx = split(grad)[rank]
                if mean_flag:
                    dx = F.tensor_mul(dx, scale)
                if add_accu:
                    assign_add(z, dx)
                dx = F.depend(dx, z)
        else:
            dx = dout

        return (dx, zeros_like(z))

    return bprop


@bprop_getters.register(_MicroStepAllGather)
def get_bprop_micro_step_all_gather(self):
    """Generate bprop for _MicroStepAllGather"""
    fusion = self.get_attr_dict()["fusion"]
    mean_flag = self.get_attr_dict()["mean_flag"]
    param_name = " "
    if 'mirror_user_id' in self.get_attr_dict():
        param_name = self.get_attr_dict()['mirror_user_id']
    do_mirror = False
    if self.group != "":
        do_mirror = self.get_attr_dict()["do_mirror"]
    if do_mirror:
        scale = 1.0 / self.rank_size
        reduce_scatter = ReduceScatter(ReduceOp.SUM, self.group).add_prim_attr("fusion", fusion)
        if "segment" in self.get_attr_dict():
            reduce_scatter.add_prim_attr("segment", self.get_attr_dict()["segment"])
        if self.instance_name:
            instance_name = "grad_" + self.instance_name
            reduce_scatter.set_prim_instance_name(instance_name)
    cast = P.Cast()
    dtype = P.DType()
    out_tensor = Tensor(1.0, mstype.float16)
    with_mirror_operator = self.get_attr_dict()["with_mirror_operator"]
    ln_print = P.Print()
    tensor_dump = P.TensorDump()
    reduce_sum = P.ReduceSum(keep_dims=False)
    square = P.Square()
    sqrt = P.Sqrt()
    dump_local_norm = ms.get_auto_parallel_context("dump_local_norm")
    dump_local_norm_path = ms.get_auto_parallel_context("dump_local_norm_path")
    dump_device_local_norm = ms.get_auto_parallel_context("dump_device_local_norm")
    if dump_local_norm_path:
        global_rank = get_rank()
        file = os.path.join(dump_local_norm_path, "rank_" + str(global_rank), "local_norm__" + param_name)
    if dump_device_local_norm:
        # init _squared _squared_device_local_norm
        squared_device_local_norm = get_squared_device_local_norm_param()
    # feature value sampling for sdc detect
    feat_value_dump_name = _sdc_detector.get_dump_name(param_name) if _sdc_detector.need_sample() else None

    def bprop(x, z, out, dout):
        if with_mirror_operator:
            if not do_mirror:
                return (dout, cast(out_tensor, dtype(z)))
            real_grad = reduce_scatter(cast(dout, dtype(z)))
            if mean_flag:
                real_grad = F.tensor_mul(real_grad, scale)
            return (real_grad, cast(out_tensor, dtype(z)))
        z = F.depend(z, dout)
        if dump_local_norm or dump_device_local_norm:
            squared_norm = reduce_sum(square((z)))
            if dump_local_norm:
                if dump_local_norm_path:
                    z = F.depend(z, tensor_dump(file, sqrt(squared_norm)))
                else:
                    z = F.depend(z, ln_print("dump local norm: ", param_name, sqrt(squared_norm)))
            if dump_device_local_norm:
                z = F.depend(z, F.assign_add(squared_device_local_norm,
                                             cast(squared_norm, squared_device_local_norm.dtype)))
        if feat_value_dump_name and z.ndim > 1:
            feat_value = square(F.max(F.abs(z))[0])
            z = F.depend(z, tensor_dump(feat_value_dump_name, feat_value))
        if not do_mirror:
            return (z, cast(out_tensor, dtype(z)))
        real_grad = reduce_scatter(z)
        if mean_flag:
            real_grad = F.tensor_mul(real_grad, scale)
        return (real_grad, cast(out_tensor, dtype(z)))

    return bprop


@bprop_getters.register(_HostAllGather)
def get_bprop_host_all_gather(self):
    """Generate bprop for _HostAllGather"""
    host_all_gather_grad = _HostReduceScatter(ReduceOp.SUM, self.group)
    if hasattr(self, "instance_name") and self.instance_name:
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
    if hasattr(self, "instance_name") and self.instance_name:
        instance_name = "grad" + self.instance_name
        reduce_scatter_grad.set_prim_instance_name(instance_name)

    if self.op != ReduceOp.SUM:
        raise RuntimeError("The reducescatter bprop only support ReduceOp.SUM until now.")

    def bprop(x, out, dout):
        dx = reduce_scatter_grad(dout)
        return (dx,)

    return bprop


@bprop_getters.register(Reduce)
def get_bprop_reduce(self):
    """Generate bprop for Reduce"""
    dest_rank = self.get_attr_dict()["dest_rank"]
    group = self.get_attr_dict()["group"]
    reduce_grad = Broadcast(dest_rank, group)
    if hasattr(self, "instance_name") and self.instance_name:
        instance_name = "grad" + self.instance_name
        reduce_grad.set_prim_instance_name(instance_name)

    def bprop(x, out, dout):
        dx = reduce_grad((dout,))
        return (dx[0],)

    return bprop


@bprop_getters.register(CollectiveGather)
def get_bprop_collective_gather(self):
    """Generate bprop for CollectiveGather"""
    group = self.get_attr_dict()["group"]
    dest_rank = self.get_attr_dict()["dest_rank"]
    collective_gather_grad = Broadcast(dest_rank, group)
    rank = get_rank(group)
    dev_num = self.rank_size
    split = P.Split(output_num=dev_num)
    if hasattr(self, "instance_name") and self.instance_name:
        instance_name = "grad" + self.instance_name
        collective_gather_grad.set_prim_instance_name(instance_name)

    def bprop(x, out, dout):
        grad = collective_gather_grad((dout,))
        dx = split(grad[0])[rank]
        return (dx,)

    return bprop


@bprop_getters.register(CollectiveScatter)
def get_bprop_collective_scatter(self):
    """Generate bprop for CollectiveScatter"""
    group = self.get_attr_dict()["group"]
    dest_rank = self.get_attr_dict()["src_rank"]
    rank = get_rank(group)
    collective_scatter_grad = CollectiveGather(dest_rank, group)
    if hasattr(self, "instance_name") and self.instance_name:
        instance_name = "grad" + self.instance_name
        collective_scatter_grad.set_prim_instance_name(instance_name)

    def bprop(x, out, dout):
        dx_out = collective_scatter_grad(dout)
        if rank == dest_rank:
            dx = dx_out
        else:
            dx = F.depend(F.zeros_like(x), dx_out)
        return (dx,)

    return bprop


@bprop_getters.register(_AllSwap)
def get_bprop_allswap(self):
    """Generate bprop for _AllSwap."""
    all_swap_grad = _AllSwap(self.group)
    if hasattr(self, "instance_name") and self.instance_name:
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
    if hasattr(self, "instance_name") and self.instance_name:
        instance_name = "grad" + self.instance_name
        host_reduce_scatter_grad.set_prim_instance_name(instance_name)

    if self.op != ReduceOp.SUM:
        raise RuntimeError("The hostreducescatter bprop only support ReduceOp.SUM until now.")

    def bprop(x, out, dout):
        dx = host_reduce_scatter_grad(dout)
        return (dx,)

    return bprop


@bprop_getters.register(NeighborExchange)
def get_bprop_neighborexchange(self):
    """Generate bprop for NeighborExchange."""
    group = self.group
    send_rank_ids = self.recv_rank_ids
    recv_rank_ids = self.send_rank_ids
    recv_shapes = self.send_shapes
    send_shapes = self.recv_shapes
    recv_type = self.recv_type
    neighborexchange_grad = NeighborExchange(send_rank_ids, recv_rank_ids, recv_shapes, send_shapes, recv_type, group)

    def bprop(x, out, dout):
        return (neighborexchange_grad(dout),)

    return bprop


@bprop_getters.register(AlltoAll)
def get_bprop_all_to_all(self):
    """Generate bprop for AlltoAll."""
    all_to_all_grad = AlltoAll(self.split_count, self.concat_dim, self.split_dim, self.group)
    if hasattr(self, "instance_name") and self.instance_name:
        instance_name = "grad" + self.instance_name
        all_to_all_grad.set_prim_instance_name(instance_name)

    def bprop(x, out, dout):
        dx = all_to_all_grad(dout)
        return (dx,)

    return bprop


@bprop_getters.register(AlltoAllV)
def get_bprop_all_to_all_v(self):
    """Generate bprop for AlltoAll."""
    all_to_all_v_grad = AlltoAllV(self.group, self.block_size)
    if hasattr(self, "instance_name") and self.instance_name:
        instance_name = "grad" + self.instance_name
        all_to_all_v_grad.set_prim_instance_name(instance_name)

    def bprop(x, send_numel_list, recv_numel_list, out, dout):
        dx = all_to_all_v_grad(dout, recv_numel_list, send_numel_list)
        return (dx, zeros_like(send_numel_list), zeros_like(recv_numel_list))

    return bprop


@bprop_getters.register(AlltoAllVC)
def get_bprop_all_to_all_v_c(self):
    """Generate bprop for AlltoAllVC."""
    all_to_all_v_c_grad = AlltoAllVC(self.group, self.block_size, transpose=True)
    if hasattr(self, "instance_name") and self.instance_name:
        instance_name = "grad" + self.instance_name
        all_to_all_v_c_grad.set_prim_instance_name(instance_name)

    def bprop(x, send_count_matrix, out, dout):
        dx = all_to_all_v_c_grad(dout, send_count_matrix)
        return (dx, zeros_like(send_count_matrix))

    return bprop


@bprop_getters.register(AllGatherV)
def get_bprop_all_gather_v(self):
    """Generate bprop for AllGatherV."""
    all_gather_v_grad = ReduceScatterV(ReduceOp.SUM, self.group)
    if hasattr(self, "instance_name") and self.instance_name:
        instance_name = "grad" + self.instance_name
        all_gather_v_grad.set_prim_instance_name(instance_name)

    def bprop(x, output_split_sizes, out, dout):
        dx = all_gather_v_grad(dout, output_split_sizes)
        return (dx, zeros_like(output_split_sizes))

    return bprop


@bprop_getters.register(ReduceScatterV)
def get_bprop_reduce_scatter_v(self):
    """Generate bprop for ReduceScatterV."""
    reduce_scatter_v_grad = AllGatherV(self.group)
    if hasattr(self, "instance_name") and self.instance_name:
        instance_name = "grad" + self.instance_name
        reduce_scatter_v_grad.set_prim_instance_name(instance_name)
    if self.op != ReduceOp.SUM:
        raise RuntimeError("The reducescatter bprop only support ReduceOp.SUM until now.")

    def bprop(x, input_split_sizes, out, dout):
        dx = reduce_scatter_v_grad(dout, input_split_sizes)
        return (dx, zeros_like(input_split_sizes))

    return bprop


@bprop_getters.register(NeighborExchangeV2)
def get_bprop_neighborexchangev2(self):
    """Generate bprop for NeighborExchangeV2."""
    group = self.group
    send_rank_ids = self.recv_rank_ids
    recv_rank_ids = self.send_rank_ids
    send_lens = self.recv_lens
    recv_lens = self.send_lens
    data_format = self.data_format
    neighborexchangev2_grad = G.NeighborExchangeV2Grad(send_rank_ids, send_lens, recv_rank_ids,
                                                       recv_lens, data_format, group)

    def bprop(x, out, dout):
        return (neighborexchangev2_grad(dout),)

    return bprop


@bprop_getters.register(_MirrorOperator)
def get_bprop_mirror_operator(self):
    """
    Backpropagator for _MirrorOperator, do allreduce or allgather for the devices in group(only for one group),
    allgather for sparse feature.
    """
    group = self.get_attr_dict()['group']
    dev_num = self.get_attr_dict()['dev_num']
    mean_flag = self.get_attr_dict()['mean_flag']
    param_name = " "
    if 'mirror_user_id' in self.get_attr_dict():
        param_name = self.get_attr_dict()['mirror_user_id']

    dev_num_r = 1.0
    dump_local_norm = ms.get_auto_parallel_context("dump_local_norm")
    dump_local_norm_path = ms.get_auto_parallel_context("dump_local_norm_path")
    dump_device_local_norm = ms.get_auto_parallel_context("dump_device_local_norm")
    if dump_local_norm_path:
        global_rank = get_rank()
        file = os.path.join(dump_local_norm_path, "rank_" + str(global_rank), "local_norm__" + param_name)
    if dump_device_local_norm:
        # init _squared _squared_device_local_norm
        squared_device_local_norm = get_squared_device_local_norm_param()
    # feature value sampling for sdc detect
    feat_value_dump_name = _sdc_detector.get_dump_name(param_name) if _sdc_detector.need_sample() else None
    if dev_num > 1:
        dev_num_r = 1.0 / dev_num
        all_reduce = AllReduce(group=group)
        all_gather = AllGather(group=group)
        mul = P.Mul()
        cast = P.Cast()
        ln_print = P.Print()
        tensor_dump = P.TensorDump()
        reduce_sum = P.ReduceSum(keep_dims=False)
        square = P.Square()
        sqrt = P.Sqrt()

        fusion = self.get_attr_dict()["fusion"]
        all_reduce.add_prim_attr("fusion", fusion)
        parameter = " "
        if hasattr(self, 'parameter'):
            parameter = self.parameter
            all_reduce.add_prim_attr("parameter", parameter)

        if self.instance_name:
            instance_name = "grad_mirror" + self.instance_name
            all_reduce.set_prim_instance_name(instance_name)

    def bprop(x, out, dout):
        if dump_local_norm or dump_device_local_norm:
            squared_norm = reduce_sum(square((dout)))
            if dump_local_norm:
                if dump_local_norm_path:
                    dout = F.depend(dout, tensor_dump(file, sqrt(squared_norm)))
                else:
                    dout = F.depend(dout, ln_print("dump local norm: ", param_name, sqrt(squared_norm)))
            if dump_device_local_norm:
                dout = F.depend(dout, F.assign_add(squared_device_local_norm,
                                                   cast(squared_norm, squared_device_local_norm.dtype)))
        if feat_value_dump_name and dout.ndim > 1:
            feat_value = square(F.max(F.abs(dout))[0])
            dout = F.depend(dout, tensor_dump(feat_value_dump_name, feat_value))

        if dev_num == 1:
            return (dout,)
        if mean_flag:
            if issubclass_(F.typeof(dout), mstype.tensor_type):
                dx = all_reduce(dout)
                dx = mul(dx, cast(F.scalar_to_tensor(dev_num_r), F.dtype(dx)))
            else:
                indices = all_gather(dout.indices)
                grad = all_gather(dout.values)
                grad = mul(grad, cast(F.scalar_to_tensor(dev_num_r), F.dtype(grad)))
                dx = RowTensorInner(indices, grad, dout.dense_shape)
        else:
            if issubclass_(F.typeof(dout), mstype.tensor_type):
                dx = all_reduce(dout)
            else:
                indices = all_gather(dout.indices)
                grad = all_gather(dout.values)
                dx = RowTensorInner(indices, grad, dout.dense_shape)

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
    dev_num_r = 1.0 / dev_num

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
            if issubclass_(F.typeof(dout), mstype.tensor_type):
                if do_mirror:
                    z = F.depend(z, F.assign_add(z, dout))
                    real_grad = all_reduce(z)
                    dx = real_grad
                else:
                    dx = dout
                dx = mul(dx, cast(F.scalar_to_tensor(dev_num_r), F.dtype(dx)))
            else:
                dx = zeros_like(x)  # The grad accumulation do not support row tensor now
        else:
            if issubclass_(F.typeof(dout), mstype.tensor_type):
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
        if issubclass_(F.typeof(dout), mstype.tensor_type):
            if issubclass_(F.dtype(dout), mstype.bool_) or issubclass_(F.dtype(dout), mstype.int32) \
                    or issubclass_(F.dtype(dout), mstype.int16):
                return (dout,)
            dx = op(dout, cast(F.scalar_to_tensor(divisor), dtype(dout)))
            return (dx,)

        if issubclass_(F.typeof(dout), mstype.tuple_):
            dx = ()
            input_nums = F.tuple_len(dout)
            for i in range(input_nums):
                ele_grad = op(dout[i], cast(F.scalar_to_tensor(divisor), dtype(dout[i])))
                dx = dx + (ele_grad,)
            return (dx,)

        dx = []
        input_nums = F.list_len(dout)
        for i in range(input_nums):
            ele_grad = op(dout[i], cast(F.scalar_to_tensor(divisor), dtype(dout[i])))
            dx.append(ele_grad)
        return (dx,)

    return bprop


@bprop_getters.register(_GetTensorSlice)
def get_bprop_get_tensor_slice_operator(self):
    """Backpropagator for _GetTensorSlice"""

    def bprop(x, dev_mat, tensor_map, out, dout):
        return (zeros_like(x),)

    return bprop
