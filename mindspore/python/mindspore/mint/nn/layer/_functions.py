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
"""functions for mint"""
import mindspore
from mindspore import ops, mint
from mindspore import _checkparam as validator
from mindspore.nn.cell import Cell
from mindspore.communication.comm_func import all_gather_into_tensor
from mindspore.communication.comm_func import all_reduce
from mindspore.communication.management import get_rank, get_group_size, GlobalComm, _get_group
from mindspore.ops.auto_generate.gen_ops_prim import BatchNormReduceGrad
from mindspore.ops.auto_generate.gen_ops_prim import BatchNormElemtGrad
from mindspore.ops.primitive import Primitive, prim_arg_register, PrimitiveWithInfer, prim_attr_register
from mindspore.ops.operations.comm_ops import ReduceOp, check_collective_target_dtype

batch_norm_reduce_grad = BatchNormReduceGrad()
batch_norm_elemt_grad = BatchNormElemtGrad()
shape = ops.Shape()


class AllGather(PrimitiveWithInfer):
    """AllGather operation for distributed training."""
    @prim_arg_register
    def __init__(self, group=GlobalComm.WORLD_COMM_GROUP):
        super().__init__(self.__class__.__name__)
        self.group = _get_group(group)
        validator.check_value_type('group', self.group, (str,), self.name)
        self.rank = get_rank(self.group)
        self.rank_size = get_group_size(self.group)
        validator.check('rank', self.rank, 'rank_size', self.rank_size, validator.LT, self.name)
        self.add_prim_attr('rank_size', self.rank_size)
        self.add_prim_attr('group', self.group)
        self.add_prim_attr('fusion', 0)
        self.add_prim_attr('mean_flag', False)
        self.add_prim_attr('no_eliminate', True)

    def __call__(self, combined):
        output, _ = all_gather_into_tensor(combined, group=self.group)
        return output

    def infer_shape(self, x_shape):
        validator.check_positive_int(len(x_shape), "x shape", self.name)
        if x_shape[0] > 0:
            x_shape[0] = x_shape[0] * self.rank_size
        return x_shape

    def infer_dtype(self, x_dtype):
        check_collective_target_dtype('x', x_dtype, self.name)
        return x_dtype


class AllReduce(Primitive):
    """AllReduce operation for distributed training."""
    @prim_attr_register
    def __init__(self, op=ReduceOp.SUM, group=GlobalComm.WORLD_COMM_GROUP):
        """Initialize AllReduce."""
        super().__init__(name="AllReduce")
        self.group = _get_group(group)
        if not isinstance(op, type(ReduceOp.SUM)):
            raise TypeError(f"For '{self.name}', the 'op' must be str, but got {type(op).__name__}.")
        if not isinstance(self.group, str):
            raise TypeError(f"For '{self.name}', the 'group' must be str, "
                            f"but got {type(self.group).__name__}.")
        self.op = op
        self.add_prim_attr('group', self.group)
        self.add_prim_attr('fusion', 0)
        self.add_prim_attr('index', 0)
        self.add_prim_attr('no_eliminate', True)

    def __call__(self, combined):
        output, _ = all_reduce(combined, group=self.group)
        return output


class SyncBatchNormInner(Cell):
    """SyncBatchNorm inner implementation for distributed training."""
    def __init__(self, self_num_features, self_world_size):
        super().__init__()
        self.num_features = self_num_features
        self.world_size = self_world_size

    def construct(self, input, weight, bias, running_mean, running_var, eps, momentum, process_group, world_size):
        if self.world_size != world_size:
            raise ValueError('World Size Error')
        input = input.contiguous()
        if weight is not None:
            weight = weight.contiguous()

        input_shape = shape(input)
        input_numel = ops.numel(input)
        size = int(input_numel // input_shape[1])
        if size == 1 and world_size < 2:
            raise ValueError(
                'Expected more than 1 value per channel when training, got input size {}'.format(size))

        # calculate mean/invstd for input.
        mean, invstd = mint.batch_norm_stats(input, eps)
        count = mint.full((1,), input_numel // input_shape[1], dtype=mean.dtype)

        num_channels = input_shape[1]
        if self.num_features != num_channels:
            raise ValueError('Features Error')
        # C, C, 1 -> (2C + 1)
        combined = mint.cat([mean, invstd, count], dim=0)
        # Use allgather instead of allreduce because count could be different across
        # ranks, simple all reduce op can not give correct results.
        # batch_norm_gather_stats_with_counts calculates global mean & invstd based on
        # all gathered mean, invstd and count.
        # world_size * (2C + 1)
        all_gather_op = AllGather(process_group)
        combined = all_gather_op(combined)
        combined = ops.reshape(combined, [world_size, -1])
        # world_size * (2C + 1) -> world_size * C, world_size * C, world_size * 1
        mean_val_all, invstd_val_all, count_val_all = mint.split(
            combined, num_channels, dim=1)
        # calculate global mean & invstd
        mean, invstd = mint.batch_norm_gather_stats_with_counts(input, mean_val_all, invstd_val_all, running_mean,
                                                                running_var, momentum, eps, count_val_all.view(-1))

        # apply element-wise normalization
        out = mint.batch_norm_elemt(input, weight, bias, mean, invstd, eps)
        return (out, mean, invstd, count_val_all.view(-1))

    def bprop(self, input_x, weight, bias, running_mean, running_var, eps, momentum,
              process_group, world_size, output, doutput):
        """Backward propagation for SyncBatchNorm."""
        _, mean_param, invstd_param, count_all_param = output
        dout, _, _, _ = doutput

        # 不支持 KBK模式
        dout = dout.contiguous()

        grad_input = grad_weight = grad_bias = None

        inputG = True
        weightG = True
        biasG = True

        # calculate local stats as well as grad_weight / grad_bias
        sum_dy, sum_dy_xmu, grad_weight, grad_bias = batch_norm_reduce_grad(
            dout,
            input_x,
            mean_param,
            invstd_param,
            weight,
            inputG,
            weightG,
            biasG
        )

        if inputG:
            # synchronizing stats used to calculate input gradient.
            sum_dy_shape = shape(sum_dy)
            num_channels = sum_dy_shape[0]
            combined = mint.cat([sum_dy, sum_dy_xmu], dim=0)
            all_reduce_op = AllReduce(group=process_group)
            new_combined = all_reduce_op(combined)

            sum_dy, sum_dy_xmu = mint.split(new_combined, num_channels)

            # backward pass for gradient calculation
            grad_input = batch_norm_elemt_grad(
                dout,
                input_x,
                mean_param,
                invstd_param,
                weight,
                sum_dy,
                sum_dy_xmu,
                count_all_param
            )

        # synchronizing of grad_weight / grad_bias is not needed as distributed
        # training would handle all reduce.
        if weight is None or not weightG:
            grad_weight = None

        if weight is None or not biasG:
            grad_bias = None

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None


class _SyncBatchNorm(Cell):
    """SyncBatchNorm wrapper for distributed training."""
    def __init__(self, num_features, world_size, dtype=mindspore.float32):
        super().__init__()
        self.num_features = num_features
        self.world_size = world_size
        self.inner = SyncBatchNormInner(self.num_features, self.world_size)

    def construct(self, input, weight, bias, running_mean, running_var, eps, momentum, process_group, world_size):
        res = self.inner(input, weight, bias, running_mean,
                         running_var, eps, momentum, process_group, world_size)
        output, _, _, _ = res
        return output
