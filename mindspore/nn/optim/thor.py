# Copyright 2021 Huawei Technologies Co., Ltd
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
"""THOR"""
import numpy as np
from mindspore.ops import functional as F, composite as C, operations as P
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter, ParameterTuple
from mindspore.common.tensor import Tensor
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore._checkparam import Validator
from mindspore.nn.optim.optimizer import Optimizer
from mindspore.parallel._utils import _get_device_num, _get_gradients_mean
from mindspore import context
from mindspore.context import ParallelMode
from mindspore.nn.layer import Dense_Thor, Conv2d_Thor, Embedding_Thor
from mindspore.nn.wrap import DistributedGradReducer
from mindspore.train.train_thor.convert_utils import ConvertNetUntils
from mindspore.parallel._auto_parallel_context import auto_parallel_context

# Enumerates types of Layer
Other = -1
Conv = 1
FC = 2
Embedding = 3
LayerNorm = 4
BatchNorm = 5


_momentum_opt = C.MultitypeFuncGraph("momentum_opt")

op_add = P.AddN()
apply_decay = C.MultitypeFuncGraph("apply_decay")


@apply_decay.register("Number", "Bool", "Tensor", "Tensor")
def _tensor_apply_decay(weight_decay, if_apply, weight, gradient):
    """Get grad with weight_decay."""
    if if_apply:
        return op_add((weight * weight_decay, gradient))
    return gradient


@_momentum_opt.register("Function", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor")
def _tensor_run_opt_ext(opt, momentum, learning_rate, gradient, weight, moment):
    """Apply momentum optimizer to the weight parameter using Tensor."""
    success = True
    success = F.depend(success, opt(weight, moment, learning_rate, gradient, momentum))
    return success

C0 = 16


def caculate_device_shape(matrix_dim, channel, is_A):
    ll = (0)
    if is_A:
        if channel // C0 == 0:
            matrix_dim = (matrix_dim / channel) * C0
        ll = (int(matrix_dim // C0), int(matrix_dim // C0), C0, C0), int(matrix_dim)
    else:
        ll = (int(matrix_dim // C0), int(matrix_dim // C0), C0, C0), int(matrix_dim)
    return ll


def caculate_matmul_shape(matrix_A_dim, matrix_G_dim, split_dim):
    """get matmul shape"""
    split_dimA = split_dim
    split_dimG = split_dim
    if matrix_A_dim % split_dim == 0:
        batch_w = matrix_A_dim // split_dim
    else:
        if matrix_A_dim < split_dim:
            batch_w = 1
            split_dimA = matrix_A_dim
        else:
            batch_w = matrix_A_dim // split_dim + 1

    if matrix_G_dim % split_dim == 0:
        batch_h = matrix_G_dim // split_dim
    else:
        if matrix_G_dim < split_dim:
            batch_h = 1
            split_dimG = matrix_G_dim
        else:
            batch_h = matrix_G_dim // split_dim + 1
    matrix_A_shape = (batch_h, batch_w, split_dimA, split_dimA)
    matrix_G_shape = (batch_h, split_dimG, split_dimG)
    return matrix_A_shape, matrix_G_shape


def find_net_layertype_recur(net, layertype_map):
    """get net layer type recursively."""
    cells = net.name_cells()
    for name in cells:
        subcell = cells[name]
        print("thor subcell name: ", name)
        if subcell == net:
            continue
        elif isinstance(subcell, Conv2d_Thor):
            layertype_map.append(Conv)
        elif isinstance(subcell, Dense_Thor):
            layertype_map.append(FC)
        elif isinstance(subcell, Embedding_Thor):
            layertype_map.append(Embedding)
        elif isinstance(subcell, nn.LayerNorm):
            layertype_map.append(LayerNorm)
        elif isinstance(subcell, nn.BatchNorm2d):
            layertype_map.append(BatchNorm)
        elif isinstance(subcell, (nn.Conv2d, nn.Dense, nn.Embedding, nn.Conv2dTranspose, nn.Conv1d, nn.Conv1dTranspose,
                                  nn.BatchNorm1d, nn.GroupNorm, nn.GlobalBatchNorm)):
            layertype_map.append(Other)
        else:
            find_net_layertype_recur(subcell, layertype_map)

def get_net_layertype_mask(net):
    layertype_map = []
    find_net_layertype_recur(net, layertype_map)
    return layertype_map

def get_layer_counter(layer_type, layer_counter, params, idx):
    """get layer counter"""
    if layer_type in [Conv, FC, LayerNorm, BatchNorm]:
        if layer_type in [LayerNorm, BatchNorm]:
            if "beta" in params[idx].name.lower():
                layer_counter = layer_counter + 1
        else:
            if "bias" in params[idx].name.lower():
                layer_counter = layer_counter + 1
            else:
                if idx < len(params) - 1 and "bias" not in params[idx + 1].name.lower():
                    layer_counter = layer_counter + 1
    else:
        layer_counter = layer_counter + 1
    return layer_counter


def THOR(net, learning_rate, damping, momentum, weight_decay=0.0, loss_scale=1.0, batch_size=32,
         use_nesterov=False, decay_filter=lambda x: x.name not in [], split_indices=None):
    context.set_context(max_call_depth=10000)
    ConvertNetUntils().convert_to_thor_net(net)
    if context.get_context("device_target") == "Ascend":
        return THOR_Ascend(net, learning_rate, damping, momentum, weight_decay, loss_scale, batch_size, decay_filter,
                           split_indices=split_indices)
    return THOR_GPU(net, learning_rate, damping, momentum, weight_decay, loss_scale, batch_size,
                    use_nesterov, decay_filter, split_indices=split_indices)


class THOR_GPU(Optimizer):
    """
    THOR_GPU
    """
    def __init__(self, net, learning_rate, damping, momentum, weight_decay=0.0, loss_scale=1.0, batch_size=32,
                 use_nesterov=False, decay_filter=lambda x: x.name not in [], split_indices=None):
        params = filter(lambda x: x.requires_grad, net.get_parameters())
        super(THOR_GPU, self).__init__(learning_rate, params, weight_decay, loss_scale)
        Validator.check_value_type("momentum", momentum, [float], self.cls_name)
        if isinstance(momentum, float) and momentum < 0.0:
            raise ValueError("momentum should be at least 0.0, but got momentum {}".format(momentum))
        self.momentum = Parameter(Tensor(momentum, mstype.float32), name="momentum")
        self.params = self.parameters
        self.use_nesterov = Validator.check_bool(use_nesterov)
        self.moments = self.params.clone(prefix="moments", init='zeros')
        self.hyper_map = C.HyperMap()
        self.opt = P.ApplyMomentum(use_nesterov=self.use_nesterov)
        self.net = net
        self.matrix_A_cov = ParameterTuple(filter(lambda x: 'matrix_A' in x.name, net.get_parameters()))
        self.matrix_G_cov = ParameterTuple(filter(lambda x: 'matrix_G' in x.name, net.get_parameters()))
        self.A_normalizer = ParameterTuple(filter(lambda x: 'A_normalizer' in x.name, net.get_parameters()))
        self.G_normalizer = ParameterTuple(filter(lambda x: 'G_normalizer' in x.name, net.get_parameters()))
        self.transpose = P.Transpose()
        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.matmul = P.MatMul()
        self.assign = P.Assign()
        self.mul = P.Mul()
        self.damping = damping
        self.gather = P.GatherV2()
        self.one = Tensor(1, mstype.int32)
        self.batch_size = Tensor(batch_size, mstype.float32)
        self.loss_scale = Tensor(1 / (loss_scale * loss_scale), mstype.float32)
        self.batch_size_scale = Tensor(batch_size * batch_size, mstype.float32)
        self.feature_map = Tensor(1.0, mstype.float32)
        self.axis = 0
        self.cov_step = Parameter(initializer(0, [1], mstype.int32), name="cov_step", requires_grad=False)
        self.cast = P.Cast()
        self.sqrt = P.Sqrt()
        self.eye = P.Eye()
        split_dim = 128
        self.embedding_cholesky = P.CholeskyTrsm()
        self.cholesky = P.CholeskyTrsm(split_dim=split_dim)
        self.vector_matmul = P.BatchMatMul(transpose_a=True)
        self.reduce_sum = P.ReduceSum(keep_dims=False)
        self.inv = P.Reciprocal()
        self.square = P.Square()
        self.expand = P.ExpandDims()
        self.thor = True

        self.matrix_A = ()
        self.matrix_G = ()
        self.matrix_A_shape = ()
        self.thor_layer_count = 0
        self.conv_layer_count = 0
        self.weight_fim_idx_map = ()
        self.weight_conv_idx_map = ()
        self.weight_layerType_idx_map = ()
        layer_type_map = get_net_layertype_mask(net)

        layer_counter = 0
        for idx in range(len(self.params)):
            layer_type = layer_type_map[layer_counter]
            weight = self.params[idx]
            weight_shape = self.shape(weight)
            if layer_type in [Conv, FC] and "bias" not in self.params[idx].name.lower():
                in_channels = weight_shape[1]
                out_channels = weight_shape[0]
                matrix_A_dim = in_channels
                if layer_type == Conv:
                    matrix_A_dim = in_channels * weight_shape[2] * weight_shape[3]
                matrix_G_dim = out_channels
                matrix_A_shape, matrix_G_shape = caculate_matmul_shape(matrix_A_dim, matrix_G_dim, split_dim)
                matrix_A_inv = Parameter(np.zeros(matrix_A_shape).astype(np.float32),
                                         name='matrix_A_inv_' + str(self.thor_layer_count), requires_grad=False)
                matrix_G_inv = Parameter(np.zeros(matrix_G_shape).astype(np.float32),
                                         name="matrix_G_inv_" + str(self.thor_layer_count), requires_grad=False)
                self.matrix_A = self.matrix_A + (matrix_A_inv,)
                self.matrix_G = self.matrix_G + (matrix_G_inv,)
                self.matrix_A_shape = self.matrix_A_shape + (matrix_A_shape,)
            elif layer_type == Embedding:
                vocab_size = weight_shape[0]
                embedding_size = weight_shape[1]
                matrix_A_inv = Parameter(Tensor(np.zeros([vocab_size]).astype(np.float32)),
                                         name='matrix_A_inv_' + str(self.thor_layer_count), requires_grad=False)
                matrix_G_inv = Parameter(Tensor(np.zeros([embedding_size, embedding_size]).astype(np.float32)),
                                         name="matrix_G_inv_" + str(self.thor_layer_count), requires_grad=False)
                self.matrix_A = self.matrix_A + (matrix_A_inv,)
                self.matrix_G = self.matrix_G + (matrix_G_inv,)
                self.matrix_A_shape = self.matrix_A_shape + ((vocab_size,),)

            if layer_type in [Conv, FC, Embedding] and "bias" not in self.params[idx].name.lower():
                self.weight_fim_idx_map = self.weight_fim_idx_map + (self.thor_layer_count,)
                self.weight_layerType_idx_map = self.weight_layerType_idx_map + (layer_type,)
                self.thor_layer_count = self.thor_layer_count + 1
                if layer_type == Conv:
                    self.weight_conv_idx_map = self.weight_conv_idx_map + (self.conv_layer_count,)
                    self.conv_layer_count = self.conv_layer_count + 1
                else:
                    self.weight_conv_idx_map = self.weight_conv_idx_map + (-1,)
            else:
                self.weight_fim_idx_map = self.weight_fim_idx_map + (-1,)
                self.weight_conv_idx_map = self.weight_conv_idx_map + (-1,)
                if layer_type == LayerNorm:
                    self.weight_layerType_idx_map = self.weight_layerType_idx_map + (LayerNorm,)
                else:
                    self.weight_layerType_idx_map = self.weight_layerType_idx_map + (Other,)
                # bert.cls1.output_bias: not a network layer, only a trainable param
            if "output_bias" not in self.params[idx].name.lower():
                layer_counter = get_layer_counter(layer_type, layer_counter, self.params, idx)

        self.matrix_A = ParameterTuple(self.matrix_A)
        self.matrix_G = ParameterTuple(self.matrix_G)
        self.weight_decay = weight_decay
        self.decay_flags = tuple(decay_filter(x) for x in self.parameters)
        self.update_gradient = P.UpdateThorGradient(split_dim=split_dim)

        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        self.is_distributed = (self.parallel_mode != ParallelMode.STAND_ALONE)
        if self.is_distributed:
            mean = _get_gradients_mean()
            degree = _get_device_num()
            if self.conv_layer_count > 0:
                if not split_indices:
                    self.split_indices = [len(self.matrix_A) - 1]
                else:
                    self.split_indices = split_indices
                auto_parallel_context().set_all_reduce_fusion_split_indices(self.split_indices, "hccl_world_groupsum2")
                auto_parallel_context().set_all_reduce_fusion_split_indices(self.split_indices, "hccl_world_groupsum4")
                auto_parallel_context().set_all_reduce_fusion_split_indices(self.split_indices, "hccl_world_groupsum6")
                auto_parallel_context().set_all_reduce_fusion_split_indices(self.split_indices, "hccl_world_groupsum8")
                self.grad_reducer_Amax = DistributedGradReducer(self.matrix_A, mean, degree, fusion_type=2)
                self.grad_reducer_Gmax = DistributedGradReducer(self.matrix_A, mean, degree, fusion_type=4)
                self.grad_reducer_A = DistributedGradReducer(self.matrix_A, mean, degree, fusion_type=6)
                self.grad_reducer_G = DistributedGradReducer(self.matrix_A, mean, degree, fusion_type=8)
            else:
                if not split_indices:
                    self.split_indices = [len(self.params) - 1]
                else:
                    self.split_indices = split_indices
                auto_parallel_context().set_all_reduce_fusion_split_indices(self.split_indices, "hccl_world_groupsum3")
                self.grad_reducer_g = DistributedGradReducer(self.params, mean, degree, fusion_type=3)

    def _get_Ainv_Ginv_list(self, gradients, damping_step, matrix_a_allreduce, matrix_g_allreduce):
        """get matrixA inverse list and matrix G inverse list"""
        for i in range(len(self.params)):
            thor_layer_count = self.weight_fim_idx_map[i]
            conv_layer_count = self.weight_conv_idx_map[i]
            layer_type = self.weight_layerType_idx_map[i]
            if layer_type in [Conv, FC, Embedding]:
                g = gradients[i]
                matrix_A = self.matrix_A_cov[thor_layer_count]
                matrix_G = self.matrix_G_cov[thor_layer_count]
                matrix_A = F.depend(matrix_A, g)
                matrix_G = F.depend(matrix_G, g)
                dampingA = damping_step
                dampingG = damping_step
                feature_map = self.feature_map
                if layer_type == Conv:
                    A_normalizer = self.A_normalizer[conv_layer_count]
                    G_normalizer = self.G_normalizer[conv_layer_count]
                    A_normalizer = F.depend(A_normalizer, g)
                    G_normalizer = F.depend(G_normalizer, g)
                    dampingA = self.mul(damping_step, 1.0 / A_normalizer)
                    dampingG = self.mul(damping_step, 1.0 / G_normalizer)
                    feature_map = self.sqrt(1.0 / A_normalizer)
                A_shape = self.shape(matrix_A)
                A_eye = self.eye(A_shape[0], A_shape[0], mstype.float32)
                dampingA = self.sqrt(dampingA)
                dampingG = self.sqrt(dampingG)
                G_shape = self.shape(matrix_G)
                G_eye = self.eye(G_shape[0], G_shape[1], mstype.float32)
                matrix_G = self.mul(matrix_G, self.loss_scale)
                matrix_G = self.mul(matrix_G, self.batch_size_scale)
                matrix_G = matrix_G + dampingG * G_eye
                if layer_type == Embedding:
                    A_eye = P.OnesLike()(matrix_A)
                    matrix_A = self.mul(matrix_A, 1.0 / self.batch_size)
                    matrix_A = matrix_A + dampingA * A_eye
                    matrix_A = self.inv(matrix_A)
                    matrix_G = self.embedding_cholesky(matrix_G)
                    matrix_G = self.matmul(matrix_G, matrix_G)
                else:
                    matrix_A = matrix_A + dampingA * A_eye
                    matrix_A = self.cholesky(matrix_A)
                    matrix_A = self.vector_matmul(matrix_A, matrix_A)
                    matrix_A = P.BroadcastTo(self.matrix_A_shape[thor_layer_count])(matrix_A)
                    matrix_G = self.cholesky(matrix_G)
                    matrix_G = self.vector_matmul(matrix_G, matrix_G)
                matrix_A = self.mul(matrix_A, feature_map)
                matrix_G = self.mul(matrix_G, feature_map)
                matrix_a_allreduce = matrix_a_allreduce + (matrix_A,)
                matrix_g_allreduce = matrix_g_allreduce + (matrix_G,)
        return matrix_a_allreduce, matrix_g_allreduce

    def construct(self, gradients):
        params = self.params
        moments = self.moments
        gradients = self.scale_grad(gradients)
        damping_step = self.gather(self.damping, self.cov_step, self.axis)
        damping_step = self.cast(damping_step, mstype.float32)
        new_grads = ()
        if self.thor:
            matrix_Ainv_list = ()
            matrix_Ginv_list = ()
            matrix_A_allreduce, matrix_G_allreduce = self._get_Ainv_Ginv_list(gradients, damping_step,
                                                                              matrix_Ainv_list, matrix_Ginv_list)
            if self.is_distributed and self.conv_layer_count > 0:
                matrix_A_allreduce = self.grad_reducer_A(matrix_A_allreduce)
                matrix_G_allreduce = self.grad_reducer_G(matrix_G_allreduce)

            for i in range(len(self.params)):
                g = gradients[i]
                thor_layer_count = self.weight_fim_idx_map[i]
                conv_layer_count = self.weight_conv_idx_map[i]
                layer_type = self.weight_layerType_idx_map[i]
                if layer_type in [Conv, FC]:
                    g_shape = self.shape(g)
                    g = self.reshape(g, (g_shape[0], -1))
                    matrix_A = matrix_A_allreduce[thor_layer_count]
                    matrix_G = matrix_G_allreduce[thor_layer_count]
                    g = self.update_gradient(matrix_G, g, matrix_A)
                    fake_A = self.assign(self.matrix_A[thor_layer_count], matrix_A)
                    fake_G = self.assign(self.matrix_G[thor_layer_count], matrix_G)
                    g = F.depend(g, fake_A)
                    g = F.depend(g, fake_G)
                    if conv_layer_count != -1:
                        g = self.reshape(g, g_shape)
                elif layer_type == Embedding:
                    matrix_A = matrix_A_allreduce[thor_layer_count]
                    matrix_G = matrix_G_allreduce[thor_layer_count]
                    fake_A = self.assign(self.matrix_A[thor_layer_count], matrix_A)
                    fake_G = self.assign(self.matrix_G[thor_layer_count], matrix_G)
                    g = F.depend(g, fake_A)
                    g = F.depend(g, fake_G)
                    temp_a = self.expand(matrix_A, 1)
                    g = self.mul(temp_a, g)
                    g = self.matmul(g, matrix_G)
                elif layer_type == LayerNorm:
                    damping = self.sqrt(damping_step)
                    normalizer = self.batch_size
                    normalizer = self.cast(normalizer, mstype.float32)
                    fim_cov = self.square(g)
                    fim_cov = self.mul(fim_cov, 1.0 / normalizer)
                    fim_cov = fim_cov + damping
                    fim_inv = self.inv(fim_cov)
                    g = self.mul(fim_inv, g)
                new_grads = new_grads + (g,)
        else:
            for j in range(len(self.params)):
                g = gradients[j]
                thor_layer_count = self.weight_fim_idx_map[j]
                conv_layer_count = self.weight_conv_idx_map[j]
                layer_type = self.weight_layerType_idx_map[j]
                if layer_type in [Conv, FC]:
                    g_shape = self.shape(g)
                    g = self.reshape(g, (g_shape[0], -1))
                    matrix_A = self.matrix_A[thor_layer_count]
                    matrix_G = self.matrix_G[thor_layer_count]
                    g = self.update_gradient(matrix_G, g, matrix_A)
                    if conv_layer_count != -1:
                        g = self.reshape(g, g_shape)
                elif layer_type == Embedding:
                    matrix_A = self.matrix_A[thor_layer_count]
                    matrix_G = self.matrix_G[thor_layer_count]
                    g = gradients[j]
                    temp_a = self.expand(matrix_A, 1)
                    g = self.mul(temp_a, g)
                    g = self.matmul(g, matrix_G)
                elif layer_type == LayerNorm:
                    damping = self.sqrt(damping_step)
                    normalizer = self.batch_size
                    normalizer = self.cast(normalizer, mstype.float32)
                    fim_cov = self.square(g)
                    fim_cov = self.mul(fim_cov, 1.0 / normalizer)
                    fim_cov = fim_cov + damping
                    fim_inv = self.inv(fim_cov)
                    g = self.mul(fim_inv, g)
                new_grads = new_grads + (g,)
        gradients = new_grads

        if self.is_distributed and self.conv_layer_count == 0:
            gradients = self.grad_reducer_g(gradients)
        self.cov_step = self.cov_step + self.one
        if self.weight_decay > 0:
            gradients = self.hyper_map(F.partial(apply_decay, self.weight_decay), self.decay_flags, params, gradients)
        lr = self.get_lr()
        success = self.hyper_map(F.partial(_momentum_opt, self.opt, self.momentum, lr), gradients, params, moments)
        return success


class THOR_Ascend(Optimizer):
    """THOR"""

    def __init__(self, net, learning_rate, damping, momentum, weight_decay=0.0, loss_scale=1.0, batch_size=32,
                 decay_filter=lambda x: x.name not in [], split_indices=None):
        params = filter(lambda x: x.requires_grad, net.get_parameters())
        super(THOR_Ascend, self).__init__(learning_rate, params, weight_decay, loss_scale)
        if isinstance(momentum, float) and momentum < 0.0:
            raise ValueError("momentum should be at least 0.0, but got momentum {}".format(momentum))
        self.momentum = Parameter(Tensor(momentum, mstype.float32), name="momentum")
        self.params = self.parameters
        self.moments = self.params.clone(prefix="moments", init='zeros')
        self.hyper_map = C.HyperMap()
        self.opt = P.ApplyMomentum()
        self.net = net
        self.matrix_A_cov = ParameterTuple(filter(lambda x: 'matrix_A' in x.name, net.get_parameters()))
        self.matrix_G_cov = ParameterTuple(filter(lambda x: 'matrix_G' in x.name, net.get_parameters()))
        self.A_normalizer = ParameterTuple(filter(lambda x: 'A_normalizer' in x.name, net.get_parameters()))
        self.G_normalizer = ParameterTuple(filter(lambda x: 'G_normalizer' in x.name, net.get_parameters()))
        self.cube_matmul_left = P.CusMatMulCubeFraczLeftCast()
        self.cube_matmul_left_fc = P.CusMatMulCubeDenseLeft()
        self.cube_matmul_right_fc = P.CusMatMulCubeDenseRight()
        self.cube_matmul_right_mul = P.CusMatMulCubeFraczRightMul()
        self.transpose = P.Transpose()
        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.mul = P.Mul()

        self.C0 = 16
        self.matrix_A_dim = ()
        self.padA_flag = ()
        self.device_shape_pad_flag = ()
        self.diag_block_dim = 128
        self.matrix_A = ()
        self.matrix_G = ()
        print("matrix_A_cov len is", len(self.matrix_A_cov))
        self.thor_layer_count = 0
        self.conv_layer_count = 0
        self.weight_fim_idx_map = ()
        self.weight_conv_idx_map = ()
        self.weight_layerType_idx_map = ()
        layer_type_map = get_net_layertype_mask(net)
        layer_counter = 0
        for idx in range(len(self.params)):
            layer_type = layer_type_map[layer_counter]
            weight = self.params[idx]
            weight_shape = self.shape(weight)
            if layer_type == Conv and "bias" not in self.params[idx].name.lower():
                in_channels = weight_shape[1]
                out_channels = weight_shape[0]
                matrix_A_dim = in_channels * weight_shape[2] * weight_shape[3]
                matrix_G_dim = out_channels
                matrix_A_device_shape, matrix_A_device_dim = caculate_device_shape(matrix_A_dim, in_channels, True)
                matrix_G_device_shape, matrix_G_device_dim = caculate_device_shape(matrix_G_dim, in_channels, False)
                matrix_A_inv = Parameter(
                    Tensor(np.reshape(np.identity(matrix_A_device_dim).astype(np.float16), matrix_A_device_shape)),
                    name='matrix_A_inv_' + str(self.thor_layer_count), requires_grad=False)
                matrix_G_inv = Parameter(
                    Tensor(np.reshape(np.identity(matrix_G_device_dim).astype(np.float16), matrix_G_device_shape)),
                    name="matrix_G_inv_" + str(self.thor_layer_count), requires_grad=False)
                self.matrix_A = self.matrix_A + (matrix_A_inv,)
                self.matrix_G = self.matrix_G + (matrix_G_inv,)
                self.matrix_A_dim = self.matrix_A_dim + (matrix_A_dim,)
                padA_flag = False
                if (matrix_A_dim // self.diag_block_dim) * self.diag_block_dim != matrix_A_dim \
                    and matrix_A_dim > self.diag_block_dim:
                    padA_flag = True
                self.padA_flag = self.padA_flag + (padA_flag,)
                device_shape_pad_flag = False
                if matrix_A_dim != matrix_A_device_dim:
                    device_shape_pad_flag = True
                self.device_shape_pad_flag = self.device_shape_pad_flag + (device_shape_pad_flag,)
            elif layer_type == FC and "bias" not in self.params[idx].name.lower():
                out_channels = weight_shape[0]
                if out_channels == 1001:
                    fc_matrix_A = Parameter(Tensor(np.zeros([128, 128, 16, 16]).astype(np.float16)),
                                            name='matrix_A_inv_' + str(self.thor_layer_count),
                                            requires_grad=False)
                    fc_matrix_G = Parameter(Tensor(np.zeros([63, 63, 16, 16]).astype(np.float16)),
                                            name="matrix_G_inv_" + str(self.thor_layer_count),
                                            requires_grad=False)
                    self.matrix_A = self.matrix_A + (fc_matrix_A,)
                    self.matrix_G = self.matrix_G + (fc_matrix_G,)

            if layer_type in [Conv, FC, Embedding] and "bias" not in self.params[idx].name.lower():
                self.weight_fim_idx_map = self.weight_fim_idx_map + (self.thor_layer_count,)
                self.weight_layerType_idx_map = self.weight_layerType_idx_map + (layer_type,)
                self.thor_layer_count = self.thor_layer_count + 1
                if layer_type == Conv:
                    self.weight_conv_idx_map = self.weight_conv_idx_map + (self.conv_layer_count,)
                    self.conv_layer_count = self.conv_layer_count + 1
                else:
                    self.weight_conv_idx_map = self.weight_conv_idx_map + (-1,)
            else:
                self.weight_fim_idx_map = self.weight_fim_idx_map + (-1,)
                self.weight_conv_idx_map = self.weight_conv_idx_map + (-1,)
                if layer_type == LayerNorm:
                    self.weight_layerType_idx_map = self.weight_layerType_idx_map + (LayerNorm,)
                else:
                    self.weight_layerType_idx_map = self.weight_layerType_idx_map + (Other,)
            # bert.cls1.output_bias: not a network layer, only a trainable param
            if "output_bias" not in self.params[idx].name.lower():
                layer_counter = get_layer_counter(layer_type, layer_counter, self.params, idx)

        self.matrix_A = ParameterTuple(self.matrix_A)
        self.matrix_G = ParameterTuple(self.matrix_G)
        self.matrix_max_inv = ()
        for i in range(len(self.matrix_A)):
            self.matrix_max_inv = self.matrix_max_inv + (
                Parameter(initializer(1, [1], mstype.float32), name="matrix_max" + str(i), requires_grad=False),)
        self.log = P.Log()
        self.exp = P.Exp()
        self.sqrt = P.Sqrt()
        self.matrix_max_inv = ParameterTuple(self.matrix_max_inv)
        self.assign = P.Assign()
        self.cast = P.Cast()
        self.thor = True
        self.weight_decay = weight_decay * loss_scale
        self.decay_flags = tuple(decay_filter(x) for x in self.parameters)
        self.damping = damping
        self.gather = P.GatherV2()
        self.one = Tensor(1, mstype.int32)
        self.batch_size = Tensor(batch_size, mstype.float32)
        self.loss_scale = Tensor(1 / (loss_scale * loss_scale), mstype.float32)
        self.batch_size_scale = Tensor(batch_size * batch_size, mstype.float32)
        self.axis = 0
        self.cov_step = Parameter(initializer(0, [1], mstype.int32), name="cov_step", requires_grad=False)
        self.cast = P.Cast()
        self.eye = P.Eye()
        self.cholesky = P.CusCholeskyTrsm()
        self.vector_matmul = P.CusBatchMatMul()
        self.fused_abs_max2 = P.CusFusedAbsMax1()
        self.matrix_combine = P.CusMatrixCombine()
        self.slice = P.Slice()
        self.expand = P.ExpandDims()
        self.reduce_sum = P.ReduceSum(keep_dims=False)
        self.square = P.Square()
        self.inv = P.Inv()
        self.matmul = P.MatMul()

        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        self.is_distributed = (self.parallel_mode != ParallelMode.STAND_ALONE)
        if self.is_distributed:
            mean = _get_gradients_mean()
            degree = _get_device_num()
            if self.conv_layer_count > 0:
                if not split_indices:
                    self.split_indices = [len(self.matrix_A) - 1]
                else:
                    self.split_indices = split_indices
                auto_parallel_context().set_all_reduce_fusion_split_indices(self.split_indices, "hccl_world_groupsum2")
                auto_parallel_context().set_all_reduce_fusion_split_indices(self.split_indices, "hccl_world_groupsum4")
                auto_parallel_context().set_all_reduce_fusion_split_indices(self.split_indices, "hccl_world_groupsum6")
                auto_parallel_context().set_all_reduce_fusion_split_indices(self.split_indices, "hccl_world_groupsum8")
                self.grad_reducer_Amax = DistributedGradReducer(self.matrix_A, mean, degree, fusion_type=2)
                self.grad_reducer_Gmax = DistributedGradReducer(self.matrix_A, mean, degree, fusion_type=4)
                self.grad_reducer_A = DistributedGradReducer(self.matrix_A, mean, degree, fusion_type=6)
                self.grad_reducer_G = DistributedGradReducer(self.matrix_A, mean, degree, fusion_type=8)
            else:
                if not split_indices:
                    self.split_indices = [len(self.params) - 1]
                else:
                    self.split_indices = split_indices
                auto_parallel_context().set_all_reduce_fusion_split_indices(self.split_indices, "hccl_world_groupsum3")
                self.grad_reducer_g = DistributedGradReducer(self.params, mean, degree, fusion_type=3)

    def _get_Ainv_Ginv_Amax_Gmax_list(self, gradients, damping_step, matrix_a_allreduce, matrix_g_allreduce,
                                      matrix_a_max_allreduce, matrix_g_max_allreduce):
        """get matrixA inverse list, matrixG inverse list, matrixA_max list, matrixG_max list"""
        for i in range(len(self.params)):
            thor_layer_count = self.weight_fim_idx_map[i]
            conv_layer_count = self.weight_conv_idx_map[i]
            layer_type = self.weight_layerType_idx_map[i]
            if layer_type in [Conv, FC, Embedding]:
                g = gradients[i]
                matrix_A = self.matrix_A_cov[thor_layer_count]
                matrix_G = self.matrix_G_cov[thor_layer_count]
                matrix_A = F.depend(matrix_A, g)
                matrix_G = F.depend(matrix_G, g)
                A_shape = self.shape(matrix_A)
                A_eye = self.eye(A_shape[0], A_shape[0], mstype.float32)
                G_shape = self.shape(matrix_G)
                G_eye = self.eye(G_shape[0], G_shape[0], mstype.float32)
                if layer_type == Conv:
                    A_normalizer = self.A_normalizer[conv_layer_count]
                    G_normalizer = self.G_normalizer[conv_layer_count]
                    A_normalizer = F.depend(A_normalizer, g)
                    G_normalizer = F.depend(G_normalizer, g)
                    dampingA = self.mul(damping_step, self.batch_size / A_normalizer)
                    dampingG = self.mul(damping_step, self.batch_size / G_normalizer)
                    dampingA = self.sqrt(dampingA)
                    matrix_A = matrix_A + dampingA * A_eye
                    matrix_A_inv = self.cholesky(matrix_A)
                    matrix_A_inv = self.vector_matmul(matrix_A_inv, matrix_A_inv)
                    A_max = P.CusFusedAbsMax1([self.matrix_A_dim[conv_layer_count],
                                               self.matrix_A_dim[conv_layer_count]])(matrix_A_inv)
                    A_max = self.fused_abs_max2(A_max)
                    matrix_A_inv = self.matrix_combine(matrix_A_inv)
                    if self.padA_flag[conv_layer_count]:
                        matrix_A_inv = self.slice(matrix_A_inv, (0, 0), (self.matrix_A_dim[conv_layer_count],
                                                                         self.matrix_A_dim[conv_layer_count]))
                    if self.device_shape_pad_flag[conv_layer_count]:
                        weight = self.params[i]
                        weight_shape = self.shape(weight)
                        kernel_hw = weight_shape[2] * weight_shape[3]
                        in_channels = weight_shape[1]
                        matrix_A_inv = self.reshape(matrix_A_inv, (kernel_hw, in_channels, kernel_hw, in_channels))
                        matrix_A_inv = P.Pad(((0, 0), (0, self.C0 - in_channels), (0, 0),
                                              (0, self.C0 - in_channels)))(matrix_A_inv)
                    matrix_A_inv_shape = self.shape(self.matrix_A[thor_layer_count])
                    matrix_A_device_temp_shape = (matrix_A_inv_shape[0], matrix_A_inv_shape[2],
                                                  matrix_A_inv_shape[1], matrix_A_inv_shape[3])
                    matrix_A_inv = self.reshape(matrix_A_inv, matrix_A_device_temp_shape)
                    matrix_A_inv = self.transpose(matrix_A_inv, (2, 0, 1, 3))

                    dampingG = self.sqrt(dampingG)
                    matrix_G = self.mul(matrix_G, self.loss_scale)
                    matrix_G = self.mul(matrix_G, self.batch_size_scale)
                    matrix_G = matrix_G + dampingG * G_eye
                    matrix_G_inv = self.cholesky(matrix_G)
                    matrix_G_inv = self.vector_matmul(matrix_G_inv, matrix_G_inv)
                    G_max = self.fused_abs_max2(matrix_G_inv)
                    G_max = self.fused_abs_max2(G_max)
                    matrix_G_inv = self.matrix_combine(matrix_G_inv)
                    matrix_G_inv_shape = self.shape(self.matrix_G[thor_layer_count])
                    matrix_G_device_temp_shape = (matrix_G_inv_shape[0], matrix_G_inv_shape[2],
                                                  matrix_G_inv_shape[1], matrix_G_inv_shape[3])
                    matrix_G_inv = self.reshape(matrix_G_inv, matrix_G_device_temp_shape)
                    matrix_G_inv = self.transpose(matrix_G_inv, (2, 0, 1, 3))

                    A_max = F.depend(A_max, g)
                    G_max = F.depend(G_max, g)
                    matrix_a_allreduce = matrix_a_allreduce + (matrix_A_inv,)
                    matrix_g_allreduce = matrix_g_allreduce + (matrix_G_inv,)
                    matrix_a_max_allreduce = matrix_a_max_allreduce + (A_max,)
                    matrix_g_max_allreduce = matrix_g_max_allreduce + (G_max,)
                elif layer_type == FC:
                    damping = self.sqrt(damping_step)
                    matrix_A = matrix_A + damping * A_eye
                    matrix_A_inv = self.cholesky(matrix_A)
                    matrix_A_inv = self.vector_matmul(matrix_A_inv, matrix_A_inv)
                    weight_shape = self.shape(self.params[i])
                    out_channels = weight_shape[0]
                    if out_channels == 2:
                        matrix_A_inv = self.matrix_combine(matrix_A_inv)
                        matrix_G_inv = G_eye
                    else:
                        matrix_G = self.mul(matrix_G, self.loss_scale)
                        matrix_G = self.mul(matrix_G, self.batch_size_scale)
                        matrix_G = matrix_G + damping * G_eye
                        matrix_G_inv = self.cholesky(matrix_G)
                        matrix_G_inv = self.vector_matmul(matrix_G_inv, matrix_G_inv)
                        if out_channels == 1001:
                            matrix_A_inv_max = self.fused_abs_max2(matrix_A_inv)
                            A_max = self.fused_abs_max2(matrix_A_inv_max)
                            matrix_A_inv = self.matrix_combine(matrix_A_inv)
                            matrix_A_inv_shape = self.shape(matrix_A_inv)
                            matrix_A_inv = self.reshape(matrix_A_inv,
                                                        (matrix_A_inv_shape[0] / 16, 16,
                                                         matrix_A_inv_shape[0] / 16, 16))
                            matrix_A_inv = self.transpose(matrix_A_inv, (2, 0, 1, 3))
                            matrix_G_inv_max = P.CusFusedAbsMax1([1001, 1001])(matrix_G_inv)
                            G_max = self.fused_abs_max2(matrix_G_inv_max)
                            matrix_G_inv = self.matrix_combine(matrix_G_inv)
                            matrix_G_inv = self.slice(matrix_G_inv, (0, 0), (1001, 1001))
                            matrix_G_inv = P.Pad(((0, 7), (0, 7)))(matrix_G_inv)
                            matrix_G_inv_shape = self.shape(matrix_G_inv)
                            matrix_G_inv = self.reshape(matrix_G_inv,
                                                        (matrix_G_inv_shape[0] / 16, 16,
                                                         matrix_G_inv_shape[0] / 16, 16))
                            matrix_G_inv = self.transpose(matrix_G_inv, (2, 0, 1, 3))
                            A_max = F.depend(A_max, g)
                            G_max = F.depend(G_max, g)
                            matrix_a_max_allreduce = matrix_a_max_allreduce + (A_max,)
                            matrix_g_max_allreduce = matrix_g_max_allreduce + (G_max,)
                        else:
                            matrix_A_inv = self.matrix_combine(matrix_A_inv)
                            matrix_G_inv = self.matrix_combine(matrix_G_inv)
                    matrix_a_allreduce = matrix_a_allreduce + (matrix_A_inv,)
                    matrix_g_allreduce = matrix_g_allreduce + (matrix_G_inv,)
                elif layer_type == Embedding:
                    damping = self.sqrt(damping_step)
                    A_eye = P.OnesLike()(matrix_A)
                    matrix_A = self.mul(matrix_A, 1.0 / self.batch_size)
                    matrix_A = matrix_A + damping * A_eye
                    matrix_A_inv = self.inv(matrix_A)
                    matrix_G = self.mul(matrix_G, self.loss_scale)
                    matrix_G = self.mul(matrix_G, self.batch_size_scale)
                    matrix_G = matrix_G + damping * G_eye
                    matrix_G_inv = self.cholesky(matrix_G)
                    matrix_G_inv = self.vector_matmul(matrix_G_inv, matrix_G_inv)
                    matrix_G_inv = self.matrix_combine(matrix_G_inv)
                    matrix_a_allreduce = matrix_a_allreduce + (matrix_A_inv,)
                    matrix_g_allreduce = matrix_g_allreduce + (matrix_G_inv,)
        return matrix_a_allreduce, matrix_g_allreduce, matrix_a_max_allreduce, matrix_g_max_allreduce

    def _process_layernorm(self, damping_step, gradient):
        """process layernorm layer for thor"""
        damping = self.sqrt(damping_step)
        normalizer = self.cast(self.batch_size, mstype.float32)
        fim_cov = self.square(gradient)
        fim_cov = self.mul(fim_cov, 1.0 / normalizer)
        fim_cov = fim_cov + damping
        fim_inv = self.inv(fim_cov)
        gradient = self.mul(fim_inv, gradient)
        return gradient

    def _get_second_gradients(self, new_grads, damping_step, gradients):
        """get second gradients for thor"""
        params_len = len(self.params)
        for i in range(params_len):
            g = gradients[i]
            thor_layer_count = self.weight_fim_idx_map[i]
            layer_type = self.weight_layerType_idx_map[i]
            if self.conv_layer_count > 0:
                matrix_A = self.matrix_A[thor_layer_count]
                matrix_G = self.matrix_G[thor_layer_count]
                matrix_max = self.matrix_max_inv[thor_layer_count]
                if layer_type == FC:
                    g = self.cube_matmul_left_fc(matrix_G, g)
                    g = self.cube_matmul_right_fc(g, matrix_A, matrix_max)
                elif layer_type == Conv:
                    g = self.cube_matmul_left(matrix_G, g)
                    g = self.cube_matmul_right_mul(g, matrix_A, matrix_max)
            else:
                if layer_type == Embedding:
                    temp_a_ori = self.matrix_A_cov[thor_layer_count]
                    temp_g = self.matrix_G_cov[thor_layer_count]
                    temp_a = self.expand(temp_a_ori, 1)
                    g = self.mul(temp_a, g)
                    temp_g = self.cast(temp_g, mstype.float16)
                    g = self.cast(g, mstype.float16)
                    g = self.matmul(g, temp_g)
                    g = self.cast(g, mstype.float32)
                elif layer_type == FC:
                    temp_a = self.matrix_A_cov[thor_layer_count]
                    temp_g = self.matrix_G_cov[thor_layer_count]
                    temp_a = self.cast(temp_a, mstype.float16)
                    temp_g = self.cast(temp_g, mstype.float16)
                    g = self.cast(g, mstype.float16)
                    g = self.matmul(temp_g, g)
                    g = self.matmul(g, temp_a)
                    g = self.cast(g, mstype.float32)
                elif layer_type == LayerNorm:
                    g = self._process_layernorm(damping_step, g)
            new_grads = new_grads + (g,)
        return new_grads

    def construct(self, gradients):
        params = self.params
        moments = self.moments
        damping_step = self.gather(self.damping, self.cov_step, self.axis)
        damping_step = self.cast(damping_step, mstype.float32)
        if self.thor:
            matrix_A_allreduce = ()
            matrix_G_allreduce = ()
            matrix_A_max_allreduce = ()
            matrix_G_max_allreduce = ()
            matrix_A_allreduce, matrix_G_allreduce, matrix_A_max_allreduce, matrix_G_max_allreduce = \
                self._get_Ainv_Ginv_Amax_Gmax_list(gradients, damping_step, matrix_A_allreduce, matrix_G_allreduce,
                                                   matrix_A_max_allreduce, matrix_G_max_allreduce)
            if self.is_distributed and self.conv_layer_count > 0:
                matrix_A_allreduce = self.grad_reducer_A(matrix_A_allreduce)
                matrix_G_allreduce = self.grad_reducer_G(matrix_G_allreduce)
                matrix_A_max_allreduce = self.grad_reducer_Amax(matrix_A_max_allreduce)
                matrix_G_max_allreduce = self.grad_reducer_Gmax(matrix_G_max_allreduce)

            new_grads = ()
            for i in range(len(self.params)):
                g = gradients[i]
                thor_layer_count = self.weight_fim_idx_map[i]
                conv_layer_count = self.weight_conv_idx_map[i]
                layer_type = self.weight_layerType_idx_map[i]
                if self.conv_layer_count > 0:
                    temp_a = matrix_A_allreduce[thor_layer_count]
                    temp_g = matrix_G_allreduce[thor_layer_count]
                    matrix_A_inv_max = self.log(matrix_A_max_allreduce[thor_layer_count])
                    matrix_A_inv_max = self.mul(matrix_A_inv_max, -1)
                    matrix_A_inv_max = self.exp(matrix_A_inv_max)
                    temp_a = self.mul(temp_a, matrix_A_inv_max)
                    matrix_G_inv_max = self.log(matrix_G_max_allreduce[thor_layer_count])
                    matrix_G_inv_max = self.mul(matrix_G_inv_max, -1)
                    matrix_G_inv_max = self.exp(matrix_G_inv_max)
                    temp_g = self.mul(temp_g, matrix_G_inv_max)
                    temp_max = self.mul(matrix_A_max_allreduce[thor_layer_count],
                                        matrix_G_max_allreduce[thor_layer_count])
                    temp_a = self.cast(temp_a, mstype.float16)
                    temp_g = self.cast(temp_g, mstype.float16)
                    if layer_type == FC:
                        g = self.cube_matmul_left_fc(temp_g, g)
                        g = self.cube_matmul_right_fc(g, temp_a, temp_max)
                    elif layer_type == Conv:
                        A_normalizer = self.A_normalizer[conv_layer_count]
                        A_normalizer = F.depend(A_normalizer, g)
                        temp_max = self.mul(temp_max, self.batch_size / A_normalizer)
                        g = self.cube_matmul_left(temp_g, g)
                        g = self.cube_matmul_right_mul(g, temp_a, temp_max)
                    fake_A = self.assign(self.matrix_A[thor_layer_count], temp_a)
                    fake_G = self.assign(self.matrix_G[thor_layer_count], temp_g)
                    fake_max = self.assign(self.matrix_max_inv[thor_layer_count], temp_max)
                    g = F.depend(g, fake_A)
                    g = F.depend(g, fake_G)
                    g = F.depend(g, fake_max)
                else:
                    if layer_type == Embedding:
                        temp_a_ori = matrix_A_allreduce[thor_layer_count]
                        temp_g = matrix_G_allreduce[thor_layer_count]
                        fake_A = self.assign(self.matrix_A_cov[thor_layer_count], temp_a_ori)
                        fake_G = self.assign(self.matrix_G_cov[thor_layer_count], temp_g)
                        g = F.depend(g, fake_A)
                        g = F.depend(g, fake_G)
                        temp_a = self.expand(temp_a_ori, 1)
                        g = self.mul(temp_a, g)
                        temp_g = self.cast(temp_g, mstype.float16)
                        g = self.cast(g, mstype.float16)
                        g = self.matmul(g, temp_g)
                        g = self.cast(g, mstype.float32)
                    elif layer_type == FC:
                        temp_a = matrix_A_allreduce[thor_layer_count]
                        temp_g = matrix_G_allreduce[thor_layer_count]
                        fake_A = self.assign(self.matrix_A_cov[thor_layer_count], temp_a)
                        fake_G = self.assign(self.matrix_G_cov[thor_layer_count], temp_g)
                        g = F.depend(g, fake_A)
                        g = F.depend(g, fake_G)
                        temp_a = self.cast(temp_a, mstype.float16)
                        temp_g = self.cast(temp_g, mstype.float16)
                        g = self.cast(g, mstype.float16)
                        g = self.matmul(temp_g, g)
                        g = self.matmul(g, temp_a)
                        g = self.cast(g, mstype.float32)
                    elif layer_type == LayerNorm:
                        g = self._process_layernorm(damping_step, g)
                new_grads = new_grads + (g,)
            gradients = new_grads
        else:
            new_grads = ()
            gradients = self._get_second_gradients(new_grads, damping_step, gradients)

        if self.is_distributed and self.conv_layer_count == 0:
            gradients = self.grad_reducer_g(gradients)
        self.cov_step = self.cov_step + self.one
        if self.weight_decay > 0:
            gradients = self.hyper_map(F.partial(apply_decay, self.weight_decay), self.decay_flags, params, gradients)
        gradients = self.scale_grad(gradients)
        lr = self.get_lr()
        success = self.hyper_map(F.partial(_momentum_opt, self.opt, self.momentum, lr), gradients, params, moments)
        return success
