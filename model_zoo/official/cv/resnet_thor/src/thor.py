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
"""THOR"""
from mindspore.ops import functional as F, composite as C, operations as P
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter, ParameterTuple
from mindspore.common.tensor import Tensor
import mindspore.common.dtype as mstype
from mindspore._checkparam import Validator
from mindspore.nn.optim.optimizer import Optimizer
from mindspore.parallel._utils import _get_device_num, _get_gradients_mean
from src.grad_reducer_thor import DistributedGradReducerThor

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


class THOR_GPU(Optimizer):
    """
    THOR
    """
    def __init__(self, params, learning_rate, momentum, matrix_A, matrix_G, A_inv_max, G_inv_max,
                 weight_decay=0.0, loss_scale=1.0, use_nesterov=False, decay_filter=lambda x: x.name not in []):
        super(THOR_GPU, self).__init__(learning_rate, params, weight_decay, loss_scale)
        Validator.check_value_type("momentum", momentum, [float], self.cls_name)
        if isinstance(momentum, float) and momentum < 0.0:
            raise ValueError("momentum should be at least 0.0, but got momentum {}".format(momentum))
        self.momentum = Parameter(Tensor(momentum, mstype.float32))
        self.params = self.parameters
        self.use_nesterov = Validator.check_bool(use_nesterov)
        self.moments = self.params.clone(prefix="moments", init='zeros')
        self.hyper_map = C.HyperMap()
        self.opt = P.ApplyMomentum(use_nesterov=self.use_nesterov)

        self.feature_map = [1.0 / 12544, 1.0 / 3136, 1.0 / 3136, 1.0 / 3136, 1.0 / 3136, 1.0 / 3136, 1.0 / 3136,
                            1.0 / 3136, 1.0 / 3136, 1.0 / 3136, 1.0 / 3136, 1.0 / 3136,
                            1.0 / 784, 1.0 / 784, 1.0 / 784, 1.0 / 784, 1.0 / 784, 1.0 / 784, 1.0 / 784, 1.0 / 784,
                            1.0 / 784, 1.0 / 784, 1.0 / 784, 1.0 / 784, 1.0 / 784,
                            1.0 / 196, 1.0 / 196, 1.0 / 196, 1.0 / 196, 1.0 / 196, 1.0 / 196, 1.0 / 196, 1.0 / 196,
                            1.0 / 196, 1.0 / 196, 1.0 / 196, 1.0 / 196, 1.0 / 196, 1.0 / 196, 1.0 / 196, 1.0 / 196,
                            1.0 / 196, 1.0 / 196, 1.0 / 196,
                            1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49,
                            1.0]
        self.feature_map_new = [x ** 0.5 for x in self.feature_map]
        self.transpose = P.Transpose()
        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.matmul = P.MatMul()
        self.matrix_A = ParameterTuple(matrix_A)
        self.matrix_G = ParameterTuple(matrix_G)
        self.A_inv_max = ParameterTuple(A_inv_max)
        self.G_inv_max = ParameterTuple(G_inv_max)
        self.assign = P.Assign()
        self.mul = P.Mul()

        mean = _get_gradients_mean()
        degree = _get_device_num()

        parameter_length = len(self.feature_map)
        self.grad_reducer_thorA = DistributedGradReducerThor(parameter_length, ((parameter_length,), 0), mean, degree)
        self.grad_reducer_thorG = DistributedGradReducerThor(parameter_length, ((parameter_length,), 0), mean, degree)
        self.weight_decay = weight_decay
        self.decay_flags = tuple(decay_filter(x) for x in self.parameters)
        self.update_gradient = P.UpdateThorGradient(split_dim=128)

    def construct(self, gradients):
        params = self.params
        moments = self.moments
        gradients = self.scale_grad(gradients)
        new_grads = ()
        if self.thor:
            matrix_A_allreduce = ()
            matrix_G_allreduce = ()
            for i in range(54):
                g = gradients[i * 3]
                matrix_A = self.matrix_A[i]
                matrix_G = self.matrix_G[i]
                matrix_A = F.depend(matrix_A, g)
                matrix_G = F.depend(matrix_G, g)
                matrix_A = self.mul(matrix_A, self.feature_map_new[i])
                matrix_G = self.mul(matrix_G, self.feature_map_new[i])
                matrix_A_allreduce = matrix_A_allreduce + (matrix_A,)
                matrix_G_allreduce = matrix_G_allreduce + (matrix_G,)
            matrix_A_allreduce = self.grad_reducer_thorA(matrix_A_allreduce)
            matrix_G_allreduce = self.grad_reducer_thorG(matrix_G_allreduce)
            for i in range(54):
                g = gradients[i * 3]
                g_shape = self.shape(g)
                g = self.reshape(g, (g_shape[0], -1))
                matrix_A = matrix_A_allreduce[i]
                matrix_G = matrix_G_allreduce[i]
                g = self.update_gradient(matrix_G, g, matrix_A)
                fake_A = self.assign(self.matrix_A[i], matrix_A)
                fake_G = self.assign(self.matrix_G[i], matrix_G)
                g = F.depend(g, fake_A)
                g = F.depend(g, fake_G)
                if i == 53:
                    new_grads = new_grads + (g,)
                else:
                    g = self.reshape(g, g_shape)
                    new_grads = new_grads + (g, gradients[i * 3 + 1], gradients[i * 3 + 2])
        else:
            for i in range(54):
                g = gradients[i * 3]
                g_shape = self.shape(g)
                g = self.reshape(g, (g_shape[0], -1))
                matrix_A = self.matrix_A[i]
                matrix_G = self.matrix_G[i]
                g = self.update_gradient(matrix_G, g, matrix_A)
                if i == 53:
                    new_grads = new_grads + (g,)
                else:
                    g = self.reshape(g, g_shape)
                    new_grads = new_grads + (g, gradients[i * 3 + 1], gradients[i * 3 + 2])
        gradients = new_grads
        if self.weight_decay > 0:
            gradients = self.hyper_map(F.partial(apply_decay, self.weight_decay), self.decay_flags,
                                       params, gradients)
        lr = self.get_lr()
        success = self.hyper_map(F.partial(_momentum_opt, self.opt, self.momentum, lr), gradients, params, moments)
        return success

class THOR(Optimizer):
    """THOR"""
    def __init__(self, params, learning_rate, momentum, matrix_A, matrix_G, A_inv_max, G_inv_max, weight_decay=0.0,
                 loss_scale=1.0,
                 decay_filter=lambda x: x.name not in []):
        super(THOR, self).__init__(learning_rate, params, weight_decay, loss_scale)
        if isinstance(momentum, float) and momentum < 0.0:
            raise ValueError("momentum should be at least 0.0, but got momentum {}".format(momentum))
        self.momentum = Parameter(Tensor(momentum, mstype.float32))
        self.params = self.parameters
        self.moments = self.params.clone(prefix="moments", init='zeros')
        self.hyper_map = C.HyperMap()
        self.opt = P.ApplyMomentum()
        self.matrix_A = ParameterTuple(matrix_A)
        self.matrix_G = ParameterTuple(matrix_G)
        self.A_inv_max = ParameterTuple(A_inv_max)
        self.G_inv_max = ParameterTuple(G_inv_max)
        self.cube_matmul_left = P.CusMatMulCubeFraczLeftCast()
        self.cube_matmul_left_fc = P.CusMatMulCubeDenseLeft()
        self.cube_matmul_right_fc = P.CusMatMulCubeDenseRight()
        self.cube_matmul_right_mul = P.CusMatMulCubeFraczRightMul()
        self.transpose = P.Transpose()
        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.mul = P.Mul()
        self.weight_idx = []
        for i in range(len(self.params)):
            if "conv" in self.params[i].name or "end_point" in self.params[i].name:
                self.weight_idx.append(i)
        self.weight_idx.append(len(self.params))
        self.feature_map = [1.0 / 12544, 1.0 / 3136, 1.0 / 3136, 1.0 / 3136, 1.0 / 3136, 1.0 / 3136, 1.0 / 3136,
                            1.0 / 3136, 1.0 / 3136, 1.0 / 3136, 1.0 / 3136, 1.0 / 3136,
                            1.0 / 784, 1.0 / 784, 1.0 / 784, 1.0 / 784, 1.0 / 784, 1.0 / 784, 1.0 / 784, 1.0 / 784,
                            1.0 / 784, 1.0 / 784, 1.0 / 784, 1.0 / 784, 1.0 / 784,
                            1.0 / 196, 1.0 / 196, 1.0 / 196, 1.0 / 196, 1.0 / 196, 1.0 / 196, 1.0 / 196, 1.0 / 196,
                            1.0 / 196, 1.0 / 196, 1.0 / 196, 1.0 / 196, 1.0 / 196, 1.0 / 196, 1.0 / 196, 1.0 / 196,
                            1.0 / 196, 1.0 / 196, 1.0 / 196,
                            1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49, 1.0 / 49,
                            1.0]
        mean = _get_gradients_mean()
        degree = _get_device_num()
        parameter_length = len(self.feature_map)
        self.grad_reducer_Amax = DistributedGradReducerThor(parameter_length, ((27,), 2), mean, degree)
        self.grad_reducer_Gmax = DistributedGradReducerThor(parameter_length, ((27,), 4), mean, degree)
        self.grad_reducer_A = DistributedGradReducerThor(parameter_length, ((27,), 6), mean, degree)
        self.grad_reducer_G = DistributedGradReducerThor(parameter_length, ((27,), 8), mean, degree)
        self.matrix_A_inv = ()
        self.matrix_G_inv = ()
        self.matrix_max_inv = ()

        for i in range(54):
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

    def construct(self, gradients):
        params = self.params
        moments = self.moments
        if self.thor:
            matrix_A_allreduce = ()
            matrix_G_allreduce = ()
            matrix_A_max_allreduce = ()
            matrix_G_max_allreduce = ()
            for i in range(54):
                g = gradients[i * 3]
                matrix_A = self.matrix_A[i]
                matrix_G = self.matrix_G[i]
                A_max = self.A_inv_max[i]
                G_max = self.G_inv_max[i]
                matrix_A = F.depend(matrix_A, g)
                matrix_G = F.depend(matrix_G, g)
                A_max = F.depend(A_max, g)
                G_max = F.depend(G_max, g)
                matrix_A_allreduce = matrix_A_allreduce + (matrix_A,)
                matrix_G_allreduce = matrix_G_allreduce + (matrix_G,)
                matrix_A_max_allreduce = matrix_A_max_allreduce + (A_max,)
                matrix_G_max_allreduce = matrix_G_max_allreduce + (G_max,)
            matrix_A_allreduce = self.grad_reducer_A(matrix_A_allreduce)
            matrix_G_allreduce = self.grad_reducer_G(matrix_G_allreduce)
            matrix_A_max_allreduce = self.grad_reducer_Amax(matrix_A_max_allreduce)
            matrix_G_max_allreduce = self.grad_reducer_Gmax(matrix_G_max_allreduce)
            new_grads = ()
            for i in range(54):
                g = gradients[i * 3]
                temp_a = matrix_A_allreduce[i]
                temp_g = matrix_G_allreduce[i]
                temp_a = self.cast(temp_a, mstype.float32)
                temp_g = self.cast(temp_g, mstype.float32)
                matrix_A_inv_max = self.log(matrix_A_max_allreduce[i])
                matrix_A_inv_max = self.mul(matrix_A_inv_max, -1)
                matrix_A_inv_max = self.exp(matrix_A_inv_max)
                temp_a = self.mul(temp_a, matrix_A_inv_max)
                matrix_G_inv_max = self.log(matrix_G_max_allreduce[i])
                matrix_G_inv_max = self.mul(matrix_G_inv_max, -1)
                matrix_G_inv_max = self.exp(matrix_G_inv_max)
                temp_g = self.mul(temp_g, matrix_G_inv_max)
                temp_max = self.mul(matrix_A_max_allreduce[i], matrix_G_max_allreduce[i])
                temp_max = self.mul(temp_max, self.feature_map[i])
                temp_a = self.cast(temp_a, mstype.float16)
                temp_g = self.cast(temp_g, mstype.float16)
                if i == 53:
                    g = self.cube_matmul_left_fc(temp_g, g)
                    g = self.cube_matmul_right_fc(g, temp_a, temp_max)
                else:
                    g = self.cube_matmul_left(temp_g, g)
                    g = self.cube_matmul_right_mul(g, temp_a, temp_max)
                fake_A = self.assign(self.matrix_A[i], temp_a)
                fake_G = self.assign(self.matrix_G[i], temp_g)
                fake_max = self.assign(self.matrix_max_inv[i], temp_max)
                g = F.depend(g, fake_A)
                g = F.depend(g, fake_G)
                g = F.depend(g, fake_max)
                if i == 53:
                    new_grads = new_grads + (g,)
                else:
                    new_grads = new_grads + (g, gradients[i * 3 + 1], gradients[i * 3 + 2])
            gradients = new_grads
        else:
            new_grads = ()
            for i in range(54):
                g = gradients[i * 3]
                matrix_A = self.matrix_A[i]
                matrix_G = self.matrix_G[i]
                matrix_max = self.matrix_max_inv[i]
                if i == 53:
                    g = self.cube_matmul_left_fc(matrix_G, g)
                    g = self.cube_matmul_right_fc(g, matrix_A, matrix_max)
                    new_grads = new_grads + (g,)
                else:
                    g = self.cube_matmul_left(matrix_G, g)
                    g = self.cube_matmul_right_mul(g, matrix_A, matrix_max)
                    new_grads = new_grads + (g, gradients[i * 3 + 1], gradients[i * 3 + 2])
            gradients = new_grads

        if self.weight_decay > 0:
            gradients = self.hyper_map(F.partial(apply_decay, self.weight_decay), self.decay_flags,
                                       params, gradients)
        gradients = self.scale_grad(gradients)
        lr = self.get_lr()
        success = self.hyper_map(F.partial(_momentum_opt, self.opt, self.momentum, lr), gradients, params, moments)
        return success
