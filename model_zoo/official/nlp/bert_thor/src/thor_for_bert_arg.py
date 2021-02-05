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
"""momentum"""
import mindspore.common.dtype as mstype
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
from mindspore.common.parameter import ParameterTuple
from mindspore.common.tensor import Tensor
from mindspore.nn.optim.optimizer import Optimizer
from mindspore.ops import functional as F, composite as C, operations as P
from mindspore.parallel._utils import _get_device_num, _get_gradients_mean
from .grad_reducer_thor import DistributedGradReducerThor

momentum_opt = C.MultitypeFuncGraph("momentum_opt")


@momentum_opt.register("Function", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor")
def _tensor_run_opt_ext(opt, learning_rate, momentum, gradient, weight, moment):
    """Apply momentum optimizer to the weight parameter using Tensor."""
    success = True
    success = F.depend(success, opt(weight, moment, learning_rate, gradient, momentum))
    return success


op_add = P.AddN()
apply_decay = C.MultitypeFuncGraph("apply_decay")


@apply_decay.register("Number", "Bool", "Tensor", "Tensor")
def _tensor_apply_decay(weight_decay, if_apply, weight, gradient):
    """Get grad with weight_decay."""
    if if_apply:
        return op_add((weight * weight_decay, gradient))
    return gradient


class THOR(Optimizer):
    """THOR"""
    def __init__(self, params, learning_rate, momentum, matrix_A, matrix_G, weight_decay=0.0,
                 loss_scale=1.0, num_hidden_layers=24, batch_size=12, damping=0.03,
                 decay_filter=lambda x: 'layernorm' not in x.name.lower() and 'bias' not in x.name.lower()):
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
        self.matmul = P.MatMul()
        self.transpose = P.Transpose()
        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.mul = P.Mul()
        self.gather = P.Gather()
        self.matrix_A_inv = ()
        self.matrix_G_inv = ()
        self.num_hidden_layers = num_hidden_layers
        self.sqrt = P.Sqrt()
        self.assign = P.Assign()
        self.cast = P.Cast()
        self.thor = True
        self.weight_decay = weight_decay * loss_scale
        self.decay_flags = tuple(decay_filter(x) for x in self.parameters)
        self.expand = P.ExpandDims()
        self.square = P.Square()
        self.inv = P.Inv()
        self.batch_size = batch_size
        self.damping = damping
        self.one = Tensor(1, mstype.int32)
        self.cov_step = Parameter(initializer(0, [1], mstype.int32), requires_grad=False)
        mean = _get_gradients_mean()
        degree = _get_device_num()
        self.grad_reducer_g = DistributedGradReducerThor(self.parameters, 3, mean, degree)

    def construct(self, gradients):
        """construct of THOR"""
        params = self.params
        moments = self.moments
        encoder_layers_num = 16
        if self.thor:
            new_grads = ()
            # process embedding layer
            for em_idx in range(3):
                g = gradients[em_idx]
                matrix_idx = em_idx
                temp_a_ori = self.matrix_A[matrix_idx]
                temp_g = self.matrix_G[matrix_idx]
                temp_a_ori = F.depend(temp_a_ori, g)
                temp_g = F.depend(temp_g, g)
                temp_a = self.expand(temp_a_ori, 1)
                temp_a = self.cast(temp_a, mstype.float16)
                temp_g = self.cast(temp_g, mstype.float16)
                g = self.cast(g, mstype.float16)
                g = self.mul(temp_a, g)
                g = self.matmul(g, temp_g)
                g = self.cast(g, mstype.float32)
                fake_A = self.assign(self.matrix_A[matrix_idx], temp_a_ori)
                fake_G = self.assign(self.matrix_G[matrix_idx], temp_g)
                g = F.depend(g, fake_A)
                g = F.depend(g, fake_G)
                new_grads = new_grads + (g,)
            # process bert_embedding_postprocessor.layernorm
            grad_idx = 3
            beta_grad = gradients[grad_idx]
            gamma_grad = gradients[grad_idx + 1]
            normalizer = self.batch_size
            normalizer = self.cast(normalizer, mstype.float32)
            damping_step = self.gather(self.damping, self.cov_step, 0)
            damping_step = self.cast(damping_step, mstype.float32)
            self.cov_step = self.cov_step + self.one
            damping = self.sqrt(damping_step)
            beta = self.square(beta_grad)
            beta_cov = self.mul(beta, 1.0 / normalizer)
            beta_cov = beta_cov + damping
            beta_inv = self.inv(beta_cov)
            gamma = self.square(gamma_grad)
            gamma_cov = self.mul(gamma, 1.0 / normalizer)
            gamma_cov = gamma_cov + damping
            gamma_inv = self.inv(gamma_cov)
            beta = self.mul(beta_inv, beta_grad)
            gamma = self.mul(gamma_inv, gamma_grad)
            new_grads = new_grads + (beta, gamma)

            for i in range(self.num_hidden_layers):
                encoder_begin_idx = encoder_layers_num * i + 5
                for j in range(0, encoder_layers_num, 2):
                    grad_idx = encoder_begin_idx + j
                    if j in (8, 14):
                        # process layernorm layer
                        beta_grad = gradients[grad_idx]
                        gamma_grad = gradients[grad_idx + 1]
                        normalizer = self.batch_size
                        normalizer = self.cast(normalizer, mstype.float32)
                        beta = self.square(beta_grad)
                        beta_cov = self.mul(beta, 1.0 / normalizer)
                        beta_cov = beta_cov + damping
                        beta_inv = self.inv(beta_cov)
                        gamma = self.square(gamma_grad)
                        gamma_cov = self.mul(gamma, 1.0 / normalizer)
                        gamma_cov = gamma_cov + damping
                        gamma_inv = self.inv(gamma_cov)
                        beta = self.mul(beta_inv, beta_grad)
                        gamma = self.mul(gamma_inv, gamma_grad)
                        new_grads = new_grads + (beta, gamma)
                    else:
                        g = gradients[grad_idx]
                        offset_idx = 0
                        if j in (0, 2, 4, 6):
                            offset_idx = j // 2
                        elif j in (10, 12):
                            offset_idx = j // 2 - 1
                        matrix_idx = 6 * i + offset_idx + 3
                        temp_a = self.matrix_A[matrix_idx]
                        temp_g = self.matrix_G[matrix_idx]
                        temp_a = F.depend(temp_a, g)
                        temp_g = F.depend(temp_g, g)
                        temp_a = self.cast(temp_a, mstype.float16)
                        temp_g = self.cast(temp_g, mstype.float16)
                        g = self.cast(g, mstype.float16)
                        g = self.matmul(temp_g, g)
                        g = self.matmul(g, temp_a)
                        g = self.cast(g, mstype.float32)
                        fake_A = self.assign(self.matrix_A[matrix_idx], temp_a)
                        fake_G = self.assign(self.matrix_G[matrix_idx], temp_g)
                        g = F.depend(g, fake_A)
                        g = F.depend(g, fake_G)
                        new_grads = new_grads + (g,)
                        new_grads = new_grads + (gradients[grad_idx + 1],)

            # process pooler layer
            pooler_layer_idx = encoder_layers_num * self.num_hidden_layers + 5
            matrix_idx = self.num_hidden_layers * 6 + 3
            g = gradients[pooler_layer_idx]
            pooler_bias = gradients[pooler_layer_idx + 1]
            temp_a = self.matrix_A[matrix_idx]
            temp_g = self.matrix_G[matrix_idx]
            temp_a = F.depend(temp_a, g)
            temp_g = F.depend(temp_g, g)
            temp_a = self.cast(temp_a, mstype.float16)
            temp_g = self.cast(temp_g, mstype.float16)
            g = self.cast(g, mstype.float16)
            g = self.matmul(temp_g, g)
            g = self.matmul(g, temp_a)
            g = self.cast(g, mstype.float32)
            fake_A = self.assign(self.matrix_A[matrix_idx], temp_a)
            fake_G = self.assign(self.matrix_G[matrix_idx], temp_g)
            g = F.depend(g, fake_A)
            g = F.depend(g, fake_G)
            new_grads = new_grads + (g, pooler_bias)

            # cls1 fully connect layer for masked language model(mlm)
            mlm_fc_idx = encoder_layers_num * self.num_hidden_layers + 8
            matrix_idx = self.num_hidden_layers * 6 + 4
            g = gradients[mlm_fc_idx]
            mlm_bias = gradients[mlm_fc_idx + 1]
            temp_a = self.matrix_A[matrix_idx]
            temp_g = self.matrix_G[matrix_idx]
            temp_a = F.depend(temp_a, g)
            temp_g = F.depend(temp_g, g)
            temp_a = self.cast(temp_a, mstype.float16)
            temp_g = self.cast(temp_g, mstype.float16)
            g = self.cast(g, mstype.float16)
            g = self.matmul(temp_g, g)
            g = self.matmul(g, temp_a)
            g = self.cast(g, mstype.float32)
            # add bert.cls1.output_bias grad
            fake_A = self.assign(self.matrix_A[matrix_idx], temp_a)
            fake_G = self.assign(self.matrix_G[matrix_idx], temp_g)
            g = F.depend(g, fake_A)
            g = F.depend(g, fake_G)
            new_grads = new_grads + (gradients[mlm_fc_idx - 1],)
            new_grads = new_grads + (g, mlm_bias)
            # add bert.cls1.layernorm grad
            begin_idx = mlm_fc_idx + 2
            end_idx = mlm_fc_idx + 4
            new_grads = new_grads + gradients[begin_idx: end_idx]

            length = len(gradients)
            new_grads = new_grads + gradients[length - 2: length]
            gradients = new_grads
            gradients = self.grad_reducer_g(gradients)
        else:
            new_grads = ()
            # process embedding layer
            for em_idx in range(3):
                g = gradients[em_idx]
                matrix_idx = em_idx
                temp_a = self.matrix_A[matrix_idx]
                temp_g = self.matrix_G[matrix_idx]
                temp_a = self.expand(temp_a, 1)
                temp_a = self.cast(temp_a, mstype.float16)
                temp_g = self.cast(temp_g, mstype.float16)
                g = self.cast(g, mstype.float16)
                g = self.mul(temp_a, g)
                g = self.matmul(g, temp_g)
                g = self.cast(g, mstype.float32)
                new_grads = new_grads + (g,)
            # process bert_embedding_postprocessor.layernorm
            grad_idx = 3
            beta_grad = gradients[grad_idx]
            gamma_grad = gradients[grad_idx + 1]
            normalizer = self.batch_size
            normalizer = self.cast(normalizer, mstype.float32)
            damping_step = self.gather(self.damping, self.cov_step, 0)
            damping_step = self.cast(damping_step, mstype.float32)
            self.cov_step = self.cov_step + self.one
            damping = self.sqrt(damping_step)
            beta = self.square(beta_grad)
            beta_cov = self.mul(beta, 1.0 / normalizer)
            beta_cov = beta_cov + damping
            beta_inv = self.inv(beta_cov)
            gamma = self.square(gamma_grad)
            gamma_cov = self.mul(gamma, 1.0 / normalizer)
            gamma_cov = gamma_cov + damping
            gamma_inv = self.inv(gamma_cov)
            beta = self.mul(beta_inv, beta_grad)
            gamma = self.mul(gamma_inv, gamma_grad)
            new_grads = new_grads + (beta, gamma)

            for i in range(self.num_hidden_layers):
                encoder_begin_idx = encoder_layers_num * i + 5
                for j in range(0, encoder_layers_num, 2):
                    grad_idx = encoder_begin_idx + j
                    if j in (8, 14):
                        # process layernorm layer
                        beta_grad = gradients[grad_idx]
                        gamma_grad = gradients[grad_idx + 1]
                        normalizer = self.batch_size
                        normalizer = self.cast(normalizer, mstype.float32)
                        beta = self.square(beta_grad)
                        beta_cov = self.mul(beta, 1.0 / normalizer)
                        beta_cov = beta_cov + damping
                        beta_inv = self.inv(beta_cov)
                        gamma = self.square(gamma_grad)
                        gamma_cov = self.mul(gamma, 1.0 / normalizer)
                        gamma_cov = gamma_cov + damping
                        gamma_inv = self.inv(gamma_cov)
                        beta = self.mul(beta_inv, beta_grad)
                        gamma = self.mul(gamma_inv, gamma_grad)
                        new_grads = new_grads + (beta, gamma)
                    else:
                        g = gradients[grad_idx]
                        offset_idx = 0
                        if j in (0, 2, 4, 6):
                            offset_idx = j // 2
                        elif j in (10, 12):
                            offset_idx = j // 2 - 1
                        matrix_idx = 6 * i + offset_idx + 3
                        temp_a = self.matrix_A[matrix_idx]
                        temp_g = self.matrix_G[matrix_idx]
                        temp_a = self.cast(temp_a, mstype.float16)
                        temp_g = self.cast(temp_g, mstype.float16)
                        g = self.cast(g, mstype.float16)
                        g = self.matmul(temp_g, g)
                        g = self.matmul(g, temp_a)
                        g = self.cast(g, mstype.float32)
                        new_grads = new_grads + (g,)
                        new_grads = new_grads + (gradients[grad_idx + 1],)

            # process pooler layer
            pooler_layer_idx = encoder_layers_num * self.num_hidden_layers + 5
            matrix_idx = self.num_hidden_layers * 6 + 3
            g = gradients[pooler_layer_idx]
            pooler_bias = gradients[pooler_layer_idx + 1]
            temp_a = self.matrix_A[matrix_idx]
            temp_g = self.matrix_G[matrix_idx]
            temp_a = self.cast(temp_a, mstype.float16)
            temp_g = self.cast(temp_g, mstype.float16)
            g = self.cast(g, mstype.float16)
            g = self.matmul(temp_g, g)
            g = self.matmul(g, temp_a)
            g = self.cast(g, mstype.float32)
            new_grads = new_grads + (g, pooler_bias)

            # cls1 fully connect layer for masked language model(mlm)
            mlm_fc_idx = encoder_layers_num * self.num_hidden_layers + 8
            matrix_idx = self.num_hidden_layers * 6 + 4
            g = gradients[mlm_fc_idx]
            mlm_bias = gradients[mlm_fc_idx + 1]
            temp_a = self.matrix_A[matrix_idx]
            temp_g = self.matrix_G[matrix_idx]
            temp_a = self.cast(temp_a, mstype.float16)
            temp_g = self.cast(temp_g, mstype.float16)
            g = self.cast(g, mstype.float16)
            g = self.matmul(temp_g, g)
            g = self.matmul(g, temp_a)
            g = self.cast(g, mstype.float32)
            # add bert.cls1.output_bias grad
            new_grads = new_grads + (gradients[mlm_fc_idx - 1],)
            new_grads = new_grads + (g, mlm_bias)
            # add bert.cls1.layernorm grad
            begin_idx = mlm_fc_idx + 2
            end_idx = mlm_fc_idx + 4
            new_grads = new_grads + gradients[begin_idx: end_idx]

            length = len(gradients)
            new_grads = new_grads + gradients[length - 2: length]
            gradients = new_grads
            gradients = self.grad_reducer_g(gradients)

        if self.weight_decay > 0:
            gradients = self.hyper_map(F.partial(apply_decay, self.weight_decay), self.decay_flags,
                                       params, gradients)
        gradients = self.scale_grad(gradients)
        lr = self.get_lr()
        success = self.hyper_map(F.partial(momentum_opt, self.opt, lr, self.momentum), gradients, params, moments)
        return success
