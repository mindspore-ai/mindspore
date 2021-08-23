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
# ===========================================================================
""" Fat-deepFFM"""
import numpy as np
from src.lr_generator import get_warmup_linear_lr

import mindspore.common.dtype as mstype
from mindspore.common.initializer import initializer

import mindspore.ops as P
from mindspore.ops import composite as C
from mindspore.ops import functional as F

from mindspore import Parameter, ParameterTuple
from mindspore import Tensor

from mindspore import nn
from mindspore.nn import Adam, DistributedGradReducer, Dropout
from mindspore.nn.probability.distribution import Uniform
from mindspore.context import ParallelMode

from mindspore.parallel._utils import _get_parallel_mode, _get_gradients_mean, _get_device_num


def init_method(method, shape, name, max_val=1.0):
    """ Initialize weight"""
    params = None
    if method in ['uniform']:
        params = Parameter(initializer(Uniform(max_val), shape, mstype.float32), name=name)
    elif method == "one":
        params = Parameter(initializer("ones", shape, mstype.float32), name=name)
    elif method == 'zero':
        params = Parameter(initializer("zeros", shape, mstype.float32), name=name)
    elif method == "normal":
        params = Parameter(Tensor(np.random.normal(loc=0.0, scale=0.01, size=shape).astype(dtype=np.float32)),
                           name=name)
    return params


class DenseFeaturesLinear(nn.Cell):
    """ First order linear combination of dense features"""

    def __init__(self, nume_dims=13, output_dim=1):
        super().__init__()
        self.dense = DenseLayer(nume_dims, output_dim, ['normal', 'normal'],
                                "relu", use_dropout=False, use_act=False, use_bn=False)

    def construct(self, x):
        res = self.dense(x)
        return res


class DenseHighFeaturesLinear(nn.Cell):
    """High-order linear combinations of dense features"""

    def __init__(self, num_dims=13, output_dim=1):
        super().__init__()
        self.dense_d3_1 = DenseLayer(num_dims, 512, ['normal', 'normal'], "relu")
        self.dense_d3_2 = DenseLayer(512, 512, ['normal', 'normal'], "relu")
        self.dense_d3_3 = DenseLayer(512, output_dim, ['normal', 'normal'], "relu",
                                     use_dropout=False, use_act=False, use_bn=False)

    def construct(self, x):
        x = self.dense_d3_1(x)
        x = self.dense_d3_2(x)
        res = self.dense_d3_3(x)
        return res


# 计算FFM一阶类别特征
class SparseFeaturesLinear(nn.Cell):
    """First-order linear combination of sparse features"""

    def __init__(self, config, output_dim=1):
        super().__init__()
        self.weight = Parameter(Tensor(
            np.random.normal(loc=0.0, scale=0.01, size=[config.vocb_size, output_dim]).astype(dtype=np.float32)))
        self.reduceSum = P.ReduceSum(keep_dims=True)
        self.gather = P.Gather()
        self.squeeze = P.Squeeze(2)

    def construct(self, x):  # [b,26]
        res = self.gather(self.weight, x, 0)
        res = self.reduceSum(res, 1)
        res = self.squeeze(res)
        return res


class SparseFeaturesFFMEmbedding(nn.Cell):
    """The sparse features are dense"""

    def __init__(self, config):
        super().__init__()
        self.num_field = 26
        self.gather = P.Gather()
        self.concat = P.Concat(axis=1)
        self.weights = []
        for _ in range(self.num_field):
            weight = Parameter(Tensor(
                np.random.normal(loc=0.0, scale=0.01, size=[config.vocb_size, config.emb_dim]).astype(
                    dtype=np.float32)))
            self.weights.append(weight)

    def construct(self, x):
        xs = ()
        for i in range(self.num_field):
            xs += (self.gather(self.weights[i], x, 0),)
        xs = self.concat(xs)
        return xs


class DenseLayer(nn.Cell):
    """Full connection layer templates"""

    def __init__(self, input_dim, output_dim, weight_bias_init, act_str, keep_prob=0.9, convert_dtype=True,
                 use_dropout=True, use_act=True, use_bn=True):
        super(DenseLayer, self).__init__()
        weight_init, bias_init = weight_bias_init
        self.weight = init_method(weight_init, [input_dim, output_dim], name="weight")
        self.bias = init_method(bias_init, [output_dim], name="bias")
        self.act_func = self._init_activation(act_str)
        self.matmul = P.MatMul(transpose_b=False)
        self.bias_add = P.BiasAdd()
        self.cast = P.Cast()
        self.dropout = Dropout(keep_prob=keep_prob)
        self.mul = P.Mul()
        self.realDiv = P.RealDiv()
        self.convert_dtype = convert_dtype
        self.use_act = use_act
        self.use_dropout = use_dropout
        self.use_bn = use_bn
        self.bn = nn.BatchNorm1d(output_dim)

    def _init_activation(self, act_str):
        act_func = None
        act_str = act_str.lower()
        if act_str == "relu":
            act_func = P.ReLU()
        elif act_str == "sigmoid":
            act_func = P.Sigmoid()
        elif act_str == "tanh":
            act_func = P.Tanh()
        return act_func

    def construct(self, x):
        """Construct function"""
        if self.convert_dtype:
            x = self.cast(x, mstype.float16)
            weight = self.cast(self.weight, mstype.float16)
            bias = self.cast(self.bias, mstype.float16)
            wx = self.matmul(x, weight)
            wx = self.bias_add(wx, bias)
            if self.use_bn:
                wx = self.bn(wx)
            if self.use_act:
                wx = self.act_func(wx)
            wx = self.cast(wx, mstype.float32)
        else:
            wx = self.matmul(x, self.weight)
            wx = self.bias_add(wx, self.bias)
            if self.use_bn:
                wx = self.bn(wx)
            if self.use_act:
                wx = self.act_func(wx)
        if self.use_dropout:
            wx = self.dropout(wx)
        return wx


class AttentionFeaturelayer(nn.Cell):
    """Attentional mechanism"""

    def __init__(self, config):
        super().__init__()
        self.cats_field = config.cats_dim
        self.weight_bias_init = config.weight_bias_init
        self.att_dim, self.att_layer_act = config.att_layer_args
        # attention部分
        self.att_conv = nn.Conv1d(in_channels=config.emb_dim, out_channels=1, kernel_size=1, stride=1)
        self.att_bn = nn.BatchNorm1d(676)
        self.att_re = nn.ReLU()
        self.dense_att_1 = DenseLayer(self.cats_field * self.cats_field, self.att_dim, self.weight_bias_init,
                                      self.att_layer_act)
        self.dense_att_2 = DenseLayer(self.att_dim, self.cats_field * self.cats_field, self.weight_bias_init,
                                      self.att_layer_act)
        self.transpose = P.Transpose()
        self.squeeze = P.Squeeze(axis=1)
        self.mul = P.Mul()
        self.addDim = P.ExpandDims()

    def construct(self, x):  # [b,676,8]
        tx = self.transpose(x, (0, 2, 1))  # 转换维度
        tx = self.att_conv(tx)  # [b ,1, 676]
        att_xs = self.att_re(self.att_bn(self.squeeze(tx)))  # (b,676)
        att_xs = self.dense_att_1(att_xs)  # [b, 256]
        att_xs = self.dense_att_2(att_xs)  # [b, 676]
        att_xs = self.addDim(att_xs, 2)  # [b,676,1]
        out = self.mul(x, att_xs)
        return out


# 计算FFM二阶类别特征
class FieldAwareFactorizationMachine(nn.Cell):
    """ Sparse feature crossover"""

    def __init__(self):
        super().__init__()
        self.num_fields = 26
        self.concat = P.Concat(axis=1)
        self.mul = P.Mul()
        self.stack = P.Stack(axis=0)
        self.cat = P.Concat(axis=0)
        self.sum = P.ReduceSum(keep_dims=True)
        self.squeeze = P.Squeeze(axis=2)
        self.squeeze1 = P.Squeeze(axis=1)
        self.transpose = P.Transpose()

    def construct(self, x):  # [b,676,8]
        """ Sparse feature crossover """
        ix = ()
        for i in range(25):
            for j in range(i + 1, 26):
                m = 26 * j + i
                n = 26 * i + j
                ix += (self.squeeze1(self.mul(x[::, m:m + 1:1, ::], x[::, n:n + 1:1, ::])),)
        ix1 = self.stack(ix[:190])  # [190 b 8]
        ix2 = self.stack(ix[190:])  # [135 b 8]
        ix = self.cat((ix1, ix2))  # [325 b 8]
        ix = self.sum(ix, 2)  # [325 b 1]
        ix = self.squeeze(ix)  # [325 b]
        ix = self.transpose(ix, (1, 0))  # [b 325]
        ix = self.sum(ix, 1)  # [b 1]
        return ix


# 计算深度网络mlp
class MultiLayerPerceptron(nn.Cell):
    """Deep network layer"""

    def __init__(self, config, input_dim):
        super().__init__()
        self.weight_bias_init = config.weight_bias_init
        self.att_dim, self.att_layer_act = config.deep_layer_args
        self.keep_prob = config.keep_prob
        self.flatten = nn.Flatten()
        self.d_dense = DenseLayer(config.dense_dim, input_dim, self.weight_bias_init,
                                  self.att_layer_act, self.keep_prob)
        self.dense1 = DenseLayer(input_dim, self.att_dim[0], self.weight_bias_init,
                                 self.att_layer_act, self.keep_prob)
        self.dense2 = DenseLayer(self.att_dim[0], self.att_dim[1], self.weight_bias_init,
                                 self.att_layer_act, self.keep_prob)
        self.dense3 = DenseLayer(self.att_dim[1], self.att_dim[2], self.weight_bias_init,
                                 self.att_layer_act, self.keep_prob)
        self.dense4 = DenseLayer(self.att_dim[2], self.att_dim[3], self.weight_bias_init,
                                 self.att_layer_act, self.keep_prob)
        self.dense5 = DenseLayer(self.att_dim[3], self.att_dim[4], self.weight_bias_init,
                                 self.att_layer_act, self.keep_prob, use_dropout=False, use_bn=False, use_act=False)

    def construct(self, d, x):
        x = self.flatten(x) + self.d_dense(d)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.dense5(x)
        return x


class Fat_DeepFFM(nn.Cell):
    """"The general model"""

    def __init__(self, config):
        super().__init__()
        self.dense_1st = DenseFeaturesLinear()  # 一阶数值特征
        self.dense_high = DenseHighFeaturesLinear()
        self.sparse_1st = SparseFeaturesLinear(config)  # 一阶类别特征
        self.FFMEmb = SparseFeaturesFFMEmbedding(config)
        self.attention = AttentionFeaturelayer(config)
        self.ffm = FieldAwareFactorizationMachine()
        self.embed_output_dim = 26 * 26 * config.emb_dim
        self.mlp = MultiLayerPerceptron(config, self.embed_output_dim)

    def construct(self, cats_vals, num_vals):
        """"cats_vals:[b,13], num_vals:[b 26]"""
        X_dense, X_sparse = num_vals, cats_vals
        FFME = self.FFMEmb(X_sparse)  # [b,676,8]
        dense_1st_res = self.dense_1st(X_dense)  # [b,1]
        dense_1st__high_res = self.dense_high(X_dense)
        sparse_1st_res = self.sparse_1st(X_sparse)  # [b,1]
        attention_res = self.attention(FFME)  # [b,676,8]
        ffm_res = self.ffm(attention_res)  # [b,1]
        mlp_res = self.mlp(X_dense, FFME)  # # [b,1]
        res = dense_1st_res + dense_1st__high_res + sparse_1st_res + ffm_res + mlp_res  # [b,1]
        return res


class NetWithLossClass(nn.Cell):
    """Get the model results"""

    def __init__(self, network):
        super(NetWithLossClass, self).__init__(auto_prefix=False)
        self.network = network
        self.loss = P.SigmoidCrossEntropyWithLogits()

    def construct(self, cats_vals, num_vals, label):
        predict = self.network(cats_vals, num_vals)
        loss = self.loss(predict, label)
        return loss


class TrainStepWrap(nn.Cell):
    """Reverse passing"""

    def __init__(self, network, config):
        super(TrainStepWrap, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_train()
        self.lr = get_warmup_linear_lr(config.lr_init, config.lr_end, config.epoch_size * config.steps_per_epoch)
        self.weights = ParameterTuple(network.trainable_params())
        self.optimizer = Adam(self.weights, learning_rate=self.lr, eps=config.epsilon,
                              loss_scale=config.loss_scale)
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.sens = config.loss_scale

        self.reducer_flag = False
        self.grad_reducer = None
        parallel_mode = _get_parallel_mode()
        if parallel_mode in (ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL):
            self.reducer_flag = True
        if self.reducer_flag:
            mean = _get_gradients_mean()
            degree = _get_device_num()
            self.grad_reducer = DistributedGradReducer(self.optimizer.parameters, mean, degree)

    def construct(self, cats_vals, num_vals, label):
        weights = self.weights
        loss = self.network(cats_vals, num_vals, label)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)  #
        grads = self.grad(self.network, weights)(cats_vals, num_vals, label, sens)
        if self.reducer_flag:
            grads = self.grad_reducer(grads)
        return F.depend(loss, self.optimizer(grads))


class ModelBuilder:
    """Get the model"""

    def __init__(self, config):
        self.config = config

    def get_train_eval_net(self):
        deepfm_net = Fat_DeepFFM(self.config)
        train_net = NetWithLossClass(deepfm_net)
        train_net = TrainStepWrap(train_net, self.config)
        test_net = PredictWithSigmoid(deepfm_net)
        return train_net, test_net


class PredictWithSigmoid(nn.Cell):
    """Model to predict"""

    def __init__(self, network):
        super(PredictWithSigmoid, self).__init__(auto_prefix=False)
        self.network = network
        self.sigmoid = P.Sigmoid()

    def construct(self, cats_vals, num_vals, label):
        logits = self.network(cats_vals, num_vals)
        pred_probs = self.sigmoid(logits)
        return logits, pred_probs, label
