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
"""wide and deep model"""
import numpy as np
from mindspore import nn
from mindspore import Parameter, ParameterTuple
import mindspore.common.dtype as mstype
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore.nn.optim import Adam, FTRL
from mindspore.common.initializer import Uniform, initializer
from mindspore.parallel._utils import _get_device_num, _get_parallel_mode, _get_gradients_mean
from mindspore.context import ParallelMode
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from mindspore.communication.management import get_group_size

np_type = np.float32
ms_type = mstype.float32


def init_method(method, shape, name, max_val=1.0):
    '''
    parameter init method
    '''
    if method in ['uniform']:
        params = Parameter(initializer(
            Uniform(max_val), shape, ms_type), name=name)
    elif method == "one":
        params = Parameter(initializer("ones", shape, ms_type), name=name)
    elif method == 'zero':
        params = Parameter(initializer("zeros", shape, ms_type), name=name)
    elif method == "normal":
        params = Parameter(initializer("normal", shape, ms_type), name=name)
    return params


def init_var_dict(init_args, in_vars):
    '''
    var init function
    '''
    var_map = {}
    _, _max_val = init_args
    for _, item in enumerate(in_vars):
        key, shape, method = item
        if key not in var_map.keys():
            if method in ['random', 'uniform']:
                var_map[key] = Parameter(initializer(
                    Uniform(_max_val), shape, ms_type), name=key)
            elif method == "one":
                var_map[key] = Parameter(initializer(
                    "ones", shape, ms_type), name=key)
            elif method == "zero":
                var_map[key] = Parameter(initializer(
                    "zeros", shape, ms_type), name=key)
            elif method == 'normal':
                var_map[key] = Parameter(initializer(
                    "normal", shape, ms_type), name=key)
    return var_map


class DenseLayer(nn.Cell):
    """
    Dense Layer for Deep Layer of WideDeep Model;
    Containing: activation, matmul, bias_add;
    Args:
    """

    def __init__(self, input_dim, output_dim, weight_bias_init, act_str,
                 keep_prob=0.7, scale_coef=1.0, convert_dtype=True):
        super(DenseLayer, self).__init__()
        weight_init, bias_init = weight_bias_init
        self.weight = init_method(
            weight_init, [input_dim, output_dim], name="weight")
        self.bias = init_method(bias_init, [output_dim], name="bias")
        self.act_func = self._init_activation(act_str)
        self.matmul = P.MatMul(transpose_b=False)
        self.bias_add = P.BiasAdd()
        self.cast = P.Cast()
        #self.dropout = Dropout(keep_prob=keep_prob)
        self.mul = P.Mul()
        self.realDiv = P.RealDiv()
        self.scale_coef = scale_coef
        self.convert_dtype = convert_dtype

    def _init_activation(self, act_str):
        act_str = act_str.lower()
        if act_str == "relu":
            act_func = P.ReLU()
        elif act_str == "sigmoid":
            act_func = P.Sigmoid()
        elif act_str == "tanh":
            act_func = P.Tanh()
        return act_func

    def construct(self, x):
        x = self.act_func(x)
        x = self.mul(x, self.scale_coef)
        if self.convert_dtype:
            x = self.cast(x, mstype.float16)
            weight = self.cast(self.weight, mstype.float16)
            wx = self.matmul(x, weight)
            wx = self.cast(wx, mstype.float32)
        else:
            wx = self.matmul(x, self.weight)
        wx = self.realDiv(wx, self.scale_coef)
        output = self.bias_add(wx, self.bias)
        return output


class WideDeepModel(nn.Cell):
    """
        From paper: " Wide & Deep Learning for Recommender Systems"
        Args:
            config (Class): The default config of Wide&Deep
    """

    def __init__(self, config):
        super(WideDeepModel, self).__init__()
        self.batch_size = config.batch_size
        parallel_mode = _get_parallel_mode()
        if parallel_mode in (ParallelMode.SEMI_AUTO_PARALLEL, ParallelMode.AUTO_PARALLEL):
            self.batch_size = self.batch_size * get_group_size()
        self.field_size = config.field_size
        self.vocab_size = config.vocab_size
        self.emb_dim = config.emb_dim
        self.deep_layer_dims_list = config.deep_layer_dim
        self.deep_layer_act = config.deep_layer_act
        self.init_args = config.init_args
        self.weight_init, self.bias_init = config.weight_bias_init
        self.weight_bias_init = config.weight_bias_init
        self.emb_init = config.emb_init
        self.drop_out = config.dropout_flag
        self.keep_prob = config.keep_prob
        self.deep_input_dims = self.field_size * self.emb_dim
        self.layer_dims = self.deep_layer_dims_list + [1]
        self.all_dim_list = [self.deep_input_dims] + self.layer_dims

        init_acts = [('Wide_w', [self.vocab_size, 1], self.emb_init),
                     ('V_l2', [self.vocab_size, self.emb_dim], self.emb_init),
                     ('Wide_b', [1], self.emb_init)]
        var_map = init_var_dict(self.init_args, init_acts)
        self.wide_w = var_map["Wide_w"]
        self.wide_b = var_map["Wide_b"]
        self.embedding_table = var_map["V_l2"]
        self.dense_layer_1 = DenseLayer(self.all_dim_list[0],
                                        self.all_dim_list[1],
                                        self.weight_bias_init,
                                        self.deep_layer_act, convert_dtype=True)
        self.dense_layer_2 = DenseLayer(self.all_dim_list[1],
                                        self.all_dim_list[2],
                                        self.weight_bias_init,
                                        self.deep_layer_act, convert_dtype=True)
        self.dense_layer_3 = DenseLayer(self.all_dim_list[2],
                                        self.all_dim_list[3],
                                        self.weight_bias_init,
                                        self.deep_layer_act, convert_dtype=True)
        self.dense_layer_4 = DenseLayer(self.all_dim_list[3],
                                        self.all_dim_list[4],
                                        self.weight_bias_init,
                                        self.deep_layer_act, convert_dtype=True)
        self.dense_layer_5 = DenseLayer(self.all_dim_list[4],
                                        self.all_dim_list[5],
                                        self.weight_bias_init,
                                        self.deep_layer_act, convert_dtype=True)

        self.gather_v2 = P.Gather().shard(((1, 8), (1, 1)))
        self.gather_v2_1 = P.Gather()
        self.mul = P.Mul()
        self.reduce_sum = P.ReduceSum(keep_dims=False)
        self.reshape = P.Reshape()
        self.square = P.Square()
        self.shape = P.Shape()
        self.tile = P.Tile()
        self.concat = P.Concat(axis=1)
        self.cast = P.Cast()

    def construct(self, id_hldr, wt_hldr):
        """
        Args:
            id_hldr: batch ids;
            wt_hldr: batch weights;
        """
        mask = self.reshape(wt_hldr, (self.batch_size, self.field_size, 1))
        # Wide layer
        wide_id_weight = self.gather_v2_1(self.wide_w, id_hldr, 0)
        wx = self.mul(wide_id_weight, mask)
        wide_out = self.reshape(self.reduce_sum(wx, 1) + self.wide_b, (-1, 1))
        # Deep layer
        deep_id_embs = self.gather_v2(self.embedding_table, id_hldr, 0)
        vx = self.mul(deep_id_embs, mask)
        deep_in = self.reshape(vx, (-1, self.field_size * self.emb_dim))
        deep_in = self.dense_layer_1(deep_in)
        deep_in = self.dense_layer_2(deep_in)
        deep_in = self.dense_layer_3(deep_in)
        deep_in = self.dense_layer_4(deep_in)
        deep_out = self.dense_layer_5(deep_in)
        out = wide_out + deep_out
        return out, self.embedding_table


class NetWithLossClass(nn.Cell):

    """"
    Provide WideDeep training loss through network.
    Args:
        network (Cell): The training network
        config (Class): WideDeep config
    """

    def __init__(self, network, config):
        super(NetWithLossClass, self).__init__(auto_prefix=False)
        self.network = network
        self.l2_coef = config.l2_coef
        self.loss = P.SigmoidCrossEntropyWithLogits()
        self.square = P.Square().shard(((1, get_group_size()),))
        self.reduceMean_false = P.ReduceMean(keep_dims=False)
        self.reduceSum_false = P.ReduceSum(keep_dims=False)

    def construct(self, batch_ids, batch_wts, label):
        predict, embedding_table = self.network(batch_ids, batch_wts)
        log_loss = self.loss(predict, label)
        wide_loss = self.reduceMean_false(log_loss)
        l2_loss_v = self.reduceSum_false(self.square(embedding_table)) / 2
        deep_loss = self.reduceMean_false(log_loss) + self.l2_coef * l2_loss_v

        return wide_loss, deep_loss


class IthOutputCell(nn.Cell):
    def __init__(self, network, output_index):
        super(IthOutputCell, self).__init__()
        self.network = network
        self.output_index = output_index

    def construct(self, x1, x2, x3):
        predict = self.network(x1, x2, x3)[self.output_index]
        return predict


class TrainStepWrap(nn.Cell):
    """
    Encapsulation class of WideDeep network training.
    Append Adam and FTRL optimizers to the training network after that construct
    function can be called to create the backward graph.
    Args:
        network (Cell): the training network. Note that loss function should have been added.
        sens (Number): The adjust parameter. Default: 1000.0
    """

    def __init__(self, network, sens=1000.0):
        super(TrainStepWrap, self).__init__()
        self.network = network
        self.network.set_train()
        self.trainable_params = network.trainable_params()
        weights_w = []
        weights_d = []
        for params in self.trainable_params:
            if 'wide' in params.name:
                weights_w.append(params)
            else:
                weights_d.append(params)
        self.weights_w = ParameterTuple(weights_w)
        self.weights_d = ParameterTuple(weights_d)
        self.optimizer_w = FTRL(learning_rate=1e-2, params=self.weights_w,
                                l1=1e-8, l2=1e-8, initial_accum=1.0)
        self.optimizer_d = Adam(
            self.weights_d, learning_rate=3.5e-4, eps=1e-8, loss_scale=sens)
        self.hyper_map = C.HyperMap()
        self.grad_w = C.GradOperation(get_by_list=True,
                                      sens_param=True)
        self.grad_d = C.GradOperation(get_by_list=True,
                                      sens_param=True)
        self.sens = sens
        self.loss_net_w = IthOutputCell(network, output_index=0)
        self.loss_net_d = IthOutputCell(network, output_index=1)

        self.reducer_flag = False
        self.grad_reducer_w = None
        self.grad_reducer_d = None
        parallel_mode = _get_parallel_mode()
        self.reducer_flag = parallel_mode in (ParallelMode.DATA_PARALLEL,
                                              ParallelMode.HYBRID_PARALLEL)
        if self.reducer_flag:
            mean = _get_gradients_mean()
            degree = _get_device_num()
            self.grad_reducer_w = DistributedGradReducer(self.optimizer_w.parameters, mean, degree)
            self.grad_reducer_d = DistributedGradReducer(self.optimizer_d.parameters, mean, degree)

    def construct(self, batch_ids, batch_wts, label):
        weights_w = self.weights_w
        weights_d = self.weights_d
        loss_w, loss_d = self.network(batch_ids, batch_wts, label)
        sens_w = P.Fill()(P.DType()(loss_w), P.Shape()(loss_w), self.sens)
        sens_d = P.Fill()(P.DType()(loss_d), P.Shape()(loss_d), self.sens)
        grads_w = self.grad_w(self.loss_net_w, weights_w)(batch_ids, batch_wts,
                                                          label, sens_w)
        grads_d = self.grad_d(self.loss_net_d, weights_d)(batch_ids, batch_wts,
                                                          label, sens_d)
        if self.reducer_flag:
            grads_w = self.grad_reducer_w(grads_w)
            grads_d = self.grad_reducer_d(grads_d)
        return F.depend(loss_w, self.optimizer_w(grads_w)), F.depend(loss_d,
                                                                     self.optimizer_d(grads_d))


class PredictWithSigmoid(nn.Cell):
    def __init__(self, network):
        super(PredictWithSigmoid, self).__init__()
        self.network = network
        self.sigmoid = P.Sigmoid()

    def construct(self, batch_ids, batch_wts, labels):
        logits, _, _, = self.network(batch_ids, batch_wts)
        pred_probs = self.sigmoid(logits)
        return logits, pred_probs, labels
