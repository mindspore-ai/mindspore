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
from mindspore import nn, context
from mindspore import Parameter, ParameterTuple
import mindspore.common.dtype as mstype
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore.nn import Dropout
from mindspore.nn.optim import Adam, FTRL, LazyAdam
from mindspore.common.initializer import Uniform, initializer
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
    for _, iterm in enumerate(in_vars):
        key, shape, method = iterm
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
                 keep_prob=0.5, use_activation=True, convert_dtype=True, drop_out=False):
        super(DenseLayer, self).__init__()
        weight_init, bias_init = weight_bias_init
        self.weight = init_method(
            weight_init, [input_dim, output_dim], name="weight")
        self.bias = init_method(bias_init, [output_dim], name="bias")
        self.act_func = self._init_activation(act_str)
        self.matmul = P.MatMul(transpose_b=False)
        self.bias_add = P.BiasAdd()
        self.cast = P.Cast()
        self.dropout = Dropout(keep_prob=keep_prob)
        self.use_activation = use_activation
        self.convert_dtype = convert_dtype
        self.drop_out = drop_out

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
        '''
        Construct Dense layer
        '''
        if self.training and self.drop_out:
            x = self.dropout(x)
        if self.convert_dtype:
            x = self.cast(x, mstype.float16)
            weight = self.cast(self.weight, mstype.float16)
            bias = self.cast(self.bias, mstype.float16)
            wx = self.matmul(x, weight)
            wx = self.bias_add(wx, bias)
            if self.use_activation:
                wx = self.act_func(wx)
            wx = self.cast(wx, mstype.float32)
        else:
            wx = self.matmul(x, self.weight)
            wx = self.bias_add(wx, self.bias)
            if self.use_activation:
                wx = self.act_func(wx)
        return wx


class WideDeepModel(nn.Cell):
    """
        From paper: " Wide & Deep Learning for Recommender Systems"
        Args:
            config (Class): The default config of Wide&Deep
    """

    def __init__(self, config):
        super(WideDeepModel, self).__init__()
        self.batch_size = config.batch_size
        host_device_mix = bool(config.host_device_mix)
        parameter_server = bool(config.parameter_server)
        parallel_mode = context.get_auto_parallel_context("parallel_mode")
        is_auto_parallel = parallel_mode in (ParallelMode.SEMI_AUTO_PARALLEL, ParallelMode.AUTO_PARALLEL)
        if is_auto_parallel:
            self.batch_size = self.batch_size * get_group_size()
        is_field_slice = config.field_slice
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

        init_acts = [('Wide_b', [1], self.emb_init)]
        var_map = init_var_dict(self.init_args, init_acts)
        self.wide_b = var_map["Wide_b"]
        self.dense_layer_1 = DenseLayer(self.all_dim_list[0],
                                        self.all_dim_list[1],
                                        self.weight_bias_init,
                                        self.deep_layer_act,
                                        convert_dtype=True, drop_out=config.dropout_flag)
        self.dense_layer_2 = DenseLayer(self.all_dim_list[1],
                                        self.all_dim_list[2],
                                        self.weight_bias_init,
                                        self.deep_layer_act,
                                        convert_dtype=True, drop_out=config.dropout_flag)
        self.dense_layer_3 = DenseLayer(self.all_dim_list[2],
                                        self.all_dim_list[3],
                                        self.weight_bias_init,
                                        self.deep_layer_act,
                                        convert_dtype=True, drop_out=config.dropout_flag)
        self.dense_layer_4 = DenseLayer(self.all_dim_list[3],
                                        self.all_dim_list[4],
                                        self.weight_bias_init,
                                        self.deep_layer_act,
                                        convert_dtype=True, drop_out=config.dropout_flag)
        self.dense_layer_5 = DenseLayer(self.all_dim_list[4],
                                        self.all_dim_list[5],
                                        self.weight_bias_init,
                                        self.deep_layer_act,
                                        use_activation=False, convert_dtype=True, drop_out=config.dropout_flag)
        self.wide_mul = P.Mul()
        self.deep_mul = P.Mul()
        self.reduce_sum = P.ReduceSum(keep_dims=False)
        self.reshape = P.Reshape()
        self.deep_reshape = P.Reshape()
        self.square = P.Square()
        self.shape = P.Shape()
        self.tile = P.Tile()
        self.concat = P.Concat(axis=1)
        self.cast = P.Cast()
        if is_auto_parallel and host_device_mix and not is_field_slice:
            self.dense_layer_1.dropout.dropout_do_mask.shard(((1, get_group_size()),))
            self.dense_layer_1.dropout.dropout.shard(((1, get_group_size()),))
            self.dense_layer_1.matmul.shard(((1, get_group_size()), (get_group_size(), 1)))
            self.deep_embeddinglookup = nn.EmbeddingLookup(self.vocab_size, self.emb_dim,
                                                           slice_mode=nn.EmbeddingLookup.TABLE_COLUMN_SLICE)
            self.wide_embeddinglookup = nn.EmbeddingLookup(self.vocab_size, 1,
                                                           slice_mode=nn.EmbeddingLookup.TABLE_ROW_SLICE)
            self.deep_mul.shard(((1, 1, get_group_size()), (1, 1, 1)))
            self.deep_reshape.add_prim_attr("skip_redistribution", True)
            self.reduce_sum.add_prim_attr("cross_batch", True)
            self.embedding_table = self.deep_embeddinglookup.embedding_table
        elif is_auto_parallel and host_device_mix and is_field_slice and config.full_batch and config.manual_shape:
            manual_shapes = tuple((s[0] for s in config.manual_shape))
            self.deep_embeddinglookup = nn.EmbeddingLookup(self.vocab_size, self.emb_dim,
                                                           slice_mode=nn.EmbeddingLookup.FIELD_SLICE,
                                                           manual_shapes=manual_shapes)
            self.wide_embeddinglookup = nn.EmbeddingLookup(self.vocab_size, 1,
                                                           slice_mode=nn.EmbeddingLookup.FIELD_SLICE,
                                                           manual_shapes=manual_shapes)
            self.deep_mul.shard(((1, get_group_size(), 1), (1, get_group_size(), 1)))
            self.wide_mul.shard(((1, get_group_size(), 1), (1, get_group_size(), 1)))
            self.reduce_sum.shard(((1, get_group_size(), 1),))
            self.dense_layer_1.dropout.dropout_do_mask.shard(((1, get_group_size()),))
            self.dense_layer_1.dropout.dropout.shard(((1, get_group_size()),))
            self.dense_layer_1.matmul.shard(((1, get_group_size()), (get_group_size(), 1)))
            self.embedding_table = self.deep_embeddinglookup.embedding_table
        elif parameter_server:
            self.deep_embeddinglookup = nn.EmbeddingLookup(self.vocab_size, self.emb_dim)
            self.wide_embeddinglookup = nn.EmbeddingLookup(self.vocab_size, 1)
            self.embedding_table = self.deep_embeddinglookup.embedding_table
            self.deep_embeddinglookup.embedding_table.set_param_ps()
            self.wide_embeddinglookup.embedding_table.set_param_ps()
        else:
            self.deep_embeddinglookup = nn.EmbeddingLookup(self.vocab_size, self.emb_dim, target='DEVICE')
            self.wide_embeddinglookup = nn.EmbeddingLookup(self.vocab_size, 1, target='DEVICE')
            self.embedding_table = self.deep_embeddinglookup.embedding_table

    def construct(self, id_hldr, wt_hldr):
        """
        Args:
            id_hldr: batch ids;
            wt_hldr: batch weights;
        """
        mask = self.reshape(wt_hldr, (self.batch_size, self.field_size, 1))
        # Wide layer
        wide_id_weight = self.wide_embeddinglookup(id_hldr)
        wx = self.wide_mul(wide_id_weight, mask)
        wide_out = self.reshape(self.reduce_sum(wx, 1) + self.wide_b, (-1, 1))
        # Deep layer
        deep_id_embs = self.deep_embeddinglookup(id_hldr)
        vx = self.deep_mul(deep_id_embs, mask)
        deep_in = self.deep_reshape(vx, (-1, self.field_size * self.emb_dim))
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
        host_device_mix = bool(config.host_device_mix)
        parameter_server = bool(config.parameter_server)
        parallel_mode = context.get_auto_parallel_context("parallel_mode")
        is_auto_parallel = parallel_mode in (ParallelMode.SEMI_AUTO_PARALLEL, ParallelMode.AUTO_PARALLEL)
        self.no_l2loss = (is_auto_parallel if (host_device_mix or config.field_slice) else parameter_server)
        self.network = network
        self.l2_coef = config.l2_coef
        self.loss = P.SigmoidCrossEntropyWithLogits()
        self.square = P.Square()
        self.reduceMean_false = P.ReduceMean(keep_dims=False)
        if is_auto_parallel:
            self.reduceMean_false.add_prim_attr("cross_batch", True)
        self.reduceSum_false = P.ReduceSum(keep_dims=False)

    def construct(self, batch_ids, batch_wts, label):
        '''
        Construct NetWithLossClass
        '''
        predict, embedding_table = self.network(batch_ids, batch_wts)
        log_loss = self.loss(predict, label)
        wide_loss = self.reduceMean_false(log_loss)
        if self.no_l2loss:
            deep_loss = wide_loss
        else:
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
        network (Cell): The training network. Note that loss function should have been added.
        sens (Number): The adjust parameter. Default: 1024.0
        host_device_mix (Bool): Whether run in host and device mix mode. Default: False
        parameter_server (Bool): Whether run in parameter server mode. Default: False
    """

    def __init__(self, network, sens=1024.0, host_device_mix=False, parameter_server=False):
        super(TrainStepWrap, self).__init__()
        parallel_mode = context.get_auto_parallel_context("parallel_mode")
        is_auto_parallel = parallel_mode in (ParallelMode.SEMI_AUTO_PARALLEL, ParallelMode.AUTO_PARALLEL)
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

        if (host_device_mix and is_auto_parallel) or parameter_server:
            self.optimizer_d = LazyAdam(
                self.weights_d, learning_rate=3.5e-4, eps=1e-8, loss_scale=sens)
            self.optimizer_w = FTRL(learning_rate=5e-2, params=self.weights_w,
                                    l1=1e-8, l2=1e-8, initial_accum=1.0, loss_scale=sens)
            self.optimizer_w.sparse_opt.add_prim_attr("primitive_target", "CPU")
            self.optimizer_d.sparse_opt.add_prim_attr("primitive_target", "CPU")
        else:
            self.optimizer_d = Adam(
                self.weights_d, learning_rate=3.5e-4, eps=1e-8, loss_scale=sens)
            self.optimizer_w = FTRL(learning_rate=5e-2, params=self.weights_w,
                                    l1=1e-8, l2=1e-8, initial_accum=1.0, loss_scale=sens)
        self.hyper_map = C.HyperMap()
        self.grad_w = C.GradOperation(get_by_list=True,
                                      sens_param=True)
        self.grad_d = C.GradOperation(get_by_list=True,
                                      sens_param=True)
        self.sens = sens
        self.loss_net_w = IthOutputCell(network, output_index=0)
        self.loss_net_d = IthOutputCell(network, output_index=1)
        self.loss_net_w.set_grad()
        self.loss_net_d.set_grad()

        self.reducer_flag = False
        self.grad_reducer_w = None
        self.grad_reducer_d = None
        self.reducer_flag = parallel_mode in (ParallelMode.DATA_PARALLEL,
                                              ParallelMode.HYBRID_PARALLEL)
        if self.reducer_flag:
            mean = context.get_auto_parallel_context("gradients_mean")
            degree = context.get_auto_parallel_context("device_num")
            self.grad_reducer_w = DistributedGradReducer(self.optimizer_w.parameters, mean, degree)
            self.grad_reducer_d = DistributedGradReducer(self.optimizer_d.parameters, mean, degree)

    def construct(self, batch_ids, batch_wts, label):
        '''
        Construct wide and deep model
        '''
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
        logits, _, = self.network(batch_ids, batch_wts)
        pred_probs = self.sigmoid(logits)
        return logits, pred_probs, labels
