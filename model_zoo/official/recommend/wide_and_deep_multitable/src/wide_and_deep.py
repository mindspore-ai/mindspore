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

import mindspore.common.dtype as mstype
from mindspore import nn, context
from mindspore import Tensor, Parameter, ParameterTuple
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore.nn import Dropout, Flatten
from mindspore.nn.optim import Adam, FTRL
from mindspore.common.initializer import Uniform, initializer
from mindspore.context import ParallelMode
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer


np_type = np.float32
ms_type = mstype.float32


def init_method(method, shape, name, max_val=1.0):
    """
    Init method
    """
    if method in ['uniform']:
        params = Parameter(initializer(Uniform(max_val), shape, ms_type),
                           name=name)
    elif method == "one":
        params = Parameter(initializer("ones", shape, ms_type), name=name)
    elif method == 'zero':
        params = Parameter(initializer("zeros", shape, ms_type), name=name)
    elif method == "normal":
        params = Parameter(Tensor(
            np.random.normal(loc=0.0, scale=0.01,
                             size=shape).astype(dtype=np_type)),
                           name=name)
    return params


def init_var_dict(init_args, in_vars):
    """
    Init parameters by dict
    """
    var_map = {}
    _, _max_val = init_args
    for _, item in enumerate(in_vars):
        key, shape, method = item
        if key not in var_map.keys():
            if method in ['random', 'uniform']:
                var_map[key] = Parameter(initializer(Uniform(_max_val), shape,
                                                     ms_type),
                                         name=key)
            elif method == "one":
                var_map[key] = Parameter(initializer("ones", shape, ms_type),
                                         name=key)
            elif method == "zero":
                var_map[key] = Parameter(initializer("zeros", shape, ms_type),
                                         name=key)
            elif method == 'normal':
                var_map[key] = Parameter(Tensor(
                    np.random.normal(loc=0.0, scale=0.01,
                                     size=shape).astype(dtype=np_type)),
                                         name=key)
    return var_map


class DenseLayer(nn.Cell):
    """
    Dense Layer for Deep Layer of WideDeep Model;
    Containing: activation, matmul, bias_add;
    """
    def __init__(self,
                 input_dim,
                 output_dim,
                 weight_bias_init,
                 act_str,
                 keep_prob=0.8,
                 scale_coef=1.0,
                 use_activation=True,
                 convert_dtype=True,
                 drop_out=False):
        super(DenseLayer, self).__init__()
        weight_init, bias_init = weight_bias_init
        self.weight = init_method(weight_init, [input_dim, output_dim],
                                  name="weight")
        self.bias = init_method(bias_init, [output_dim], name="bias")
        self.act_func = self._init_activation(act_str)
        self.matmul = P.MatMul(transpose_b=False)
        self.bias_add = P.BiasAdd()
        self.cast = P.Cast()
        self.dropout = Dropout(keep_prob=keep_prob)
        self.mul = P.Mul()
        self.realDiv = P.RealDiv()
        self.scale_coef = scale_coef
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
        emb_128_size = 650000
        emb64_single_size = 17300
        emb64_multi_size = 20900
        indicator_size = 16
        deep_dim_list = [1024, 1024, 1024, 1024, 1024]

        wide_reg_coef = [0.0, 0.0]
        deep_reg_coef = [0.0, 0.0]
        wide_lr = 0.2
        deep_lr = 1.0

        self.input_emb_dim = config.input_emb_dim
        self.batch_size = config.batch_size
        self.deep_layer_act = config.deep_layers_act
        self.init_args = config.init_args
        self.weight_init, self.bias_init = config.weight_bias_init
        self.weight_bias_init = config.weight_bias_init
        self.emb_init = config.emb_init

        self.keep_prob = config.keep_prob
        self.layer_dims = deep_dim_list + [1]
        self.all_dim_list = [self.input_emb_dim] + self.layer_dims

        self.continue_field_size = 32
        self.emb_128_size = emb_128_size
        self.emb64_single_size = emb64_single_size
        self.emb64_multi_size = emb64_multi_size
        self.indicator_size = indicator_size

        self.wide_l1_coef, self.wide_l2_coef = wide_reg_coef
        self.deep_l1_coef, self.deep_l2_coef = deep_reg_coef
        self.wide_lr = wide_lr
        self.deep_lr = deep_lr

        init_acts_embedding_metrix = [
            ('emb128_embedding', [self.emb_128_size, 128], self.emb_init),
            ('emb64_single', [self.emb64_single_size, 64], self.emb_init),
            ('emb64_multi', [self.emb64_multi_size, 64], self.emb_init),
            ('emb64_indicator', [self.indicator_size, 64], self.emb_init)
        ]
        var_map = init_var_dict(self.init_args, init_acts_embedding_metrix)
        self.emb128_embedding = var_map["emb128_embedding"]
        self.emb64_single = var_map["emb64_single"]
        self.emb64_multi = var_map["emb64_multi"]
        self.emb64_indicator = var_map["emb64_indicator"]

        init_acts_wide_weight = [
            ('wide_continue_w', [self.continue_field_size], self.emb_init),
            ('wide_emb128_w', [self.emb_128_size], self.emb_init),
            ('wide_emb64_single_w', [self.emb64_single_size], self.emb_init),
            ('wide_emb64_multi_w', [self.emb64_multi_size], self.emb_init),
            ('wide_indicator_w', [self.indicator_size], self.emb_init),
            ('wide_bias', [1], self.emb_init)
        ]
        var_map = init_var_dict(self.init_args, init_acts_wide_weight)
        self.wide_continue_w = var_map["wide_continue_w"]
        self.wide_emb128_w = var_map["wide_emb128_w"]
        self.wide_emb64_single_w = var_map["wide_emb64_single_w"]
        self.wide_emb64_multi_w = var_map["wide_emb64_multi_w"]
        self.wide_indicator_w = var_map["wide_indicator_w"]
        self.wide_bias = var_map["wide_bias"]

        self.dense_layer_1 = DenseLayer(self.all_dim_list[0],
                                        self.all_dim_list[1],
                                        self.weight_bias_init,
                                        self.deep_layer_act,
                                        drop_out=config.dropout_flag,
                                        convert_dtype=True)
        self.dense_layer_2 = DenseLayer(self.all_dim_list[1],
                                        self.all_dim_list[2],
                                        self.weight_bias_init,
                                        self.deep_layer_act,
                                        drop_out=config.dropout_flag,
                                        convert_dtype=True)
        self.dense_layer_3 = DenseLayer(self.all_dim_list[2],
                                        self.all_dim_list[3],
                                        self.weight_bias_init,
                                        self.deep_layer_act,
                                        drop_out=config.dropout_flag,
                                        convert_dtype=True)
        self.dense_layer_4 = DenseLayer(self.all_dim_list[3],
                                        self.all_dim_list[4],
                                        self.weight_bias_init,
                                        self.deep_layer_act,
                                        drop_out=config.dropout_flag,
                                        convert_dtype=True)
        self.dense_layer_5 = DenseLayer(self.all_dim_list[4],
                                        self.all_dim_list[5],
                                        self.weight_bias_init,
                                        self.deep_layer_act,
                                        drop_out=config.dropout_flag,
                                        convert_dtype=True)

        self.deep_predict = DenseLayer(self.all_dim_list[5],
                                       self.all_dim_list[6],
                                       self.weight_bias_init,
                                       self.deep_layer_act,
                                       drop_out=config.dropout_flag,
                                       convert_dtype=True,
                                       use_activation=False)

        self.gather_v2 = P.Gather()
        self.mul = P.Mul()
        self.reduce_sum_false = P.ReduceSum(keep_dims=False)
        self.reduce_sum_true = P.ReduceSum(keep_dims=True)
        self.reshape = P.Reshape()
        self.square = P.Square()
        self.shape = P.Shape()
        self.tile = P.Tile()
        self.concat = P.Concat(axis=1)
        self.cast = P.Cast()
        self.reduceMean_false = P.ReduceMean(keep_dims=False)
        self.Concat = P.Concat(axis=1)
        self.BiasAdd = P.BiasAdd()
        self.expand_dims = P.ExpandDims()
        self.flatten = Flatten()

    def construct(self, continue_val, indicator_id, emb_128_id,
                  emb_64_single_id, multi_doc_ad_category_id,
                  multi_doc_ad_category_id_mask, multi_doc_event_entity_id,
                  multi_doc_event_entity_id_mask, multi_doc_ad_entity_id,
                  multi_doc_ad_entity_id_mask, multi_doc_event_topic_id,
                  multi_doc_event_topic_id_mask, multi_doc_event_category_id,
                  multi_doc_event_category_id_mask, multi_doc_ad_topic_id,
                  multi_doc_ad_topic_id_mask, display_id, ad_id,
                  display_ad_and_is_leak, is_leak):
        """
        Args:
            id_hldr: batch ids;
            wt_hldr: batch weights;
        """

        val_hldr = continue_val
        ind_hldr = indicator_id
        emb128_id_hldr = emb_128_id
        emb64_single_hldr = emb_64_single_id

        ind_emb = self.gather_v2(self.emb64_indicator, ind_hldr, 0)
        ind_emb = self.flatten(ind_emb)

        emb128_id_emb = self.gather_v2(self.emb128_embedding, emb128_id_hldr,
                                       0)
        emb128_id_emb = self.flatten(emb128_id_emb)

        emb64_sgl_emb = self.gather_v2(self.emb64_single, emb64_single_hldr, 0)
        emb64_sgl_emb = self.flatten(emb64_sgl_emb)

        mult_emb_1 = self.gather_v2(self.emb64_multi, multi_doc_ad_category_id,
                                    0)
        mult_emb_1 = self.mul(
            self.cast(mult_emb_1, mstype.float32),
            self.cast(self.expand_dims(multi_doc_ad_category_id_mask, 2),
                      mstype.float32))
        mult_emb_1 = self.reduceMean_false(mult_emb_1, 1)

        mult_emb_2 = self.gather_v2(self.emb64_multi,
                                    multi_doc_event_entity_id, 0)
        mult_emb_2 = self.mul(
            self.cast(mult_emb_2, mstype.float32),
            self.cast(self.expand_dims(multi_doc_event_entity_id_mask, 2),
                      mstype.float32))
        mult_emb_2 = self.reduceMean_false(mult_emb_2, 1)

        mult_emb_3 = self.gather_v2(self.emb64_multi, multi_doc_ad_entity_id,
                                    0)
        mult_emb_3 = self.mul(
            self.cast(mult_emb_3, mstype.float32),
            self.cast(self.expand_dims(multi_doc_ad_entity_id_mask, 2),
                      mstype.float32))
        mult_emb_3 = self.reduceMean_false(mult_emb_3, 1)

        mult_emb_4 = self.gather_v2(self.emb64_multi, multi_doc_event_topic_id,
                                    0)
        mult_emb_4 = self.mul(
            self.cast(mult_emb_4, mstype.float32),
            self.cast(self.expand_dims(multi_doc_event_topic_id_mask, 2),
                      mstype.float32))
        mult_emb_4 = self.reduceMean_false(mult_emb_4, 1)

        mult_emb_5 = self.gather_v2(self.emb64_multi,
                                    multi_doc_event_category_id, 0)
        mult_emb_5 = self.mul(
            self.cast(mult_emb_5, mstype.float32),
            self.cast(self.expand_dims(multi_doc_event_category_id_mask, 2),
                      mstype.float32))
        mult_emb_5 = self.reduceMean_false(mult_emb_5, 1)

        mult_emb_6 = self.gather_v2(self.emb64_multi, multi_doc_ad_topic_id, 0)
        mult_emb_6 = self.mul(
            self.cast(mult_emb_6, mstype.float32),
            self.cast(self.expand_dims(multi_doc_ad_topic_id_mask, 2),
                      mstype.float32))
        mult_emb_6 = self.reduceMean_false(mult_emb_6, 1)

        mult_embedding = self.Concat((mult_emb_1, mult_emb_2, mult_emb_3,
                                      mult_emb_4, mult_emb_5, mult_emb_6))

        input_embedding = self.Concat((val_hldr * 1, ind_emb, emb128_id_emb,
                                       emb64_sgl_emb, mult_embedding))
        deep_out = self.dense_layer_1(input_embedding)
        deep_out = self.dense_layer_2(deep_out)
        deep_out = self.dense_layer_3(deep_out)
        deep_out = self.dense_layer_4(deep_out)
        deep_out = self.dense_layer_5(deep_out)

        deep_out = self.deep_predict(deep_out)

        val_weight = self.mul(val_hldr,
                              self.expand_dims(self.wide_continue_w, 0))

        val_w_sum = self.reduce_sum_true(val_weight, 1)

        ind_weight = self.gather_v2(self.wide_indicator_w, ind_hldr, 0)
        ind_w_sum = self.reduce_sum_true(ind_weight, 1)

        emb128_id_weight = self.gather_v2(self.wide_emb128_w, emb128_id_hldr,
                                          0)
        emb128_w_sum = self.reduce_sum_true(emb128_id_weight, 1)

        emb64_sgl_weight = self.gather_v2(self.wide_emb64_single_w,
                                          emb64_single_hldr, 0)
        emb64_w_sum = self.reduce_sum_true(emb64_sgl_weight, 1)

        mult_weight_1 = self.gather_v2(self.wide_emb64_multi_w,
                                       multi_doc_ad_category_id, 0)
        mult_weight_1 = self.mul(
            self.cast(mult_weight_1, mstype.float32),
            self.cast(multi_doc_ad_category_id_mask, mstype.float32))
        mult_weight_1 = self.reduce_sum_true(mult_weight_1, 1)

        mult_weight_2 = self.gather_v2(self.wide_emb64_multi_w,
                                       multi_doc_event_entity_id, 0)
        mult_weight_2 = self.mul(
            self.cast(mult_weight_2, mstype.float32),
            self.cast(multi_doc_event_entity_id_mask, mstype.float32))
        mult_weight_2 = self.reduce_sum_true(mult_weight_2, 1)

        mult_weight_3 = self.gather_v2(self.wide_emb64_multi_w,
                                       multi_doc_ad_entity_id, 0)
        mult_weight_3 = self.mul(
            self.cast(mult_weight_3, mstype.float32),
            self.cast(multi_doc_ad_entity_id_mask, mstype.float32))
        mult_weight_3 = self.reduce_sum_true(mult_weight_3, 1)

        mult_weight_4 = self.gather_v2(self.wide_emb64_multi_w,
                                       multi_doc_event_topic_id, 0)
        mult_weight_4 = self.mul(
            self.cast(mult_weight_4, mstype.float32),
            self.cast(multi_doc_event_topic_id_mask, mstype.float32))
        mult_weight_4 = self.reduce_sum_true(mult_weight_4, 1)

        mult_weight_5 = self.gather_v2(self.wide_emb64_multi_w,
                                       multi_doc_event_category_id, 0)
        mult_weight_5 = self.mul(
            self.cast(mult_weight_5, mstype.float32),
            self.cast(multi_doc_event_category_id_mask, mstype.float32))
        mult_weight_5 = self.reduce_sum_true(mult_weight_5, 1)

        mult_weight_6 = self.gather_v2(self.wide_emb64_multi_w,
                                       multi_doc_ad_topic_id, 0)

        mult_weight_6 = self.mul(
            self.cast(mult_weight_6, mstype.float32),
            self.cast(multi_doc_ad_topic_id_mask, mstype.float32))
        mult_weight_6 = self.reduce_sum_true(mult_weight_6, 1)

        mult_weight_sum = mult_weight_1 + mult_weight_2 + mult_weight_3 + mult_weight_4 + mult_weight_5 + mult_weight_6

        wide_out = self.BiasAdd(
            val_w_sum + ind_w_sum + emb128_w_sum + emb64_w_sum +
            mult_weight_sum, self.wide_bias)

        out = wide_out + deep_out
        return out, self.emb128_embedding, self.emb64_single, self.emb64_multi


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
        self.square = P.Square()
        self.reduceMean_false = P.ReduceMean(keep_dims=False)
        self.reduceSum_false = P.ReduceSum(keep_dims=False)
        self.reshape = P.Reshape()

    def construct(self, label, continue_val, indicator_id, emb_128_id,
                  emb_64_single_id, multi_doc_ad_category_id,
                  multi_doc_ad_category_id_mask, multi_doc_event_entity_id,
                  multi_doc_event_entity_id_mask, multi_doc_ad_entity_id,
                  multi_doc_ad_entity_id_mask, multi_doc_event_topic_id,
                  multi_doc_event_topic_id_mask, multi_doc_event_category_id,
                  multi_doc_event_category_id_mask, multi_doc_ad_topic_id,
                  multi_doc_ad_topic_id_mask, display_id, ad_id,
                  display_ad_and_is_leak, is_leak):
        """
        NetWithLossClass construct
        """
        # emb128_embedding, emb64_single, emb64_multi
        predict, _, _, _ = self.network(
            continue_val, indicator_id, emb_128_id, emb_64_single_id,
            multi_doc_ad_category_id, multi_doc_ad_category_id_mask,
            multi_doc_event_entity_id, multi_doc_event_entity_id_mask,
            multi_doc_ad_entity_id, multi_doc_ad_entity_id_mask,
            multi_doc_event_topic_id, multi_doc_event_topic_id_mask,
            multi_doc_event_category_id, multi_doc_event_category_id_mask,
            multi_doc_ad_topic_id, multi_doc_ad_topic_id_mask, display_id,
            ad_id, display_ad_and_is_leak, is_leak)

        predict = self.reshape(predict, (-1,))
        basic_loss = self.loss(predict, label)
        wide_loss = self.reduceMean_false(basic_loss)
        deep_loss = self.reduceMean_false(basic_loss)
        return wide_loss, deep_loss


class IthOutputCell(nn.Cell):
    """
    IthOutputCell
    """
    def __init__(self, network, output_index):
        super(IthOutputCell, self).__init__()
        self.network = network
        self.output_index = output_index

    def construct(self, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13,
                  x14, x15, x16, x17, x18, x19, x20, x21):
        """
        IthOutputCell construct
        """
        predict = self.network(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11,
                               x12, x13, x14, x15, x16, x17, x18, x19, x20,
                               x21)[self.output_index]
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
    def __init__(self, network, config, sens=1000.0):
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
        self.optimizer_w = FTRL(learning_rate=config.ftrl_lr,
                                params=self.weights_w,
                                l1=5e-4,
                                l2=5e-4,
                                initial_accum=0.1,
                                loss_scale=sens)

        self.optimizer_d = Adam(self.weights_d,
                                learning_rate=config.adam_lr,
                                eps=1e-6,
                                loss_scale=sens)

        self.hyper_map = C.HyperMap()

        self.grad_w = C.GradOperation(get_by_list=True,
                                      sens_param=True)
        self.grad_d = C.GradOperation(get_by_list=True,
                                      sens_param=True)

        self.sens = sens
        self.loss_net_w = IthOutputCell(network, output_index=0)
        self.loss_net_d = IthOutputCell(network, output_index=1)
        self.loss_net_w.set_grad()
        self.loss_net_w.set_grad()

        self.reducer_flag = False
        self.grad_reducer_w = None
        self.grad_reducer_d = None
        parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if parallel_mode in (ParallelMode.DATA_PARALLEL,
                             ParallelMode.HYBRID_PARALLEL):
            self.reducer_flag = True
        if self.reducer_flag:
            mean = context.get_auto_parallel_context("gradients_mean")
            degree = context.get_auto_parallel_context("device_num")
            self.grad_reducer_w = DistributedGradReducer(
                self.optimizer_w.parameters, mean, degree)
            self.grad_reducer_d = DistributedGradReducer(
                self.optimizer_d.parameters, mean, degree)

    def construct(self, label, continue_val, indicator_id, emb_128_id,
                  emb_64_single_id, multi_doc_ad_category_id,
                  multi_doc_ad_category_id_mask, multi_doc_event_entity_id,
                  multi_doc_event_entity_id_mask, multi_doc_ad_entity_id,
                  multi_doc_ad_entity_id_mask, multi_doc_event_topic_id,
                  multi_doc_event_topic_id_mask, multi_doc_event_category_id,
                  multi_doc_event_category_id_mask, multi_doc_ad_topic_id,
                  multi_doc_ad_topic_id_mask, display_id, ad_id,
                  display_ad_and_is_leak, is_leak):
        """
        TrainStepWrap construct
        """
        weights_w = self.weights_w
        weights_d = self.weights_d
        loss_w, loss_d = self.network(
            label, continue_val, indicator_id, emb_128_id, emb_64_single_id,
            multi_doc_ad_category_id, multi_doc_ad_category_id_mask,
            multi_doc_event_entity_id, multi_doc_event_entity_id_mask,
            multi_doc_ad_entity_id, multi_doc_ad_entity_id_mask,
            multi_doc_event_topic_id, multi_doc_event_topic_id_mask,
            multi_doc_event_category_id, multi_doc_event_category_id_mask,
            multi_doc_ad_topic_id, multi_doc_ad_topic_id_mask, display_id,
            ad_id, display_ad_and_is_leak, is_leak)

        sens_w = P.Fill()(P.DType()(loss_w), P.Shape()(loss_w), self.sens)  #
        sens_d = P.Fill()(P.DType()(loss_d), P.Shape()(loss_d), self.sens)  #
        grads_w = self.grad_w(self.loss_net_w, weights_w)(
            label, continue_val, indicator_id, emb_128_id, emb_64_single_id,
            multi_doc_ad_category_id, multi_doc_ad_category_id_mask,
            multi_doc_event_entity_id, multi_doc_event_entity_id_mask,
            multi_doc_ad_entity_id, multi_doc_ad_entity_id_mask,
            multi_doc_event_topic_id, multi_doc_event_topic_id_mask,
            multi_doc_event_category_id, multi_doc_event_category_id_mask,
            multi_doc_ad_topic_id, multi_doc_ad_topic_id_mask, display_id,
            ad_id, display_ad_and_is_leak, is_leak, sens_w)
        grads_d = self.grad_d(self.loss_net_d, weights_d)(
            label, continue_val, indicator_id, emb_128_id, emb_64_single_id,
            multi_doc_ad_category_id, multi_doc_ad_category_id_mask,
            multi_doc_event_entity_id, multi_doc_event_entity_id_mask,
            multi_doc_ad_entity_id, multi_doc_ad_entity_id_mask,
            multi_doc_event_topic_id, multi_doc_event_topic_id_mask,
            multi_doc_event_category_id, multi_doc_event_category_id_mask,
            multi_doc_ad_topic_id, multi_doc_ad_topic_id_mask, display_id,
            ad_id, display_ad_and_is_leak, is_leak, sens_d)
        if self.reducer_flag:
            # apply grad reducer on grads
            grads_w = self.grad_reducer_w(grads_w)
            grads_d = self.grad_reducer_d(grads_d)
        return F.depend(loss_w, self.optimizer_w(grads_w)), F.depend(
            loss_d, self.optimizer_d(grads_d))


class PredictWithSigmoid(nn.Cell):
    """
    PredictWithSigomid
    """
    def __init__(self, network):
        super(PredictWithSigmoid, self).__init__()
        self.network = network
        self.sigmoid = P.Sigmoid()
        self.reshape = P.Reshape()

    def construct(self, label, continue_val, indicator_id, emb_128_id,
                  emb_64_single_id, multi_doc_ad_category_id,
                  multi_doc_ad_category_id_mask, multi_doc_event_entity_id,
                  multi_doc_event_entity_id_mask, multi_doc_ad_entity_id,
                  multi_doc_ad_entity_id_mask, multi_doc_event_topic_id,
                  multi_doc_event_topic_id_mask, multi_doc_event_category_id,
                  multi_doc_event_category_id_mask, multi_doc_ad_topic_id,
                  multi_doc_ad_topic_id_mask, display_id, ad_id,
                  display_ad_and_is_leak, is_leak):
        """
        PredictWithSigomid construct
        """
        logits, _, _, _ = self.network(
            continue_val, indicator_id, emb_128_id, emb_64_single_id,
            multi_doc_ad_category_id, multi_doc_ad_category_id_mask,
            multi_doc_event_entity_id, multi_doc_event_entity_id_mask,
            multi_doc_ad_entity_id, multi_doc_ad_entity_id_mask,
            multi_doc_event_topic_id, multi_doc_event_topic_id_mask,
            multi_doc_event_category_id, multi_doc_event_category_id_mask,
            multi_doc_ad_topic_id, multi_doc_ad_topic_id_mask, display_id,
            ad_id, display_ad_and_is_leak, is_leak)
        logits = self.reshape(logits, (-1,))
        pred_probs = self.sigmoid(logits)
        return logits, pred_probs, label, display_id
