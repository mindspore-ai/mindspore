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
""" test_training """
import os

import numpy as np
from sklearn.metrics import roc_auc_score
import mindspore.common.dtype as mstype
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore.nn import Dropout
from mindspore.nn.optim import Adam
from mindspore.train import Metric
from mindspore import nn, Tensor, ParameterTuple, Parameter
from mindspore.common.initializer import Uniform, initializer
from mindspore.train import ModelCheckpoint, CheckpointConfig
from mindspore.parallel._utils import _get_device_num, _get_parallel_mode, _get_gradients_mean
from mindspore.context import ParallelMode
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer

from .callback import EvalCallBack, LossCallBack


np_type = np.float32
ms_type = mstype.float32


class AUCMetric(Metric):
    """AUC metric for DeepFM model."""
    def __init__(self):
        super(AUCMetric, self).__init__()
        self.pred_probs = []
        self.true_labels = []

    def clear(self):
        """Clear the internal evaluation result."""
        self.pred_probs = []
        self.true_labels = []

    def update(self, *inputs):
        batch_predict = inputs[1].asnumpy()
        batch_label = inputs[2].asnumpy()
        self.pred_probs.extend(batch_predict.flatten().tolist())
        self.true_labels.extend(batch_label.flatten().tolist())

    def eval(self):
        if len(self.true_labels) != len(self.pred_probs):
            raise RuntimeError('true_labels.size() is not equal to pred_probs.size()')
        auc = roc_auc_score(self.true_labels, self.pred_probs)
        return auc


def init_method(method, shape, name, max_val=1.0):
    """
    The method of init parameters.

    Args:
        method (str): The method uses to initialize parameter.
        shape (list): The shape of parameter.
        name (str): The name of parameter.
        max_val (float): Max value in parameter when uses 'random' or 'uniform' to initialize parameter.

    Returns:
        Parameter.
    """
    if method in ['uniform']:
        params = Parameter(initializer(Uniform(max_val), shape, ms_type), name=name)
    elif method == "one":
        params = Parameter(initializer("ones", shape, ms_type), name=name)
    elif method == 'zero':
        params = Parameter(initializer("zeros", shape, ms_type), name=name)
    elif method == "normal":
        params = Parameter(Tensor(np.random.normal(loc=0.0, scale=0.01, size=shape).astype(dtype=np_type)), name=name)
    return params


def init_var_dict(init_args, var_list):
    """
    Init parameter.

    Args:
        init_args (list): Define max and min value of parameters.
        values (list): Define name, shape and init method of parameters.

    Returns:
        dict, a dict ot Parameter.
    """
    var_map = {}
    _, max_val = init_args
    for i, _ in enumerate(var_list):
        key, shape, method = var_list[i]
        if key not in var_map.keys():
            if method in ['random', 'uniform']:
                var_map[key] = Parameter(initializer(Uniform(max_val), shape, ms_type), name=key)
            elif method == "one":
                var_map[key] = Parameter(initializer("ones", shape, ms_type), name=key)
            elif method == "zero":
                var_map[key] = Parameter(initializer("zeros", shape, ms_type), name=key)
            elif method == 'normal':
                var_map[key] = Parameter(Tensor(np.random.normal(loc=0.0, scale=0.01, size=shape).
                                                astype(dtype=np_type)), name=key)
    return var_map


class DenseLayer(nn.Cell):
    """
    Dense Layer for Deep Layer of DeepFM Model;
    Containing: activation, matmul, bias_add;
    Args:
        input_dim (int): the shape of weight at 0-axis;
        output_dim (int): the shape of weight at 1-axis, and shape of bias
        weight_bias_init (list): weight and bias init method, "random", "uniform", "one", "zero", "normal";
        act_str (str): activation function method, "relu", "sigmoid", "tanh";
        keep_prob (float): Dropout Layer keep_prob_rate;
        scale_coef (float): input scale coefficient;
    """

    def __init__(self, input_dim, output_dim, weight_bias_init, act_str, scale_coef=1.0, convert_dtype=True,
                 use_act=True):
        super(DenseLayer, self).__init__()
        weight_init, bias_init = weight_bias_init
        self.weight = init_method(weight_init, [input_dim, output_dim], name="weight")
        self.bias = init_method(bias_init, [output_dim], name="bias")
        self.act_func = self._init_activation(act_str)
        self.matmul = P.MatMul(transpose_b=False)
        self.bias_add = P.BiasAdd()
        self.cast = P.Cast()
        self.dropout = Dropout(p=0.0)
        self.mul = P.Mul()
        self.realDiv = P.RealDiv()
        self.scale_coef = scale_coef
        self.convert_dtype = convert_dtype
        self.use_act = use_act

    def _init_activation(self, act_str):
        """Init activation function"""
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
        x = self.dropout(x)
        if self.convert_dtype:
            x = self.cast(x, mstype.float16)
            weight = self.cast(self.weight, mstype.float16)
            bias = self.cast(self.bias, mstype.float16)
            wx = self.matmul(x, weight)
            wx = self.bias_add(wx, bias)
            if self.use_act:
                wx = self.act_func(wx)
            wx = self.cast(wx, mstype.float32)
        else:
            wx = self.matmul(x, self.weight)
            wx = self.bias_add(wx, self.bias)
            if self.use_act:
                wx = self.act_func(wx)
        return wx


class DeepFMModel(nn.Cell):
    """
    From paper: "DeepFM: A Factorization-Machine based Neural Network for CTR Prediction"

    Args:
        batch_size (int):  smaple_number of per step in training; (int, batch_size=128)
        filed_size (int):  input filed number, or called id_feature number; (int, filed_size=39)
        vocab_size (int):  id_feature vocab size, id dict size;  (int, vocab_size=200000)
        emb_dim (int):  id embedding vector dim, id mapped to embedding vector; (int, emb_dim=100)
        deep_layer_args (list):  Deep Layer args, layer_dim_list, layer_activator;
                             (int, deep_layer_args=[[100, 100, 100], "relu"])
        init_args (list): init args for Parameter init; (list, init_args=[min, max, seeds])
        weight_bias_init (list): weight, bias init method for deep layers;
                            (list[str], weight_bias_init=['random', 'zero'])
        keep_prob (float): if dropout_flag is True, keep_prob rate to keep connect; (float, keep_prob=0.8)
    """

    def __init__(self, config):
        super(DeepFMModel, self).__init__()

        self.batch_size = config.batch_size
        self.field_size = config.data_field_size
        self.vocab_size = config.data_vocab_size
        self.emb_dim = config.data_emb_dim
        self.deep_layer_dims_list, self.deep_layer_act = config.deep_layer_args
        self.init_args = config.init_args
        self.weight_bias_init = config.weight_bias_init
        self.keep_prob = config.keep_prob
        init_acts = [('W_l2', [self.vocab_size, 1], 'normal'),
                     ('V_l2', [self.vocab_size, self.emb_dim], 'normal')]
        var_map = init_var_dict(self.init_args, init_acts)
        self.fm_w = var_map["W_l2"]
        self.embedding_table = var_map["V_l2"]
        " Deep Layers "
        self.deep_input_dims = self.field_size * self.emb_dim
        self.all_dim_list = [self.deep_input_dims] + self.deep_layer_dims_list + [1]
        self.dense_layer_1 = DenseLayer(self.all_dim_list[0], self.all_dim_list[1], self.weight_bias_init,
                                        self.deep_layer_act, self.keep_prob, convert_dtype=True)
        self.dense_layer_2 = DenseLayer(self.all_dim_list[1], self.all_dim_list[2], self.weight_bias_init,
                                        self.deep_layer_act, self.keep_prob, convert_dtype=True)
        self.dense_layer_3 = DenseLayer(self.all_dim_list[2], self.all_dim_list[3], self.weight_bias_init,
                                        self.deep_layer_act, self.keep_prob, convert_dtype=True)
        self.dense_layer_4 = DenseLayer(self.all_dim_list[3], self.all_dim_list[4], self.weight_bias_init,
                                        self.deep_layer_act, self.keep_prob, convert_dtype=True)
        self.dense_layer_5 = DenseLayer(self.all_dim_list[4], self.all_dim_list[5], self.weight_bias_init,
                                        self.deep_layer_act, self.keep_prob, convert_dtype=True, use_act=False)
        " FM, linear Layers "
        self.Gatherv2 = P.Gather()
        self.Mul = P.Mul()
        self.ReduceSum = P.ReduceSum(keep_dims=False)
        self.Reshape = P.Reshape()
        self.Square = P.Square()
        self.Shape = P.Shape()
        self.Tile = P.Tile()
        self.Concat = P.Concat(axis=1)
        self.Cast = P.Cast()

    def construct(self, id_hldr, wt_hldr):
        """
        Args:
            id_hldr: batch ids;   [bs, field_size]
            wt_hldr: batch weights;   [bs, field_size]
        """

        mask = self.Reshape(wt_hldr, (self.batch_size, self.field_size, 1))
        # Linear layer
        fm_id_weight = self.Gatherv2(self.fm_w, id_hldr, 0)
        wx = self.Mul(fm_id_weight, mask)
        linear_out = self.ReduceSum(wx, 1)
        # FM layer
        fm_id_embs = self.Gatherv2(self.embedding_table, id_hldr, 0)
        vx = self.Mul(fm_id_embs, mask)
        v1 = self.ReduceSum(vx, 1)
        v1 = self.Square(v1)
        v2 = self.Square(vx)
        v2 = self.ReduceSum(v2, 1)
        fm_out = 0.5 * self.ReduceSum(v1 - v2, 1)
        fm_out = self.Reshape(fm_out, (-1, 1))
        #  Deep layer
        deep_in = self.Reshape(vx, (-1, self.field_size * self.emb_dim))
        deep_in = self.dense_layer_1(deep_in)
        deep_in = self.dense_layer_2(deep_in)
        deep_in = self.dense_layer_3(deep_in)
        deep_in = self.dense_layer_4(deep_in)
        deep_out = self.dense_layer_5(deep_in)
        out = linear_out + fm_out + deep_out
        return out, self.fm_w, self.embedding_table


class NetWithLossClass(nn.Cell):
    """
    NetWithLossClass definition.
    """
    def __init__(self, network, l2_coef=1e-6):
        super(NetWithLossClass, self).__init__(auto_prefix=False)
        self.loss = P.SigmoidCrossEntropyWithLogits()
        self.network = network
        self.l2_coef = l2_coef
        self.Square = P.Square()
        self.ReduceMean_false = P.ReduceMean(keep_dims=False)
        self.ReduceSum_false = P.ReduceSum(keep_dims=False)

    def construct(self, batch_ids, batch_wts, label):
        predict, fm_id_weight, fm_id_embs = self.network(batch_ids, batch_wts)
        log_loss = self.loss(predict, label)
        mean_log_loss = self.ReduceMean_false(log_loss)
        l2_loss_w = self.ReduceSum_false(self.Square(fm_id_weight))
        l2_loss_v = self.ReduceSum_false(self.Square(fm_id_embs))
        l2_loss_all = self.l2_coef * (l2_loss_v + l2_loss_w) * 0.5
        loss = mean_log_loss + l2_loss_all
        return loss


class TrainStepWrap(nn.Cell):
    """
    TrainStepWrap definition
    """
    def __init__(self, network, lr, eps, loss_scale=1000.0):
        super(TrainStepWrap, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_train()
        self.weights = ParameterTuple(network.trainable_params())
        self.optimizer = Adam(self.weights, learning_rate=lr, eps=eps, loss_scale=loss_scale)
        self.hyper_map = C.HyperMap()
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.sens = loss_scale

        self.reducer_flag = False
        self.grad_reducer = None
        parallel_mode = _get_parallel_mode()
        if parallel_mode in (ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL):
            self.reducer_flag = True
        if self.reducer_flag:
            mean = _get_gradients_mean()
            degree = _get_device_num()
            self.grad_reducer = DistributedGradReducer(self.optimizer.parameters, mean, degree)

    def construct(self, batch_ids, batch_wts, label):
        weights = self.weights
        loss = self.network(batch_ids, batch_wts, label)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)  #
        grads = self.grad(self.network, weights)(batch_ids, batch_wts, label, sens)
        if self.reducer_flag:
            # apply grad reducer on grads
            grads = self.grad_reducer(grads)
        self.optimizer(grads)
        return loss


class PredictWithSigmoid(nn.Cell):
    """
    Eval model with sigmoid.
    """
    def __init__(self, network):
        super(PredictWithSigmoid, self).__init__(auto_prefix=False)
        self.network = network
        self.sigmoid = P.Sigmoid()

    def construct(self, batch_ids, batch_wts, labels):
        logits, _, _, = self.network(batch_ids, batch_wts)
        pred_probs = self.sigmoid(logits)

        return logits, pred_probs, labels


class ModelBuilder:
    """
    Model builder for DeepFM.

    Args:
        model_config (ModelConfig): Model configuration.
        train_config (TrainConfig): Train configuration.
    """
    def __init__(self, model_config, train_config):
        self.model_config = model_config
        self.train_config = train_config

    def get_callback_list(self, model=None, eval_dataset=None):
        """
        Get callbacks which contains checkpoint callback, eval callback and loss callback.

        Args:
            model (Cell): The network is added callback (default=None).
            eval_dataset (Dataset): Dataset for eval (default=None).
        """
        callback_list = []
        if self.train_config.save_checkpoint:
            config_ck = CheckpointConfig(save_checkpoint_steps=self.train_config.save_checkpoint_steps,
                                         keep_checkpoint_max=self.train_config.keep_checkpoint_max)
            ckpt_cb = ModelCheckpoint(prefix=self.train_config.ckpt_file_name_prefix,
                                      directory=self.train_config.output_path,
                                      config=config_ck)
            callback_list.append(ckpt_cb)
        if self.train_config.eval_callback:
            if model is None:
                raise RuntimeError("train_config.eval_callback is {}; get_callback_list() args model is {}".format(
                                        self.train_config.eval_callback, model))
            if eval_dataset is None:
                raise RuntimeError("train_config.eval_callback is {}; get_callback_list() "
                                   "args eval_dataset is {}".format(self.train_config.eval_callback, eval_dataset))
            auc_metric = AUCMetric()
            eval_callback = EvalCallBack(model, eval_dataset, auc_metric,
                                         eval_file_path=os.path.join(self.train_config.output_path,
                                                                     self.train_config.eval_file_name))
            callback_list.append(eval_callback)
        if self.train_config.loss_callback:
            loss_callback = LossCallBack(loss_file_path=os.path.join(self.train_config.output_path,
                                                                     self.train_config.loss_file_name))
            callback_list.append(loss_callback)
        if callback_list:
            return callback_list
        return None

    def get_train_eval_net(self):
        deepfm_net = DeepFMModel(self.model_config)
        loss_net = NetWithLossClass(deepfm_net, l2_coef=self.train_config.l2_coef)
        train_net = TrainStepWrap(loss_net, lr=self.train_config.learning_rate,
                                  eps=self.train_config.epsilon,
                                  loss_scale=self.train_config.loss_scale)
        eval_net = PredictWithSigmoid(deepfm_net)
        return train_net, eval_net
