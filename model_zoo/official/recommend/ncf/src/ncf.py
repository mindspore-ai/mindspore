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
"""Neural Collaborative Filtering Model"""
from mindspore import nn
from mindspore import Tensor, Parameter, ParameterTuple
from mindspore._checkparam import Validator as validator
from mindspore.nn.layer.activation import get_activation
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P
from mindspore.common.initializer import initializer
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore.parallel._utils import _get_device_num, _get_parallel_mode, _get_gradients_mean
from mindspore.context import ParallelMode
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer

from src.lr_schedule import dynamic_lr

class DenseLayer(nn.Cell):
    """
    Dense layer definition
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 weight_init='normal',
                 bias_init='zeros',
                 has_bias=True,
                 activation=None):
        super(DenseLayer, self).__init__()
        self.in_channels = validator.check_positive_int(in_channels)
        self.out_channels = validator.check_positive_int(out_channels)
        self.has_bias = validator.check_bool(has_bias)

        if isinstance(weight_init, Tensor):
            if weight_init.ndim != 2 or weight_init.shape()[0] != out_channels or \
                    weight_init.shape()[1] != in_channels:
                raise ValueError("weight_init shape error")

        self.weight = Parameter(initializer(weight_init, [out_channels, in_channels]))

        if self.has_bias:
            if isinstance(bias_init, Tensor):
                if bias_init.ndim != 1 or bias_init.shape()[0] != out_channels:
                    raise ValueError("bias_init shape error")

            self.bias = Parameter(initializer(bias_init, [out_channels]))

        self.matmul = P.MatMul(transpose_b=True)
        self.bias_add = P.BiasAdd()
        self.cast = P.Cast()

        self.activation = get_activation(activation)
        self.activation_flag = self.activation is not None

    def construct(self, x):
        """
        dense layer construct method
        """
        x = self.cast(x, mstype.float16)
        weight = self.cast(self.weight, mstype.float16)
        bias = self.cast(self.bias, mstype.float16)

        output = self.matmul(x, weight)
        if self.has_bias:
            output = self.bias_add(output, bias)
        if self.activation_flag:
            output = self.activation(output)
        output = self.cast(output, mstype.float32)
        return output

    def extend_repr(self):
        """A pretty print for Dense layer."""
        str_info = 'in_channels={}, out_channels={}, weight={}, has_bias={}' \
            .format(self.in_channels, self.out_channels, self.weight, self.has_bias)
        if self.has_bias:
            str_info = str_info + ', bias={}'.format(self.bias)

        if self.activation_flag:
            str_info = str_info + ', activation={}'.format(self.activation)

        return str_info


class NCFModel(nn.Cell):
    """
        Class for Neural Collaborative Filtering Model from paper " Neural Collaborative Filtering".
    """

    def __init__(self,
                 num_users,
                 num_items,
                 num_factors,
                 model_layers,
                 mf_regularization,
                 mlp_reg_layers,
                 mf_dim):
        super(NCFModel, self).__init__()

        self.data_path = ""
        self.model_path = ""

        self.num_users = num_users
        self.num_items = num_items
        self.num_factors = num_factors
        self.model_layers = model_layers

        self.mf_regularization = mf_regularization
        self.mlp_reg_layers = mlp_reg_layers

        self.mf_dim = mf_dim

        self.num_layers = len(self.model_layers)  # Number of layers in the MLP

        if self.model_layers[0] % 2 != 0:
            raise ValueError("The first layer size should be multiple of 2!")

        # Initializer for embedding layers
        self.embedding_initializer = "normal"

        self.embedding_user = nn.Embedding(
            self.num_users,
            self.num_factors + self.model_layers[0] // 2,
            embedding_table=self.embedding_initializer
        )
        self.embedding_item = nn.Embedding(
            self.num_items,
            self.num_factors + self.model_layers[0] // 2,
            embedding_table=self.embedding_initializer
        )

        self.mlp_dense1 = DenseLayer(in_channels=self.model_layers[0],
                                     out_channels=self.model_layers[1],
                                     activation="relu")
        self.mlp_dense2 = DenseLayer(in_channels=self.model_layers[1],
                                     out_channels=self.model_layers[2],
                                     activation="relu")

        # Logit dense layer
        self.logits_dense = DenseLayer(in_channels=self.model_layers[1],
                                       out_channels=1,
                                       weight_init="normal",
                                       activation=None)

        # ops definition
        self.mul = P.Mul()
        self.squeeze = P.Squeeze(axis=1)
        self.concat = P.Concat(axis=1)

    def construct(self, user_input, item_input):
        """
        NCF construct method.
        """
        # GMF part
        # embedding_layers
        embedding_user = self.embedding_user(user_input)  # input: (256, 1)  output: (256, 1, 16 + 32)
        embedding_item = self.embedding_item(item_input)  # input: (256, 1)  output: (256, 1, 16 + 32)

        mf_user_latent = self.squeeze(embedding_user)[:, :self.num_factors]  # input: (256, 1, 16 + 32) output: (256, 16)
        mf_item_latent = self.squeeze(embedding_item)[:, :self.num_factors]  # input: (256, 1, 16 + 32) output: (256, 16)

        # MLP part
        mlp_user_latent = self.squeeze(embedding_user)[:, self.mf_dim:]  # input: (256, 1, 16 + 32) output: (256, 32)
        mlp_item_latent = self.squeeze(embedding_item)[:, self.mf_dim:]  # input: (256, 1, 16 + 32) output: (256, 32)

        # Element-wise multiply
        mf_vector = self.mul(mf_user_latent, mf_item_latent)  # input: (256, 16), (256, 16) output: (256, 16)

        # Concatenation of two latent features
        mlp_vector = self.concat((mlp_user_latent, mlp_item_latent))  # input: (256, 32), (256, 32) output: (256, 64)

        # MLP dense layers
        mlp_vector = self.mlp_dense1(mlp_vector)  # input: (256, 64) output: (256, 32)
        mlp_vector = self.mlp_dense2(mlp_vector)  # input: (256, 32) output: (256, 16)

        # # Concatenate GMF and MLP parts
        predict_vector = self.concat((mf_vector, mlp_vector))  # input: (256, 16), (256, 16)  output: (256, 32)

        # Final prediction layer
        logits = self.logits_dense(predict_vector)  # input: (256, 32)  output: (256, 1)

        # Print model topology.
        return logits


class NetWithLossClass(nn.Cell):
    """
    NetWithLossClass definition
    """
    def __init__(self, network):
        super(NetWithLossClass, self).__init__(auto_prefix=False)
        self.loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
        self.network = network
        self.reducesum = P.ReduceSum(keep_dims=False)
        self.mul = P.Mul()
        self.squeeze = P.Squeeze(axis=1)
        self.zeroslike = P.ZerosLike()
        self.concat = P.Concat(axis=1)
        self.reciprocal = P.Reciprocal()

    def construct(self, batch_users, batch_items, labels, valid_pt_mask):
        predict = self.network(batch_users, batch_items)
        predict = self.concat((self.zeroslike(predict), predict))
        labels = self.squeeze(labels)
        loss = self.loss(predict, labels)
        loss = self.mul(loss, self.squeeze(valid_pt_mask))
        mean_loss = self.mul(self.reducesum(loss), self.reciprocal(self.reducesum(valid_pt_mask)))
        return mean_loss


class TrainStepWrap(nn.Cell):
    """
    TrainStepWrap definition
    """
    def __init__(self, network, total_steps=1, sens=16384.0):
        super(TrainStepWrap, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_train()
        self.network.add_flags(defer_inline=True)
        self.weights = ParameterTuple(network.trainable_params())

        lr = dynamic_lr(0.01, total_steps, 5000)
        self.optimizer = nn.Adam(self.weights,
                                 learning_rate=lr,
                                 beta1=0.9,
                                 beta2=0.999,
                                 eps=1e-8,
                                 loss_scale=sens)

        self.hyper_map = C.HyperMap()
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens

        self.reducer_flag = False
        self.grad_reducer = None
        parallel_mode = _get_parallel_mode()
        if parallel_mode in (ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL):
            self.reducer_flag = True
        if self.reducer_flag:
            mean = _get_gradients_mean()
            degree = _get_device_num()
            self.grad_reducer = DistributedGradReducer(self.optimizer.parameters, mean, degree)


    def construct(self, batch_users, batch_items, labels, valid_pt_mask):
        weights = self.weights
        loss = self.network(batch_users, batch_items, labels, valid_pt_mask)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)  #
        grads = self.grad(self.network, weights)(batch_users, batch_items, labels, valid_pt_mask, sens)
        if self.reducer_flag:
            # apply grad reducer on grads
            grads = self.grad_reducer(grads)
        return F.depend(loss, self.optimizer(grads))


class PredictWithSigmoid(nn.Cell):
    """
    Predict definition
    """
    def __init__(self, network, k, num_eval_neg):
        super(PredictWithSigmoid, self).__init__()
        self.network = network
        self.topk = P.TopK(sorted=True)
        self.squeeze = P.Squeeze()
        self.k = k
        self.num_eval_neg = num_eval_neg
        self.gather = P.Gather()
        self.reshape = P.Reshape()
        self.reducesum = P.ReduceSum(keep_dims=False)
        self.notequal = P.NotEqual()

    def construct(self, batch_users, batch_items, duplicated_masks):
        predicts = self.network(batch_users, batch_items)  # (bs, 1)
        predicts = self.reshape(predicts, (-1, self.num_eval_neg + 1))  # (num_user, 100)
        batch_items = self.reshape(batch_items, (-1, self.num_eval_neg + 1))  # (num_user, 100)
        duplicated_masks = self.reshape(duplicated_masks, (-1, self.num_eval_neg + 1))  # (num_user, 100)
        masks_sum = self.reducesum(duplicated_masks, 1)
        metric_weights = self.notequal(masks_sum, self.num_eval_neg)  # (num_user)
        _, indices = self.topk(predicts, self.k)  # (num_user, k)

        return indices, batch_items, metric_weights
