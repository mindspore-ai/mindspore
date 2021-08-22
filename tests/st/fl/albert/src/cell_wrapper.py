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

from mindspore import nn
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore.common.tensor import Tensor
from mindspore.common import dtype as mstype
from mindspore.common.parameter import ParameterTuple


class ClipByNorm(nn.Cell):
    """
    Clips tensor values to a maximum :math:`L_2`-norm.
    """

    def __init__(self):
        super(ClipByNorm, self).__init__()
        self.reduce_sum = P.ReduceSum(keep_dims=True)
        self.select_ = P.Select()
        self.greater_ = P.Greater()
        self.cast = P.Cast()
        self.sqrt = P.Sqrt()
        self.max_op = P.Maximum()
        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.fill = P.Fill()
        self.expand_dims = P.ExpandDims()
        self.dtype = P.DType()

    def construct(self, x, clip_norm):
        """add ms_function decorator for pynative mode"""
        mul_x = F.square(x)
        if mul_x.shape == (1,):
            l2sum = self.cast(mul_x, mstype.float32)
        else:
            l2sum = self.cast(self.reduce_sum(mul_x), mstype.float32)
        cond = self.greater_(l2sum, 0)
        ones_ = self.fill(self.dtype(cond), self.shape(cond), 1.0)
        l2sum_safe = self.select_(cond, l2sum, self.cast(ones_, self.dtype(l2sum)))
        l2norm = self.select_(cond, self.sqrt(l2sum_safe), l2sum)

        intermediate = x * clip_norm

        max_norm = self.max_op(l2norm, clip_norm)
        values_clip = self.cast(intermediate, mstype.float32) / self.expand_dims(max_norm, -1)
        values_clip = self.reshape(values_clip, self.shape(x))
        values_clip = F.identity(values_clip)
        return values_clip


clip_grad = C.MultitypeFuncGraph("clip_grad")


@clip_grad.register("Number", "Number", "Tensor")
def _clip_grad(clip_type, clip_value, grad):
    """
    Clip gradients.

    Inputs:
        clip_type (int): The way to clip, 0 for 'value', 1 for 'norm'.
        clip_value (float): Specifies how much to clip.
        grad (tuple[Tensor]): Gradients.

    Outputs:
        tuple[Tensor], clipped gradients.
    """
    if clip_type not in (0, 1):
        return grad
    dt = F.dtype(grad)
    if clip_type == 0:
        new_grad = C.clip_by_value(grad, F.cast(F.tuple_to_array((-clip_value,)), dt),
                                   F.cast(F.tuple_to_array((clip_value,)), dt))
    else:
        new_grad = ClipByNorm()(grad, F.cast(F.tuple_to_array((clip_value,)), dt))
    return new_grad


grad_scale = C.MultitypeFuncGraph("grad_scale")
reciprocal = P.Reciprocal()


@grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * reciprocal(scale)


class ClipGradients(nn.Cell):
    """
    Clip gradients.

    Inputs:
        grads (list): List of gradient tuples.
        clip_type (Tensor): The way to clip, 'value' or 'norm'.
        clip_value (Tensor): Specifies how much to clip.

    Returns:
        List, a list of clipped_grad tuples.
    """
    def __init__(self):
        super(ClipGradients, self).__init__()
        self.clip_by_norm = nn.ClipByNorm()
        self.cast = P.Cast()
        self.dtype = P.DType()

    def construct(self,
                  grads,
                  clip_type,
                  clip_value):
        """clip gradients"""
        if clip_type not in (0, 1):
            return grads
        new_grads = ()
        for grad in grads:
            dt = self.dtype(grad)
            if clip_type == 0:
                t = C.clip_by_value(grad, self.cast(F.tuple_to_array((-clip_value,)), dt),
                                    self.cast(F.tuple_to_array((clip_value,)), dt))
            else:
                t = self.clip_by_norm(grad, self.cast(F.tuple_to_array((clip_value,)), dt))
            new_grads = new_grads + (t,)
        return new_grads


class CrossEntropy(nn.Cell):
    """
    Cross Entropy loss
    """
    def __init__(self, num_labels):
        super(CrossEntropy, self).__init__()
        self.onehot = P.OneHot()
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.reduce_sum = P.ReduceSum()
        self.reduce_mean = P.ReduceMean()
        self.reshape = P.Reshape()
        self.last_idx = (-1,)
        self.neg = P.Neg()
        self.cast = P.Cast()
        self.num_labels = num_labels

    def construct(self, logits, label_ids):
        label_ids = self.reshape(label_ids, self.last_idx)
        one_hot_labels = self.onehot(label_ids, self.num_labels, self.on_value, self.off_value)
        per_example_loss = self.neg(self.reduce_sum(one_hot_labels * logits, self.last_idx))
        loss = self.reduce_mean(per_example_loss, self.last_idx)
        return_value = self.cast(loss, mstype.float32)
        return return_value


class NetworkWithCLSLoss(nn.Cell):
    def __init__(self, network):
        super(NetworkWithCLSLoss, self).__init__(auto_prefix=False)
        self.cls_network = network
        self.loss_fct = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    def construct(self, input_ids, input_mask, token_type_id, label_ids):
        logits = self.cls_network(input_ids, input_mask, token_type_id)
        cls_loss = self.loss_fct(logits, label_ids)
        return cls_loss


class NetworkWithMLMLoss(nn.Cell):
    def __init__(self, network, vocab_size=21128):
        super(NetworkWithMLMLoss, self).__init__(auto_prefix=False)
        self.mlm_network = network
        self.vocab_size = vocab_size
        self.reshape = P.Reshape()
        self.loss_fct = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    def construct(self, input_ids, input_mask, token_type_id, label_ids):
        prediction_scores = self.mlm_network(input_ids, input_mask, token_type_id)
        prediction_scores = self.reshape(prediction_scores, (-1, self.vocab_size))
        label_ids = self.reshape(label_ids, (-1,))
        mlm_loss = self.loss_fct(prediction_scores, label_ids)
        return mlm_loss


class NetworkTrainCell(nn.Cell):
    def __init__(self, network, optimizer, sens=1.0):
        super(NetworkTrainCell, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.weights = optimizer.parameters
        self.optimizer = optimizer
        self.sens = sens
        self.grad = C.GradOperation(get_by_list=True,
                                    sens_param=True)
        self.clip_type = 1
        self.clip_value = 1.0
        self.cast = P.Cast()
        self.hyper_map = C.HyperMap()

        self.get_weights_by_key = P.PullWeight()
        self.over_weights_by_key = P.PushWeight()

        self.get_weights_by_key_input_1, \
        self.get_weights_by_key_input_2, \
        self.get_weights_by_key_input_3 = self._get_weights_by_key_inputs(self.network.parameters_and_names())

        self.over_weights_by_key_input_1, \
        self.over_weights_by_key_input_2, \
        self.over_weights_by_key_input_3 = self._over_weights_by_key_inputs(self.network.parameters_and_names())

    def _communication_with_server_1(self, weights):
        result = self.hyper_map(F.partial(self.get_weights_by_key), weights,
                                self.get_weights_by_key_input_2, self.get_weights_by_key_input_3)
        return result

    def _communication_with_server_2(self, weights):
        result = self.hyper_map(F.partial(self.over_weights_by_key), weights,
                                self.over_weights_by_key_input_2,
                                self.over_weights_by_key_input_3)
        return result

    def _get_weights_by_key_inputs(self, weights):
        filtered_weights = []
        weight_names = []
        weight_indices = []
        index = 0
        for weight in weights:
            if weight[1].pull_weight_from_server:
                filtered_weights.append(weight[1])
                weight_names.append(weight[1].name)
                weight_indices.append(index)
            index += 1
        return ParameterTuple(filtered_weights), tuple(weight_names), tuple(weight_indices)

    def _over_weights_by_key_inputs(self, weights):
        filtered_weights = []
        weight_names = []
        weight_indices = []
        index = 0
        for weight in weights:
            if weight[1].push_weight_to_server:
                filtered_weights.append(weight[1])
                weight_names.append(weight[1].name)
                weight_indices.append(index)
            index += 1
        return ParameterTuple(filtered_weights), tuple(weight_names), tuple(weight_indices)

    def construct(self, input_ids, input_mask, token_type_id, label_ids):
        weights = self.weights
        res = self._communication_with_server_1(self.get_weights_by_key_input_1)
        input_ids = F.depend(input_ids, res)
        loss = self.network(input_ids, input_mask, token_type_id, label_ids)
        grads = self.grad(self.network, weights)(input_ids,
                                                 input_mask,
                                                 token_type_id,
                                                 label_ids,
                                                 self.cast(F.tuple_to_array((self.sens,)),
                                                           mstype.float32))
        grads = self.hyper_map(F.partial(clip_grad, self.clip_type, self.clip_value), grads)
        loss = F.depend(loss, self.optimizer(grads))
        weights1 = F.depend(self.over_weights_by_key_input_1, loss)
        loss = F.depend(loss, self._communication_with_server_2(weights1))
        return loss


class NetworkNoClientTrainCell(nn.Cell):
    def __init__(self, network, optimizer, sens=1.0):
        super(NetworkNoClientTrainCell, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.weights = optimizer.parameters
        self.optimizer = optimizer
        self.sens = sens
        self.grad = C.GradOperation(get_by_list=True,
                                    sens_param=True)
        self.clip_type = 1
        self.clip_value = 1.0
        self.cast = P.Cast()
        self.hyper_map = C.HyperMap()

    def construct(self, input_ids, input_mask, token_type_id, label_ids):
        weights = self.weights
        loss = self.network(input_ids, input_mask, token_type_id, label_ids)
        grads = self.grad(self.network, weights)(input_ids,
                                                 input_mask,
                                                 token_type_id,
                                                 label_ids,
                                                 self.cast(F.tuple_to_array((self.sens,)),
                                                           mstype.float32))
        grads = self.hyper_map(F.partial(clip_grad, self.clip_type, self.clip_value), grads)
        self.optimizer(grads)
        return loss
