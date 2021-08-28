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

'''
DynamicRanker network script.
'''
import numpy as np
import mindspore.ops as ops
import mindspore.nn as nn
import mindspore.numpy as mnp
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
from mindspore.common import dtype as mstype
from mindspore.common.initializer import TruncatedNormal
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from mindspore.communication.management import get_group_size
from src.bert_model import BertModel

class DynamicRankerModel(nn.Cell):
    """
    This class is responsible for DynamicRanker task evaluation.
    Args:
        config (Class): Configuration for BertModel.
        is_training (bool): True for training mode. False for eval mode.
        dropout_prob (float): The dropout probability for DynamicRankerModel. Default: 0.0.
        use_one_hot_embeddings (bool): Specifies whether to use one hot encoding form. Default: False.
    """
    def __init__(self, config, is_training, dropout_prob=0.0, use_one_hot_embeddings=False):
        super(DynamicRankerModel, self).__init__()
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.hidden_probs_dropout_prob = 0.0
        self.bert = BertModel(config, is_training, use_one_hot_embeddings)
        self.cast = P.Cast()
        self.weight_init = TruncatedNormal(config.initializer_range)
        self.dtype = config.dtype
        self.dense_1 = nn.Dense(config.hidden_size, 1, weight_init=self.weight_init,
                                has_bias=True).to_float(config.compute_type)
        self.dropout = nn.Dropout(1 - dropout_prob)

    def construct(self, input_ids, input_mask, token_type_id):
        _, pooled_output, _ = \
            self.bert(input_ids, token_type_id, input_mask)
        cls = self.cast(pooled_output, self.dtype)
        cls = self.dropout(cls)
        logits = self.dense_1(cls)
        logits = self.cast(logits, self.dtype)
        return logits

GRADIENT_CLIP_TYPE = 1
GRADIENT_CLIP_VALUE = 1.0

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
        new_grad = nn.ClipByNorm()(grad, F.cast(F.tuple_to_array((clip_value,)), dt))
    return new_grad


grad_scale = C.MultitypeFuncGraph("grad_scale")
reciprocal = P.Reciprocal()
@grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * reciprocal(scale)


class DynamicRankerFinetuneCell(nn.TrainOneStepWithLossScaleCell):
    """
    Especially defined for finetuning where only four inputs tensor are needed.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Different from the builtin loss_scale wrapper cell, we apply grad_clip before the optimization.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        scale_update_cell (Cell): Cell to do the loss scale. Default: None.
    """
    def __init__(self, network, optimizer, scale_update_cell=None):

        super(DynamicRankerFinetuneCell, self).__init__(network, optimizer, scale_update_cell)
        self.cast = P.Cast()
        self.degree = 1
        if self.reducer_flag:
            self.degree = get_group_size()
            self.grad_reducer = DistributedGradReducer(optimizer.parameters, False, self.degree)
        self.loss_scale = None
        self.loss_scaling_manager = scale_update_cell
        if scale_update_cell:
            self.loss_scale = Parameter(Tensor(scale_update_cell.get_loss_scale(), dtype=mstype.float32))

    def construct(self,
                  input_ids,
                  input_mask,
                  token_type_id,
                  label_ids,
                  sens=None):
        """DynamicRanker Finetune"""
        weights = self.weights
        loss = self.network(input_ids,
                            input_mask,
                            token_type_id,
                            label_ids)
        if sens is None:
            scaling_sens = self.loss_scale
        else:
            scaling_sens = sens

        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)
        grads = self.grad(self.network, weights)(input_ids,
                                                 input_mask,
                                                 token_type_id,
                                                 label_ids,
                                                 self.cast(scaling_sens,
                                                           mstype.float32))
        # apply grad reducer on grads
        grads = self.grad_reducer(grads)
        grads = self.hyper_map(F.partial(grad_scale, scaling_sens * self.degree), grads)
        grads = C.clip_by_global_norm(grads, 1.0, None)
        cond = self.get_overflow_status(status, grads)
        overflow = cond
        if sens is None:
            overflow = self.loss_scaling_manager(self.loss_scale, cond)
        if overflow:
            succ = False
        else:
            succ = self.optimizer(grads)
        ret = (loss, cond, scaling_sens)
        return F.depend(ret, succ)

class CrossEntropyLoss(nn.Cell):
    """
    Calculate the cross entropy loss
    Inputs:
        logits: the output logits of the backbone
        label: the ground truth label of the sample
    Returns:
        loss: Tensor, the corrsponding cross entropy loss
    """
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.cross_entropy = P.SoftmaxCrossEntropyWithLogits()
        self.mean = P.ReduceMean()
        self.one_hot = P.OneHot()
        self.one = Tensor(1.0, mstype.float32)
        self.zero = Tensor(0.0, mstype.float32)

    def construct(self, logits, label):
        label = self.one_hot(label, F.shape(logits)[1], self.one, self.zero)
        loss = self.cross_entropy(logits, label)[0]
        loss = self.mean(loss, (-1,))
        return loss

class DynamicRankerBase(nn.Cell):
    """
    Train interface for DynamicRanker base finetuning task.
    Args:
        config (Class): Configuration for BertModel.
        is_training (bool): True for training mode. False for eval mode.
        dropout_prob (float): The dropout probability for DynamicRankerModel. Default: 0.0.
        use_one_hot_embeddings (bool): Specifies whether to use one hot encoding form. Default: False.
        batch_size (int): size of input batch.
        group_size (int): group size of block.
        group_num (int): group number of block.
        rank_id (int): rank id of device.
        device_num (int): number of device.
        seq_len (int): Length of input sequence.
    """
    def __init__(self, config, is_training, dropout_prob=0.0, use_one_hot_embeddings=False,
                 batch_size=64, group_size=8, group_num=2, rank_id=0, device_num=1, seq_len=512):
        super(DynamicRankerBase, self).__init__()
        self.bert = DynamicRankerModel(config, is_training, dropout_prob, use_one_hot_embeddings)
        self.is_training = is_training
        self.labels = Tensor(np.zeros([batch_size]).astype(np.int32))
        self.reshape = P.Reshape()
        self.shape_flat = (batch_size, -1)
        self.log_softmax = P.LogSoftmax(axis=-1)
        self.allgather = ops.AllGather()
        self.loss = CrossEntropyLoss()
        self.slice = ops.Slice()
        self.group_id = rank_id * group_num // device_num
        self.begin = group_size * batch_size * self.group_id
        self.size = group_size * batch_size
        self.transpose = P.Transpose()
        self.shape1 = (device_num // group_num, batch_size, -1)
        self.shape2 = (batch_size, -1)
        self.trans_shape = (1, 0, 2)
        self.seq_len = seq_len

    def construct(self, input_ids, input_mask, token_type_id, label_ids):
        """
        construct interface for DynamicRanker base finetuning task.
        """
        input_ids = P.Reshape()(input_ids, (-1, self.seq_len))
        input_mask = P.Reshape()(input_mask, (-1, self.seq_len))
        token_type_id = P.Reshape()(token_type_id, (-1, self.seq_len))
        logits = self.bert(input_ids, input_mask, token_type_id)
        logits = self.allgather(logits)
        logits = self.slice(logits, [self.begin, 0], [self.size, 1])
        logits = self.reshape(logits, self.shape1)
        logits = self.transpose(logits, self.trans_shape)
        logits = self.reshape(logits, self.shape2)
        loss = self.loss(logits, self.labels)
        return loss

class DynamicRanker(nn.Cell):
    """
    Train interface for DynamicRanker v3 finetuning task.
    Args:
        config (Class): Configuration for BertModel.
        is_training (bool): True for training mode. False for eval mode.
        dropout_prob (float): The dropout probability for DynamicRankerModel. Default: 0.0.
        use_one_hot_embeddings (bool): Specifies whether to use one hot encoding form. Default: False.
        batch_size (int): size of input batch.
        group_size (int): group size of block.
        group_num (int): group number of block.
        rank_id (int): rank id of device.
        device_num (int): number of device.
        seq_len (int): Length of input sequence.
    """
    def __init__(self, config, is_training, dropout_prob=0.0, use_one_hot_embeddings=False,
                 batch_size=64, group_size=8, group_num=2, rank_id=0, device_num=1, seq_len=512):
        super(DynamicRanker, self).__init__()
        self.bert = DynamicRankerModel(config, is_training, dropout_prob, use_one_hot_embeddings)
        self.is_training = is_training
        self.labels = Tensor(np.zeros([batch_size]).astype(np.int32))
        self.reshape = P.Reshape()
        self.shape_flat = (batch_size, -1)
        self.log_softmax = P.LogSoftmax(axis=-1)
        self.allgather = ops.AllGather()
        self.loss = CrossEntropyLoss()
        self.slice = ops.Slice()
        self.group_id = rank_id * group_num // device_num
        self.begin = group_size * batch_size * self.group_id
        self.size = group_size * batch_size
        self.transpose = P.Transpose()
        self.shape1 = (device_num // group_num, batch_size, -1)
        self.shape2 = (batch_size, -1)
        self.trans_shape = (1, 0, 2)
        self.batch_size = batch_size
        self.group_size = group_size
        self.topk = ops.TopK(sorted=True)
        self.concat = ops.Concat(axis=1)
        self.cast = ops.Cast()
        self.seq_len = seq_len

    def construct(self, input_ids, input_mask, token_type_id, label_ids):
        """
        construct interface for DynamicRanker v3 finetuning task.
        """
        input_ids = P.Reshape()(input_ids, (-1, self.seq_len))
        input_mask = P.Reshape()(input_mask, (-1, self.seq_len))
        token_type_id = P.Reshape()(token_type_id, (-1, self.seq_len))
        logits = self.bert(input_ids, input_mask, token_type_id)
        logits = self.allgather(logits)
        logits = self.slice(logits, [self.begin, 0], [self.size, 1])
        logits = self.reshape(logits, self.shape1)
        logits = self.transpose(logits, self.trans_shape)
        logits = self.reshape(logits, self.shape2)
        pos_sample = self.slice(logits, [0, 0], [self.batch_size, 1])
        res_sample = self.slice(logits, [0, 1], [self.batch_size, self.group_size - 1])
        values, _ = self.topk(res_sample, 15)
        label_ids = P.Reshape()(label_ids, (-1, 15))
        indices_ = self.cast(label_ids, mstype.float32)
        _, indices = self.topk(indices_, 15)
        values = mnp.take_along_axis(values, indices, 1)
        c2_score = self.concat((pos_sample, values))
        loss = self.loss(c2_score, self.labels)
        return loss


class DynamicRankerPredict(nn.Cell):
    """
    Predict interface for DynamicRanker finetuning task.
    Args:
        config (Class): Configuration for BertModel.
        is_training (bool): True for training mode. False for eval mode.
        dropout_prob (float): The dropout probability for DynamicRankerModel. Default: 0.0.
        use_one_hot_embeddings (bool): Specifies whether to use one hot encoding form. Default: False.
    """
    def __init__(self, config, is_training, dropout_prob=0.0, use_one_hot_embeddings=False):
        super(DynamicRankerPredict, self).__init__()
        self.bert = DynamicRankerModel(config, is_training, dropout_prob, use_one_hot_embeddings)
    def construct(self, input_ids, input_mask, token_type_id):
        logits = self.bert(input_ids, input_mask, token_type_id)
        return logits
