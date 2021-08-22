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

"""q8bert model"""

import re
import mindspore.nn as nn
from mindspore import context
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore.common.tensor import Tensor
from mindspore.common import dtype as mstype
from mindspore.common.parameter import Parameter
from mindspore.communication.management import get_group_size
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from mindspore.context import ParallelMode
from mindspore.train.serialization import load_checkpoint
from mindspore.compression.quant.quant_utils import load_nonquant_param_into_quant_net
from .q8bert_model import BertModelCLS

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


_grad_overflow = C.MultitypeFuncGraph("_grad_overflow")
grad_overflow = P.FloatStatus()
@_grad_overflow.register("Tensor")
def _tensor_grad_overflow(grad):
    return grad_overflow(grad)


grad_scale = C.MultitypeFuncGraph("grad_scale")
reciprocal = P.Reciprocal()


@grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * reciprocal(scale)


class ClipGradients(nn.Cell):
    """
    Clip gradients.

    Args:
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


class SoftCrossEntropy(nn.Cell):
    """SoftCrossEntropy loss"""
    def __init__(self):
        super(SoftCrossEntropy, self).__init__()
        self.log_softmax = P.LogSoftmax(axis=-1)
        self.softmax = P.Softmax(axis=-1)
        self.reduce_mean = P.ReduceMean()
        self.cast = P.Cast()

    def construct(self, predicts, targets):
        likelihood = self.log_softmax(predicts)
        target_prob = self.softmax(targets)
        loss = self.reduce_mean(-target_prob * likelihood)

        return self.cast(loss, mstype.float32)


class BertTrainWithLossScaleCell(nn.Cell):
    """
    Encapsulation class of bert network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        scale_update_cell (Cell): Cell to do the loss scale. Default: None.
    """
    def __init__(self, network, optimizer, scale_update_cell=None):
        super(BertTrainWithLossScaleCell, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.weights = optimizer.parameters
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True,
                                    sens_param=True)
        self.reducer_flag = False
        self.allreduce = P.AllReduce()
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        self.grad_reducer = F.identity
        self.degree = 1
        if self.reducer_flag:
            self.degree = get_group_size()
            self.grad_reducer = DistributedGradReducer(optimizer.parameters, False, self.degree)
        self.is_distributed = (self.parallel_mode != ParallelMode.STAND_ALONE)
        self.cast = P.Cast()
        self.alloc_status = P.NPUAllocFloatStatus()
        self.get_status = P.NPUGetFloatStatus()
        self.clear_before_grad = P.NPUClearFloatStatus()
        self.reduce_sum = P.ReduceSum(keep_dims=False)
        self.depend_parameter_use = P.Depend()
        self.base = Tensor(1, mstype.float32)
        self.less_equal = P.LessEqual()
        self.hyper_map = C.HyperMap()
        self.loss_scale = None
        self.loss_scaling_manager = scale_update_cell
        if scale_update_cell:
            self.loss_scale = Parameter(Tensor(scale_update_cell.get_loss_scale(), dtype=mstype.float32))

    @C.add_flags(has_effect=True)
    def construct(self,
                  input_ids,
                  input_mask,
                  token_type_id,
                  sens=None):
        """Defines the computation performed."""
        weights = self.weights
        loss = self.network(input_ids,
                            input_mask,
                            token_type_id)
        if sens is None:
            scaling_sens = self.loss_scale
        else:
            scaling_sens = sens
        # alloc status and clear should be right before gradoperation
        init = self.alloc_status()
        self.clear_before_grad(init)
        grads = self.grad(self.network, weights)(input_ids,
                                                 input_mask,
                                                 token_type_id,
                                                 self.cast(scaling_sens,
                                                           mstype.float32))
        # apply grad reducer on grads
        grads = self.grad_reducer(grads)
        grads = self.hyper_map(F.partial(grad_scale, scaling_sens * self.degree), grads)
        grads = self.hyper_map(F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)
        self.get_status(init)
        flag_sum = self.reduce_sum(init, (0,))
        if self.is_distributed:
            # sum overflow flag over devices
            flag_reduce = self.allreduce(flag_sum)
            cond = self.less_equal(self.base, flag_reduce)
        else:
            cond = self.less_equal(self.base, flag_sum)
        overflow = cond
        if sens is None:
            overflow = self.loss_scaling_manager(self.loss_scale, cond)
        if not overflow:
            self.optimizer(grads)
        return (loss, cond, scaling_sens)


class BertTrainCell(nn.Cell):
    """
    Encapsulation class of bert network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        sens (Number): The adjust parameter. Default: 1.0.
    """
    def __init__(self, network, optimizer, sens=1.0):
        super(BertTrainCell, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.weights = optimizer.parameters
        self.optimizer = optimizer
        self.sens = sens
        self.grad = C.GradOperation(get_by_list=True,
                                    sens_param=True)
        self.reducer_flag = False
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        self.grad_reducer = F.identity
        self.degree = 1
        if self.reducer_flag:
            mean = context.get_auto_parallel_context("gradients_mean")
            self.degree = get_group_size()
            self.grad_reducer = DistributedGradReducer(optimizer.parameters, mean, self.degree)
        self.cast = P.Cast()
        self.hyper_map = C.HyperMap()

    def construct(self,
                  input_ids,
                  input_mask,
                  token_type_id):
        """Defines the computation performed."""
        weights = self.weights
        loss = self.network(input_ids,
                            input_mask,
                            token_type_id)
        grads = self.grad(self.network, weights)(input_ids,
                                                 input_mask,
                                                 token_type_id,
                                                 self.cast(F.tuple_to_array((self.sens,)),
                                                           mstype.float32))
        # apply grad reducer on grads
        grads = self.grad_reducer(grads)
        grads = self.hyper_map(F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)
        self.optimizer(grads)
        return loss


class BertNetworkWithLoss_td(nn.Cell):
    """
    Provide bert pre-training loss through network.
    Args:
        config (BertConfig): The config of BertModel.
        is_training (bool): Specifies whether to use the training mode.
        use_one_hot_embeddings (bool): Specifies whether to use one-hot for embeddings. Default: False.
    Returns:
        Tensor, the loss of the network.
    """
    def __init__(self, student_config, student_ckpt, do_quant,
                 is_training, task_type, num_labels, use_one_hot_embeddings=False,
                 is_predistill=True, is_att_fit=True, is_rep_fit=True,
                 temperature=1.0, dropout_prob=0.1):
        super(BertNetworkWithLoss_td, self).__init__()

        # load student model
        self.bert = BertModelCLS(student_config, is_training, num_labels, dropout_prob,
                                 use_one_hot_embeddings, "student")
        if do_quant:
            import src.q8bert_model as quant_bert_model
            self.bert = quant_bert_model.BertModelCLS(student_config, is_training, num_labels, dropout_prob,
                                                      use_one_hot_embeddings, "student")
        else:
            import src.bert_model as bert_model
            self.bert = bert_model.BertModelCLS(student_config, is_training, num_labels, dropout_prob,
                                                use_one_hot_embeddings, "student")

        param_dict = load_checkpoint(student_ckpt)
        if is_predistill:
            new_param_dict = {}
            for key, value in param_dict.items():
                new_key = re.sub('tinybert_', 'bert_', 'bert.' + key)
                new_param_dict[new_key] = value
            load_nonquant_param_into_quant_net(self.bert, new_param_dict)
        else:
            new_param_dict = {}
            for key, value in param_dict.items():
                new_key = re.sub('tinybert_', 'bert_', key)
                new_param_dict[new_key] = value
            load_nonquant_param_into_quant_net(self.bert, new_param_dict)
        self.cast = P.Cast()
        self.student_layers_num = student_config.num_hidden_layers
        self.is_predistill = is_predistill
        self.is_att_fit = is_att_fit
        self.is_rep_fit = is_rep_fit
        self.task_type = task_type
        self.temperature = temperature
        self.loss_mse = nn.MSELoss()
        self.select = P.Select()
        self.zeroslike = P.ZerosLike()
        self.dtype = student_config.dtype
        self.num_labels = num_labels
        self.soft_cross_entropy = SoftCrossEntropy()
        self.reshape = P.Reshape()
        self.lgt_fct = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    def construct(self,
                  input_ids,
                  input_mask,
                  token_type_id,
                  label_ids):
        """task distill network with loss"""
        # student model
        _, _, student_logits, _ = self.bert(input_ids, token_type_id, input_mask)
        total_loss = 0

        if self.task_type == "classification":
            student_logits = self.cast(student_logits, mstype.float32)
            label_ids_reshape = self.reshape(self.cast(label_ids, mstype.int32), (-1,))
            cls_loss = self.lgt_fct(student_logits, label_ids_reshape)
        else:
            student_logits = self.reshape(student_logits, (-1,))
            label_ids = self.reshape(label_ids, (-1,))
            cls_loss = self.loss_mse(student_logits, label_ids)
        total_loss += cls_loss
        return self.cast(total_loss, mstype.float32)


class BertEvaluationWithLossScaleCell(nn.Cell):
    """
    specifically defined for finetuning where only four inputs tensor are needed.
    """
    def __init__(self, network, optimizer, scale_update_cell=None):
        super(BertEvaluationWithLossScaleCell, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.weights = optimizer.parameters
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True,
                                    sens_param=True)
        self.reducer_flag = False
        self.allreduce = P.AllReduce()
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        self.grad_reducer = F.identity
        self.degree = 1
        if self.reducer_flag:
            self.degree = get_group_size()
            self.grad_reducer = DistributedGradReducer(optimizer.parameters, False, self.degree)
        self.is_distributed = (self.parallel_mode != ParallelMode.STAND_ALONE)
        self.cast = P.Cast()
        self.gpu_target = False
        if context.get_context("device_target") == "GPU":
            self.gpu_target = True
            self.float_status = P.FloatStatus()
            self.addn = P.AddN()
            self.reshape = P.Reshape()
        else:
            self.alloc_status = P.NPUAllocFloatStatus()
            self.get_status = P.NPUGetFloatStatus()
            self.clear_before_grad = P.NPUClearFloatStatus()
        self.reduce_sum = P.ReduceSum(keep_dims=False)
        self.depend_parameter_use = P.Depend()
        self.base = Tensor(1, mstype.float32)
        self.less_equal = P.LessEqual()
        self.hyper_map = C.HyperMap()
        self.loss_scale = None
        self.loss_scaling_manager = scale_update_cell
        if scale_update_cell:
            self.loss_scale = Parameter(Tensor(scale_update_cell.get_loss_scale(), dtype=mstype.float32))

    @C.add_flags(has_effect=True)
    def construct(self,
                  input_ids,
                  input_mask,
                  token_type_id,
                  label_ids,
                  sens=None):
        """Defines the computation performed."""
        weights = self.weights
        init = False
        loss = self.network(input_ids,
                            input_mask,
                            token_type_id,
                            label_ids)
        if sens is None:
            scaling_sens = self.loss_scale
        else:
            scaling_sens = sens
        if not self.gpu_target:
            init = self.alloc_status()
            clear_before_grad = self.clear_before_grad(init)
            F.depend(loss, init)
            self.depend_parameter_use(clear_before_grad, scaling_sens)
        # alloc status and clear should be right before gradoperation
        grads = self.grad(self.network, weights)(input_ids,
                                                 input_mask,
                                                 token_type_id,
                                                 label_ids,
                                                 self.cast(scaling_sens,
                                                           mstype.float32))
        # apply grad reducer on grads
        grads = self.hyper_map(F.partial(grad_scale, scaling_sens), grads)
        grads = self.hyper_map(F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)
        if self.reducer_flag:
            grads = self.grad_reducer(grads)
        if not self.gpu_target:
            flag = self.get_status(init)
            flag_sum = self.reduce_sum(init, (0,))
            F.depend(grads, flag)
            F.depend(flag, flag_sum)
        else:
            flag_sum = self.hyper_map(F.partial(_grad_overflow), grads)
            flag_sum = self.addn(flag_sum)
            flag_sum = self.reshape(flag_sum, (()))
        if self.is_distributed:
            # sum overflow flag over devices
            flag_reduce = self.allreduce(flag_sum)
            cond = self.less_equal(self.base, flag_reduce)
        else:
            cond = self.less_equal(self.base, flag_sum)
        overflow = cond
        if sens is None:
            overflow = self.loss_scaling_manager(self.loss_scale, cond)
        if not overflow:
            self.optimizer(grads)
        return (loss, cond)


class BertEvaluationCell(nn.Cell):
    """
    specifically defined for finetuning where only four inputs tensor are needed.
    """
    def __init__(self, network, optimizer, sens=1.0):
        super(BertEvaluationCell, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.weights = optimizer.parameters
        self.optimizer = optimizer
        self.sens = sens
        self.grad = C.GradOperation(get_by_list=True,
                                    sens_param=True)
        self.reducer_flag = False
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        self.grad_reducer = F.identity
        self.degree = 1
        if self.reducer_flag:
            mean = context.get_auto_parallel_context("gradients_mean")
            self.degree = get_group_size()
            self.grad_reducer = DistributedGradReducer(optimizer.parameters, mean, self.degree)
        self.is_distributed = (self.parallel_mode != ParallelMode.STAND_ALONE)
        self.cast = P.Cast()
        self.hyper_map = C.HyperMap()

    def construct(self,
                  input_ids,
                  input_mask,
                  token_type_id,
                  label_ids):
        """Defines the computation performed."""
        # return input_ids
        weights = self.weights
        loss = self.network(input_ids,
                            input_mask,
                            token_type_id,
                            label_ids)
        grads = self.grad(self.network, weights)(input_ids,
                                                 input_mask,
                                                 token_type_id,
                                                 label_ids,
                                                 self.cast(F.tuple_to_array((self.sens,)),
                                                           mstype.float32))
        # apply grad reducer on grads
        grads = self.grad_reducer(grads)
        grads = self.hyper_map(F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)
        self.optimizer(grads)
        return loss
