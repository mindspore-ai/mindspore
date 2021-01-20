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

"""Tinybert model"""

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
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from .tinybert_model import BertModel, TinyBertModel, BertModelCLS, BertModelNER


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

class BertNetworkWithLoss_gd(nn.Cell):
    """
    Provide bert pre-training loss through network.
    Args:
        config (BertConfig): The config of BertModel.
        is_training (bool): Specifies whether to use the training mode.
        use_one_hot_embeddings (bool): Specifies whether to use one-hot for embeddings. Default: False.
    Returns:
        Tensor, the loss of the network.
    """
    def __init__(self, teacher_config, teacher_ckpt, student_config, is_training, use_one_hot_embeddings=False,
                 is_att_fit=True, is_rep_fit=True):
        super(BertNetworkWithLoss_gd, self).__init__()
        # load teacher model
        self.teacher = BertModel(teacher_config, False, use_one_hot_embeddings)
        param_dict = load_checkpoint(teacher_ckpt)
        new_param_dict = {}
        for key, value in param_dict.items():
            new_key = re.sub('^bert.bert.', 'teacher.', key)
            new_param_dict[new_key] = value
        load_param_into_net(self.teacher, new_param_dict)
        # no_grad
        self.teacher.set_train(False)
        params = self.teacher.trainable_params()
        for param in params:
            param.requires_grad = False
        # student model
        self.bert = TinyBertModel(student_config, is_training, use_one_hot_embeddings)
        self.cast = P.Cast()
        self.fit_dense = nn.Dense(student_config.hidden_size,
                                  teacher_config.hidden_size).to_float(teacher_config.compute_type)
        self.teacher_layers_num = teacher_config.num_hidden_layers
        self.student_layers_num = student_config.num_hidden_layers
        self.layers_per_block = int(self.teacher_layers_num / self.student_layers_num)
        self.is_att_fit = is_att_fit
        self.is_rep_fit = is_rep_fit
        self.loss_mse = nn.MSELoss()
        self.select = P.Select()
        self.zeroslike = P.ZerosLike()
        self.dtype = teacher_config.dtype

    def construct(self,
                  input_ids,
                  input_mask,
                  token_type_id):
        """general distill network with loss"""
        # teacher model
        _, _, _, teacher_seq_output, teacher_att_output = self.teacher(input_ids, token_type_id, input_mask)
        # student model
        _, _, _, student_seq_output, student_att_output = self.bert(input_ids, token_type_id, input_mask)
        total_loss = 0
        if self.is_att_fit:
            selected_teacher_att_output = ()
            selected_student_att_output = ()
            for i in range(self.student_layers_num):
                selected_teacher_att_output += (teacher_att_output[(i + 1) * self.layers_per_block - 1],)
                selected_student_att_output += (student_att_output[i],)
            att_loss = 0
            for i in range(self.student_layers_num):
                student_att = selected_student_att_output[i]
                teacher_att = selected_teacher_att_output[i]
                student_att = self.select(student_att <= self.cast(-100.0, mstype.float32), self.zeroslike(student_att),
                                          student_att)
                teacher_att = self.select(teacher_att <= self.cast(-100.0, mstype.float32), self.zeroslike(teacher_att),
                                          teacher_att)
                att_loss += self.loss_mse(student_att, teacher_att)
            total_loss += att_loss
        if self.is_rep_fit:
            selected_teacher_seq_output = ()
            selected_student_seq_output = ()
            for i in range(self.student_layers_num + 1):
                selected_teacher_seq_output += (teacher_seq_output[i * self.layers_per_block],)
                fit_dense_out = self.fit_dense(student_seq_output[i])
                fit_dense_out = self.cast(fit_dense_out, self.dtype)
                selected_student_seq_output += (fit_dense_out,)
            rep_loss = 0
            for i in range(self.student_layers_num + 1):
                teacher_rep = selected_teacher_seq_output[i]
                student_rep = selected_student_seq_output[i]
                rep_loss += self.loss_mse(student_rep, teacher_rep)
            total_loss += rep_loss
        return self.cast(total_loss, mstype.float32)

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
        self.clear_status = P.NPUClearFloatStatus()
        self.reduce_sum = P.ReduceSum(keep_dims=False)
        self.base = Tensor(1, mstype.float32)
        self.less_equal = P.LessEqual()
        self.hyper_map = C.HyperMap()
        self.loss_scale = None
        self.loss_scaling_manager = scale_update_cell
        if scale_update_cell:
            self.loss_scale = Parameter(Tensor(scale_update_cell.get_loss_scale(), dtype=mstype.float32))

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
        init = F.depend(init, loss)
        clear_status = self.clear_status(init)
        scaling_sens = F.depend(scaling_sens, clear_status)
        grads = self.grad(self.network, weights)(input_ids,
                                                 input_mask,
                                                 token_type_id,
                                                 self.cast(scaling_sens,
                                                           mstype.float32))
        # apply grad reducer on grads
        grads = self.grad_reducer(grads)
        grads = self.hyper_map(F.partial(grad_scale, scaling_sens * self.degree), grads)
        grads = self.hyper_map(F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)
        init = F.depend(init, grads)
        get_status = self.get_status(init)
        init = F.depend(init, get_status)
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
        if overflow:
            succ = False
        else:
            succ = self.optimizer(grads)
        ret = (loss, cond, scaling_sens)
        return F.depend(ret, succ)

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
        succ = self.optimizer(grads)
        return F.depend(loss, succ)

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
    def __init__(self, teacher_config, teacher_ckpt, student_config, student_ckpt,
                 is_training, task_type, num_labels, use_one_hot_embeddings=False,
                 is_predistill=True, is_att_fit=True, is_rep_fit=True,
                 temperature=1.0, dropout_prob=0.1):
        super(BertNetworkWithLoss_td, self).__init__()
        # load teacher model
        if task_type == "classification":
            self.teacher = BertModelCLS(teacher_config, False, num_labels, dropout_prob,
                                        use_one_hot_embeddings, "teacher")
            self.bert = BertModelCLS(student_config, is_training, num_labels, dropout_prob,
                                     use_one_hot_embeddings, "student")
        elif task_type == "ner":
            self.teacher = BertModelNER(teacher_config, False, num_labels, dropout_prob,
                                        use_one_hot_embeddings, "teacher")
            self.bert = BertModelNER(student_config, is_training, num_labels, dropout_prob,
                                     use_one_hot_embeddings, "student")
        else:
            raise ValueError(f"Not support task type: {task_type}")
        param_dict = load_checkpoint(teacher_ckpt)
        new_param_dict = {}
        for key, value in param_dict.items():
            new_key = re.sub('^bert.', 'teacher.', key)
            new_param_dict[new_key] = value
        load_param_into_net(self.teacher, new_param_dict)

        # no_grad
        self.teacher.set_train(False)
        params = self.teacher.trainable_params()
        for param in params:
            param.requires_grad = False
        # load student model
        param_dict = load_checkpoint(student_ckpt)
        if is_predistill:
            new_param_dict = {}
            for key, value in param_dict.items():
                new_key = re.sub('tinybert_', 'bert_', 'bert.' + key)
                new_param_dict[new_key] = value
            load_param_into_net(self.bert, new_param_dict)
        else:
            new_param_dict = {}
            for key, value in param_dict.items():
                new_key = re.sub('tinybert_', 'bert_', key)
                new_param_dict[new_key] = value
            load_param_into_net(self.bert, new_param_dict)
        self.cast = P.Cast()
        self.fit_dense = nn.Dense(student_config.hidden_size,
                                  teacher_config.hidden_size).to_float(teacher_config.compute_type)
        self.teacher_layers_num = teacher_config.num_hidden_layers
        self.student_layers_num = student_config.num_hidden_layers
        self.layers_per_block = int(self.teacher_layers_num / self.student_layers_num)
        self.is_predistill = is_predistill
        self.is_att_fit = is_att_fit
        self.is_rep_fit = is_rep_fit
        self.use_soft_cross_entropy = task_type in ["classification", "ner"]
        self.temperature = temperature
        self.loss_mse = nn.MSELoss()
        self.select = P.Select()
        self.zeroslike = P.ZerosLike()
        self.dtype = student_config.dtype
        self.num_labels = num_labels
        self.dtype = teacher_config.dtype
        self.soft_cross_entropy = SoftCrossEntropy()

    def construct(self,
                  input_ids,
                  input_mask,
                  token_type_id,
                  label_ids):
        """task distill network with loss"""
        # teacher model
        teacher_seq_output, teacher_att_output, teacher_logits, _ = self.teacher(input_ids, token_type_id, input_mask)
        # student model
        student_seq_output, student_att_output, student_logits, _ = self.bert(input_ids, token_type_id, input_mask)
        total_loss = 0
        if self.is_predistill:
            if self.is_att_fit:
                selected_teacher_att_output = ()
                selected_student_att_output = ()
                for i in range(self.student_layers_num):
                    selected_teacher_att_output += (teacher_att_output[(i + 1) * self.layers_per_block - 1],)
                    selected_student_att_output += (student_att_output[i],)
                att_loss = 0
                for i in range(self.student_layers_num):
                    student_att = selected_student_att_output[i]
                    teacher_att = selected_teacher_att_output[i]
                    student_att = self.select(student_att <= self.cast(-100.0, mstype.float32),
                                              self.zeroslike(student_att),
                                              student_att)
                    teacher_att = self.select(teacher_att <= self.cast(-100.0, mstype.float32),
                                              self.zeroslike(teacher_att),
                                              teacher_att)
                    att_loss += self.loss_mse(student_att, teacher_att)
                total_loss += att_loss
            if self.is_rep_fit:
                selected_teacher_seq_output = ()
                selected_student_seq_output = ()
                for i in range(self.student_layers_num + 1):
                    selected_teacher_seq_output += (teacher_seq_output[i * self.layers_per_block],)
                    fit_dense_out = self.fit_dense(student_seq_output[i])
                    fit_dense_out = self.cast(fit_dense_out, self.dtype)
                    selected_student_seq_output += (fit_dense_out,)
                rep_loss = 0
                for i in range(self.student_layers_num + 1):
                    teacher_rep = selected_teacher_seq_output[i]
                    student_rep = selected_student_seq_output[i]
                    rep_loss += self.loss_mse(student_rep, teacher_rep)
                total_loss += rep_loss
        else:
            if self.use_soft_cross_entropy:
                cls_loss = self.soft_cross_entropy(student_logits / self.temperature, teacher_logits / self.temperature)
            else:
                cls_loss = self.loss_mse(student_logits[len(student_logits) - 1], label_ids[len(label_ids) - 1])
            total_loss += cls_loss
        return self.cast(total_loss, mstype.float32)

class BertEvaluationWithLossScaleCell(nn.Cell):
    """
    Especially defined for finetuning where only four inputs tensor are needed.
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
        self.alloc_status = P.NPUAllocFloatStatus()
        self.get_status = P.NPUGetFloatStatus()
        self.clear_status = P.NPUClearFloatStatus()
        self.reduce_sum = P.ReduceSum(keep_dims=False)
        self.base = Tensor(1, mstype.float32)
        self.less_equal = P.LessEqual()
        self.hyper_map = C.HyperMap()
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
        """Defines the computation performed."""
        weights = self.weights
        loss = self.network(input_ids,
                            input_mask,
                            token_type_id,
                            label_ids)
        if sens is None:
            scaling_sens = self.loss_scale
        else:
            scaling_sens = sens
        # alloc status and clear should be right before gradoperation
        init = self.alloc_status()
        init = F.depend(init, loss)
        clear_status = self.clear_status(init)
        scaling_sens = F.depend(scaling_sens, clear_status)
        grads = self.grad(self.network, weights)(input_ids,
                                                 input_mask,
                                                 token_type_id,
                                                 label_ids,
                                                 self.cast(scaling_sens,
                                                           mstype.float32))
        # apply grad reducer on grads
        grads = self.grad_reducer(grads)
        grads = self.hyper_map(F.partial(grad_scale, scaling_sens * self.degree), grads)
        grads = self.hyper_map(F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)
        init = F.depend(init, grads)
        get_status = self.get_status(init)
        init = F.depend(init, get_status)
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
        if overflow:
            succ = False
        else:
            succ = self.optimizer(grads)
        ret = (loss, cond, scaling_sens)
        return F.depend(ret, succ)


class BertEvaluationCell(nn.Cell):
    """
    Especially defined for finetuning where only four inputs tensor are needed.
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
        succ = self.optimizer(grads)
        return F.depend(loss, succ)
