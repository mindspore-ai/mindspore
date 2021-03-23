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

"""Train Cell."""

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
from .tinybert_model import BertModelCLS
from .quant import QuantizeWeightCell
from .config import gradient_cfg


class ClipByNorm(nn.Cell):
    r"""
        Clips tensor values to a maximum :math:`L_2`-norm.

        The output of this layer remains the same if the :math:`L_2`-norm of the input tensor
        is not greater than the argument clip_norm. Otherwise the tensor will be normalized as:

        .. math::
            \text{output}(X) = \frac{\text{clip_norm} * X}{L_2(X)},

        where :math:`L_2(X)` is the :math:`L_2`-norm of :math:`X`.

        Args:
            axis (Union[None, int, tuple(int)]): Compute the L2-norm along the Specific dimension.
                                                Default: None, all dimensions to calculate.

        Inputs:
            - **input** (Tensor) - Tensor of shape N-D. The type must be float32 or float16.
            - **clip_norm** (Tensor) - A scalar Tensor of shape :math:`()` or :math:`(1)`.
              Or a tensor shape can be broadcast to input shape.

        Outputs:
            Tensor, clipped tensor with the same shape as the input, whose type is float32.

        Supported Platforms:
            ``Ascend`` ``GPU``

        Examples:
            >>> net = nn.ClipByNorm()
            >>> input = Tensor(np.random.randint(0, 10, [4, 16]), mindspore.float32)
            >>> clip_norm = Tensor(np.array([100]).astype(np.float32))
            >>> output = net(input, clip_norm)
            >>> print(output.shape)
            (4, 16)

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
# pylint: disable=consider-using-in


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
    if clip_type != 0 and clip_type != 1:
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
        if clip_type != 0 and clip_type != 1:
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


class SoftmaxCrossEntropy(nn.Cell):
    """SoftmaxCrossEntropy loss"""
    def __init__(self):
        super(SoftmaxCrossEntropy, self).__init__()
        self.log_softmax = P.LogSoftmax(axis=-1)
        self.softmax = P.Softmax(axis=-1)
        self.reduce_mean = P.ReduceMean()
        self.cast = P.Cast()

    def construct(self, predicts, targets):
        likelihood = self.log_softmax(predicts)
        target_prob = self.softmax(targets)
        loss = self.reduce_mean(-target_prob * likelihood)

        return self.cast(loss, mstype.float32)


class BertNetworkWithLoss(nn.Cell):
    """
    Provide bert pre-training loss through network.
    Args:
        teacher_config (BertConfig): The config of BertModel.
        is_training (bool): Specifies whether to use the training mode.
        use_one_hot_embeddings (bool): Specifies whether to use one-hot for embeddings. Default: False.
    Returns:
        Tensor, the loss of the network.
    """
    def __init__(self, teacher_config, teacher_ckpt, student_config, student_ckpt,
                 is_training, task_type, num_labels, use_one_hot_embeddings=False,
                 temperature=1.0, dropout_prob=0.1):
        super(BertNetworkWithLoss, self).__init__()
        # load teacher model
        self.teacher = BertModelCLS(teacher_config, False, num_labels, dropout_prob,
                                    use_one_hot_embeddings, "teacher")
        param_dict = load_checkpoint(teacher_ckpt)
        new_param_dict = {}
        for key, value in param_dict.items():
            new_key = 'teacher.' + key
            new_param_dict[new_key] = value
        load_param_into_net(self.teacher, new_param_dict)

        # no_grad
        self.teacher.set_train(False)
        params = self.teacher.trainable_params()
        for param in params:
            param.requires_grad = False
        # load student model
        self.bert = BertModelCLS(student_config, is_training, num_labels, dropout_prob,
                                 use_one_hot_embeddings, "student")
        param_dict = load_checkpoint(student_ckpt)
        new_param_dict = {}
        for key, value in param_dict.items():
            new_key = 'bert.' + key
            new_param_dict[new_key] = value
        load_param_into_net(self.bert, new_param_dict)
        self.cast = P.Cast()
        self.teacher_layers_num = teacher_config.num_hidden_layers
        self.student_layers_num = student_config.num_hidden_layers
        self.layers_per_block = int(self.teacher_layers_num / self.student_layers_num)
        self.is_att_fit = student_config.is_att_fit
        self.is_rep_fit = student_config.is_rep_fit
        self.is_lgt_fit = student_config.is_lgt_fit
        self.task_type = task_type
        self.temperature = temperature
        self.loss_mse = nn.MSELoss()
        self.lgt_fct = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        self.select = P.Select()
        self.zeroslike = P.ZerosLike()
        self.dtype = student_config.dtype
        self.num_labels = num_labels
        self.soft_cross_entropy = SoftmaxCrossEntropy()
        self.compute_type = student_config.compute_type
        self.embedding_bits = student_config.embedding_bits
        self.weight_bits = student_config.weight_bits
        self.weight_clip_value = student_config.weight_clip_value
        self.reshape = P.Reshape()

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
                selected_student_seq_output += (student_seq_output[i],)
            rep_loss = 0
            for i in range(self.student_layers_num + 1):
                student_rep = selected_student_seq_output[i]
                teacher_rep = selected_teacher_seq_output[i]
                rep_loss += self.loss_mse(student_rep, teacher_rep)
            total_loss += rep_loss
        if self.task_type == 'classification':
            cls_loss = self.soft_cross_entropy(student_logits / self.temperature, teacher_logits / self.temperature)
            if self.is_lgt_fit:
                student_logits = self.cast(student_logits, mstype.float32)
                label_ids_reshape = self.reshape(self.cast(label_ids, mstype.int32), (-1,))
                lgt_loss = self.lgt_fct(student_logits, label_ids_reshape)
                total_loss += lgt_loss
        else:
            student_logits = self.reshape(student_logits, (-1,))
            label_ids = self.reshape(label_ids, (-1,))
            cls_loss = self.loss_mse(student_logits, label_ids)
        total_loss += cls_loss
        return self.cast(total_loss, mstype.float32)


class BertTrainWithLossScaleCell(nn.Cell):
    """
    Specifically defined for finetuning where only four inputs tensor are needed.
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
        self.clip_type = gradient_cfg.clip_type
        self.clip_value = gradient_cfg.clip_value
        self.is_distributed = (self.parallel_mode != ParallelMode.STAND_ALONE)
        self.cast = P.Cast()
        self.alloc_status = P.NPUAllocFloatStatus()
        self.get_status = P.NPUGetFloatStatus()
        self.clear_before_grad = P.NPUClearFloatStatus()
        self.reduce_sum = P.ReduceSum(keep_dims=False)
        self.base = Tensor(1, mstype.float32)
        self.less_equal = P.LessEqual()
        self.hyper_map = C.HyperMap()
        self.loss_scale = None
        self.loss_scaling_manager = scale_update_cell
        if scale_update_cell:
            self.loss_scale = Parameter(Tensor(scale_update_cell.get_loss_scale(), dtype=mstype.float32))

        self.saved_params = self.weights.clone(prefix='saved')
        self.length = len(self.weights)
        self.quant_embedding_list = []
        self.quant_weight_list = []
        for i, key in enumerate(self.saved_params):
            if 'embedding_lookup' in key.name:
                self.quant_embedding_list.append(i)
            elif 'weight' in key.name and 'dense_1' not in key.name:
                self.quant_weight_list.append(i)
        self.quant_embedding_list_length = len(self.quant_embedding_list)
        self.quant_weight_list_length = len(self.quant_weight_list)

        self.quantize_embedding = QuantizeWeightCell(num_bits=network.embedding_bits,
                                                     compute_type=network.compute_type,
                                                     clip_value=network.weight_clip_value)
        self.quantize_weight = QuantizeWeightCell(num_bits=network.weight_bits,
                                                  compute_type=network.compute_type,
                                                  clip_value=network.weight_clip_value)

    @C.add_flags(has_effect=True)
    def construct(self,
                  input_ids,
                  input_mask,
                  token_type_id,
                  label_ids,
                  sens=None):
        """Defines the computation performed."""
        weights = self.weights
        for i in range(self.length):
            F.assign(self.saved_params[i], weights[i])

        for i in range(self.quant_embedding_list_length):
            quant_embedding = self.quantize_embedding(weights[self.quant_embedding_list[i]])
            F.assign(weights[self.quant_embedding_list[i]], quant_embedding)

        for i in range(self.quant_weight_list_length):
            quant_weight = self.quantize_weight(weights[self.quant_weight_list[i]])
            F.assign(weights[self.quant_weight_list[i]], quant_weight)

        if sens is None:
            scaling_sens = self.loss_scale
        else:
            scaling_sens = sens

        # alloc status and clear should be right before grad operation
        init = self.alloc_status()
        self.clear_before_grad(init)
        grads = self.grad(self.network, weights)(input_ids,
                                                 input_mask,
                                                 token_type_id,
                                                 label_ids,
                                                 self.cast(scaling_sens,
                                                           mstype.float32))
        # apply grad reducer on grads
        grads = self.grad_reducer(grads)
        grads = self.hyper_map(F.partial(grad_scale, scaling_sens * self.degree), grads)
        grads = self.hyper_map(F.partial(clip_grad, self.clip_type, self.clip_value), grads)

        for i in range(self.length):
            param = F.depend(self.saved_params[i], grads)
            F.assign(weights[i], param)

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
        if overflow:
            succ = False
        else:
            succ = self.optimizer(grads)
        return succ


class BertTrainCell(nn.Cell):
    """
    Specifically defined for finetuning where only four inputs tensor are needed.
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
        self.clip_type = gradient_cfg.clip_type
        self.clip_value = gradient_cfg.clip_value
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

        self.saved_params = self.weights.clone(prefix='saved')
        self.length = len(self.weights)
        self.quant_embedding_list = []
        self.quant_weight_list = []
        for i, key in enumerate(self.saved_params):
            if 'embedding_lookup' in key.name and 'min' not in key.name and 'max' not in key.name:
                self.quant_embedding_list.append(i)
            elif 'weight' in key.name and 'dense_1' not in key.name:
                self.quant_weight_list.append(i)
        self.quant_embedding_list_length = len(self.quant_embedding_list)
        self.quant_weight_list_length = len(self.quant_weight_list)

        self.quantize_embedding = QuantizeWeightCell(num_bits=network.embedding_bits,
                                                     compute_type=network.compute_type,
                                                     clip_value=network.weight_clip_value)
        self.quantize_weight = QuantizeWeightCell(num_bits=network.weight_bits,
                                                  compute_type=network.compute_type,
                                                  clip_value=network.weight_clip_value)

    def construct(self,
                  input_ids,
                  input_mask,
                  token_type_id,
                  label_ids):
        """Defines the computation performed."""
        weights = self.weights
        for i in range(self.length):
            F.assign(self.saved_params[i], weights[i])

        for i in range(self.quant_embedding_list_length):
            quant_embedding = self.quantize_embedding(weights[self.quant_embedding_list[i]])
            F.assign(weights[self.quant_embedding_list[i]], quant_embedding)

        for i in range(self.quant_weight_list_length):
            quant_weight = self.quantize_weight(weights[self.quant_weight_list[i]])
            F.assign(weights[self.quant_weight_list[i]], quant_weight)

        grads = self.grad(self.network, weights)(input_ids,
                                                 input_mask,
                                                 token_type_id,
                                                 label_ids,
                                                 self.cast(F.tuple_to_array((self.sens,)),
                                                           mstype.float32))
        # apply grad reducer on grads
        grads = self.grad_reducer(grads)
        grads = self.hyper_map(F.partial(clip_grad, self.clip_type, self.clip_value), grads)

        for i in range(self.length):
            param = F.depend(self.saved_params[i], grads)
            F.assign(weights[i], param)

        succ = self.optimizer(grads)
        return succ
