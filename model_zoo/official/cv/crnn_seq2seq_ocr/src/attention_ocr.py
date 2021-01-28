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
#" ============================================================================
"""
CRNN-Seq2Seq-OCR model.

"""

import numpy as np

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops.operations as P
import mindspore.common.dtype as mstype
from mindspore import context, Tensor
from mindspore.nn.loss.loss import _Loss
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.context import ParallelMode
from mindspore.communication.management import get_group_size
from mindspore.parallel._auto_parallel_context import auto_parallel_context

from src.seq2seq import Encoder, Decoder


class NLLLoss(_Loss):
    def __init__(self, reduction='mean'):
        super(NLLLoss, self).__init__(reduction)
        self.one_hot = P.OneHot()
        self.reduce_sum = P.ReduceSum()

    def construct(self, logits, label):
        label_one_hot = self.one_hot(label, F.shape(logits)[-1], F.scalar_to_array(1.0), F.scalar_to_array(0.0))
        loss = self.reduce_sum(-1.0 * logits * label_one_hot, (1,))
        return self.get_loss(loss)


class AttentionOCRInfer(nn.Cell):
    def __init__(self, batch_size, conv_out_dim, encoder_hidden_size, decoder_hidden_size,
                 decoder_output_size, max_length, dropout_p=0.1):
        super(AttentionOCRInfer, self).__init__()

        self.encoder = Encoder(batch_size=batch_size,
                               conv_out_dim=conv_out_dim,
                               hidden_size=encoder_hidden_size)

        self.decoder = Decoder(hidden_size=decoder_hidden_size,
                               output_size=decoder_output_size,
                               max_length=max_length,
                               dropout_p=dropout_p)

    def construct(self, img, decoder_input, decoder_hidden):
        '''
        get token output
        '''
        encoder_outputs = self.encoder(img)
        decoder_output, decoder_hidden, decoder_attention = self.decoder(
            decoder_input, decoder_hidden, encoder_outputs)
        return decoder_output, decoder_hidden, decoder_attention


class AttentionOCR(nn.Cell):
    def __init__(self, batch_size, conv_out_dim, encoder_hidden_size, decoder_hidden_size,
                 decoder_output_size, max_length, dropout_p=0.1):
        super(AttentionOCR, self).__init__()
        self.encoder = Encoder(batch_size=batch_size,
                               conv_out_dim=conv_out_dim,
                               hidden_size=encoder_hidden_size)
        self.decoder = Decoder(hidden_size=decoder_hidden_size,
                               output_size=decoder_output_size,
                               max_length=max_length,
                               dropout_p=dropout_p)
        self.init_decoder_hidden = Tensor(np.zeros((1, batch_size, decoder_hidden_size),
                                                   dtype=np.float16), mstype.float16)
        self.shape = P.Shape()
        self.split = P.Split(axis=1, output_num=max_length)
        self.concat = P.Concat()
        self.expand_dims = P.ExpandDims()
        self.argmax = P.Argmax()
        self.select = P.Select()

    def construct(self, img, decoder_inputs, decoder_targets, teacher_force):
        encoder_outputs = self.encoder(img)
        _, text_len = self.shape(decoder_inputs)
        decoder_outputs = ()
        decoder_input_tuple = self.split(decoder_inputs)
        decoder_target_tuple = self.split(decoder_targets)
        decoder_input = decoder_input_tuple[0]
        decoder_hidden = self.init_decoder_hidden

        for i in range(text_len):
            decoder_output, decoder_hidden, _ = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            topi = self.argmax(decoder_output)
            decoder_input_top = self.expand_dims(topi, 1)
            decoder_input = self.select(teacher_force, decoder_target_tuple[i], decoder_input_top)
            decoder_output = self.expand_dims(decoder_output, 0)
            decoder_outputs += (decoder_output,)
        outputs = self.concat(decoder_outputs)
        return outputs


class AttentionOCRWithLossCell(nn.Cell):
    """AttentionOCR with Loss"""
    def __init__(self, network, max_length):
        super(AttentionOCRWithLossCell, self).__init__()
        self.network = network
        self.loss = NLLLoss()
        self.shape = P.Shape()
        self.add = P.AddN()
        self.mean = P.ReduceMean()
        self.split = P.Split(axis=0, output_num=max_length)
        self.squeeze = P.Squeeze()
        self.cast = P.Cast()

    def construct(self, img, decoder_inputs, decoder_targets, teacher_force):
        decoder_outputs = self.network(img, decoder_inputs, decoder_targets, teacher_force)
        decoder_outputs = self.cast(decoder_outputs, mstype.float32)
        _, text_len = self.shape(decoder_targets)
        loss_total = ()
        decoder_output_tuple = self.split(decoder_outputs)
        for i in range(text_len):
            loss = self.loss(self.squeeze(decoder_output_tuple[i]), decoder_targets[:, i])
            loss = self.mean(loss)
            loss_total += (loss,)
        loss_output = self.add(loss_total)
        return loss_output


grad_scale = C.MultitypeFuncGraph("grad_scale")
@grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * P.Reciprocal()(scale)


class TrainingWrapper(nn.Cell):
    def __init__(self, network, optimizer, sens=1.0):
        super(TrainingWrapper, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.weights = ms.ParameterTuple(network.trainable_params())
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer = None

        # Set parallel_mode
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        if self.reducer_flag:
            mean = context.get_auto_parallel_context("gradients_mean")
            if auto_parallel_context().get_device_num_is_set():
                degree = context.get_auto_parallel_context("device_num")
            else:
                degree = get_group_size()
            self.grad_reducer = nn.DistributedGradReducer(optimizer.parameters, mean, degree)
        self.hyper_map = C.HyperMap()

    def construct(self, *args):
        weights = self.weights
        loss = self.network(*args)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(*args, sens)
        if self.reducer_flag:
            grads = self.grad_reducer(grads)
        return F.depend(loss, self.optimizer(grads))
