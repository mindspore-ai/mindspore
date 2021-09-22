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

"""tinybert_train_export."""

import os
import sys
import numpy as np
import mindspore as M
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.common import dtype as mstype
from mindspore.train.serialization import export
#pylint: disable=wrong-import-position
if len(sys.argv) > 1:
    path = sys.argv[1]
    del sys.argv[1]
else:
    path = ''
sys.path.append(os.environ['CLOUD_MODEL_ZOO'] + 'official/nlp/tinybert')

from official.nlp.tinybert.src.tinybert_model import TinyBertModel  # noqa: 402
from official.nlp.tinybert.src.model_utils.config import bert_student_net_cfg  # noqa: 402
from train_utils import save_t  # noqa: 402


class BertNetworkWithLossGenDistill(M.nn.Cell):
    """
    Provide bert pre-training loss through network.
    Args:
        config (BertConfig): The config of BertModel.
        is_training (bool): Specifies whether to use the training mode.
        use_one_hot_embeddings (bool): Specifies whether to use one-hot for embeddings. Default: False.
    Returns:
        Tensor, the loss of the network.
    """

    def __init__(self, student_config, is_training, use_one_hot_embeddings=False,
                 is_att_fit=False, is_rep_fit=True):
        super(BertNetworkWithLossGenDistill, self).__init__()
        # load teacher model
        self.bert = TinyBertModel(
            student_config, is_training, use_one_hot_embeddings)
        self.cast = P.Cast()
        self.fit_dense = M.nn.Dense(student_config.hidden_size,
                                    student_config.hidden_size).to_float(student_config.compute_type)
        self.student_layers_num = student_config.num_hidden_layers
        self.is_att_fit = is_att_fit
        self.is_rep_fit = is_rep_fit
        self.loss_mse = M.nn.MSELoss()
        self.select = P.Select()
        self.zeroslike = P.ZerosLike()
        self.dtype = student_config.dtype

    def construct(self,
                  input_ids_,
                  input_mask_,
                  token_type_id_,
                  label_):
        """general distill network with loss"""
        # student model
        _, _, _, student_seq_output, student_att_output = self.bert(
            input_ids_, token_type_id_, input_mask_)
        total_loss = 0
        if self.is_att_fit:
            selected_student_att_output = ()
            for i in range(self.student_layers_num):
                selected_student_att_output += (student_att_output[i],)
            att_loss = 0
            for i in range(self.student_layers_num):
                student_att = selected_student_att_output[i]
                student_att = self.select(student_att <= self.cast(-100.0, mstype.float32), self.zeroslike(student_att),
                                          student_att)
                att_loss += self.loss_mse(student_att, label_)
            total_loss += att_loss
        if self.is_rep_fit:
            selected_student_seq_output = ()
            fit_dense_out = self.fit_dense(student_seq_output[0])
            fit_dense_out = self.cast(fit_dense_out, self.dtype)
            selected_student_seq_output += (fit_dense_out,)
            rep_loss = 0
            student_rep = selected_student_seq_output[0]
            rep_loss += self.loss_mse(student_rep, label_)
            total_loss += rep_loss
        return self.cast(total_loss, mstype.float32)


class BertTrainCell(M.nn.Cell):
    """
    Encapsulation class of bert network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        sens (Number): The adjust parameter. Default: 1.0.
    """

    def __init__(self, network, optimizer_, sens=1.0):
        super(BertTrainCell, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.weights = optimizer_.parameters
        self.optimizer = optimizer_
        self.sens = sens
        self.grad = C.GradOperation(get_by_list=True,
                                    sens_param=True)
        self.reducer_flag = False
        self.grad_reducer = F.identity
        self.degree = 1
        self.cast = P.Cast()
        self.hyper_map = C.HyperMap()

    def construct(self,
                  input_ids_,
                  input_mask_,
                  token_type_id_, label_):
        """Defines the computation performed."""
        weights = self.weights
        loss = self.network(input_ids_,
                            input_mask_,
                            token_type_id_, label_)
        grads = self.grad(self.network, weights)(input_ids_,
                                                 input_mask_,
                                                 token_type_id_,
                                                 label_, self.cast(F.tuple_to_array((self.sens,)),
                                                                   mstype.float32))

        # apply grad reducer on grads
        grads = self.grad_reducer(grads)
        succ = self.optimizer(grads)
        return F.depend(loss, succ)


M.context.set_context(mode=M.context.PYNATIVE_MODE,
                      device_target="CPU", save_graphs=False)

# get epoch number
epoch_num = 1

input_ids_rnd = np.abs(np.random.randn(32, 128)*2)
input_ids_np = np.clip(input_ids_rnd, 0, 1)
input_ids = M.Tensor(input_ids_np, M.int32)
token_type_id = M.Tensor(np.zeros((32, 128), np.int32))
input_mask = M.Tensor(np.zeros((32, 128), np.int32))
label = M.Tensor(np.zeros((4096, 384), np.int32))


bert_student_net_cfg.hidden_dropout_prob = 0.0
bert_student_net_cfg.attention_probs_dropout_prob = 0.0
bert_student_net_cfg.compute_type = mstype.float32

#==============Training===============
nloss = BertNetworkWithLossGenDistill(
    bert_student_net_cfg, is_training=True, use_one_hot_embeddings=False)
optimizer = M.nn.Adam(nloss.bert.trainable_params(), learning_rate=1e-3, beta1=0.5, beta2=0.7,
                      eps=1e-2, use_locking=True, use_nesterov=False, weight_decay=0.1, loss_scale=0.3)
net = BertTrainCell(nloss, optimizer)
net.set_train(True)
export(net, input_ids, token_type_id, input_mask, label,
       file_name="tinybert_train", file_format='MINDIR')
y = net(input_ids, token_type_id, input_mask, label)
net.set_train(False)
out = net.network.bert(input_ids, token_type_id, input_mask)
y = out[3]

if len(sys.argv) > 1:
    save_t(input_ids, path+'tinybert_input1.bin')
    save_t(token_type_id, path + 'tinybert_input2.bin')
    save_t(input_mask, path + 'tinybert_input4.bin')
    save_t(label, path + 'tinybert_input3.bin')
    save_t(y[0], path + 'tinybert_output1.bin')
