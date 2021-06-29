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
"""emnist_transfer_export."""

import sys
import numpy as np
from train_utils import save_inout_transfer, train_wrap
from emoji_model import EmojiModel
import mindspore.common.dtype as mstype
from mindspore import context, Tensor, nn
from mindspore.train.serialization import export

context.set_context(mode=context.GRAPH_MODE, device_target="GPU", save_graphs=False)

BATCH = 1
CLS_NUM = 10

#Export EMNIST_BB
n_bb = EmojiModel(wayc=26, use_bb=True, use_head=False)
# user can load weights of pre-trained backbone model at this stage using the load_checkpoint
x_bb = Tensor(np.random.randn(BATCH, 1, 28, 28), mstype.float32)
export(n_bb, x_bb, file_name="mindir/emnist_bb", file_format='MINDIR')

#Export EMNIST_HEAD
n = EmojiModel(wayc=26, use_bb=False, use_head=True)
loss_fn = nn.MSELoss()
optimizer = nn.Adam(n.trainable_params(), learning_rate=1e-2, beta1=0.5, beta2=0.7, eps=1e-2, use_locking=True,
                    use_nesterov=False, weight_decay=0.0, loss_scale=0.3)
net = train_wrap(n, loss_fn, optimizer)
x = Tensor(np.random.randn(BATCH, 256, 3, 3), mstype.float32)
label = Tensor(np.zeros([BATCH, 26]).astype(np.float32))
export(net, x, label, file_name="mindir/emnist_head", file_format='MINDIR')

if len(sys.argv) > 1:
    save_inout_transfer(sys.argv[1] + "emnist", x_bb, label, n_bb, n, net, sparse=False)
