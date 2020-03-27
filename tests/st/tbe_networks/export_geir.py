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
import os
import numpy as np
from resnet_torch import resnet50
from mindspore.train.callback import Callback
from mindspore.nn.optim.momentum import Momentum
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore import Tensor
import mindspore.nn as nn
from mindspore import context

from mindspore.train.serialization import save, load, save_checkpoint, load_checkpoint,\
                                          load_param_into_net, _exec_save_checkpoint,\
                                          _check_filedir_or_create, _chg_model_file_name_if_same_exist, \
                                          _read_file_last_line, context, export

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", 
enable_task_sink=True,enable_loop_sink=True,enable_ir_fusion=True)

def test_resnet50_export(batch_size=1, num_classes=5):
    context.set_context(enable_ir_fusion=False)
    input_np = np.random.uniform(0.0, 1.0, size=[batch_size, 3, 224, 224]).astype(np.float32)
    net = resnet50(batch_size, num_classes)
    #param_dict = load_checkpoint("./resnet50-1_103.ckpt")
    #load_param_into_net(net, param_dict)
    export(net, Tensor(input_np), file_name="./me_resnet50.pb", file_format="GEIR")
