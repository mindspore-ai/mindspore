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
import numpy as np
from mindspore import Tensor
from mindspore import context
from mindspore.communication.management import init
from mindspore.parallel import set_algo_parameters
from mindspore.train.model import Model
from mindspore.context import ParallelMode
from mindspore.communication._comm_helper import GlobalComm
from .test_auto_parallel_resnet import resnet50

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
context.set_context.__wrapped__(device_id=0)
GlobalComm.CHECK_ENVS = False
init()
GlobalComm.CHECK_ENVS = True

def test_train_32k_8p(batch_size=32, num_classes=32768):
    dev_num = 8
    context.set_auto_parallel_context(parallel_mode=ParallelMode.AUTO_PARALLEL, device_num=dev_num, full_batch=True)
    set_algo_parameters(elementwise_op_strategy_follow=True)
    np.random.seed(6)
    input_np = Tensor(np.ones([batch_size, 3, 224, 224]).astype(np.float32))
    net = resnet50(num_classes)
    model = Model(net)
    model.predict(input_np)
