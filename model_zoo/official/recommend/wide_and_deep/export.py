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
"""
##############export checkpoint file into air, mindir and onnx models#################
"""
import numpy as np
from mindspore import Tensor, context, load_checkpoint, export, load_param_into_net
from eval import ModelBuilder

from src.model_utils.device_adapter import get_device_id
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper


context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
if config.device_target == "Ascend":
    context.set_context(device_id=get_device_id())

def modelarts_pre_process():
    pass

@moxing_wrapper(pre_process=modelarts_pre_process)
def export_widedeep():
    """ export_widedeep """
    net_builder = ModelBuilder()
    _, eval_net = net_builder.get_net(config)

    param_dict = load_checkpoint(config.ckpt_file)
    load_param_into_net(eval_net, param_dict)
    eval_net.set_train(False)

    ids = Tensor(np.ones([config.eval_batch_size, config.field_size]).astype(np.int32))
    wts = Tensor(np.ones([config.eval_batch_size, config.field_size]).astype(np.float32))
    label = Tensor(np.ones([config.eval_batch_size, 1]).astype(np.float32))
    input_tensor_list = [ids, wts, label]
    export(eval_net, *input_tensor_list, file_name=config.file_name, file_format=config.file_format)

if __name__ == '__main__':
    export_widedeep()
