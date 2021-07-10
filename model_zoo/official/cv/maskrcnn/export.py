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
"""export checkpoint file into air, onnx, mindir models"""

import numpy as np
from src.model_utils.config import config
from src.model_utils.device_adapter import get_device_id
from src.model_utils.moxing_adapter import moxing_wrapper
from src.maskrcnn.mask_rcnn_r50 import MaskRcnn_Infer
from mindspore import Tensor, context, load_checkpoint, load_param_into_net, export


if not config.enable_modelarts:
    config.ckpt_file = config.ckpt_file_local

context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
if config.device_target == "Ascend":
    context.set_context(device_id=get_device_id())

def modelarts_process():
    pass

@moxing_wrapper(pre_process=modelarts_process)
def export_maskrcnn():
    """ export_maskrcnn """
    config.test_batch_size = config.batch_size
    net = MaskRcnn_Infer(config=config)
    param_dict = load_checkpoint(config.ckpt_file)

    param_dict_new = {}
    for key, value in param_dict.items():
        param_dict_new["network." + key] = value

    load_param_into_net(net, param_dict_new)
    net.set_train(False)

    img = Tensor(np.zeros([config.batch_size, 3, config.img_height, config.img_width], np.float16))
    img_metas = Tensor(np.zeros([config.batch_size, 4], np.float16))

    input_data = [img, img_metas]
    export(net, *input_data, file_name=config.file_name, file_format=config.file_format)

if __name__ == '__main__':
    export_maskrcnn()
