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

import os
import numpy as np
from mindspore import dtype as mstype
from mindspore import context, Tensor, export
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.unet3d_model import UNet3d, UNet3d_
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper

device_id = int(os.getenv('DEVICE_ID'))
context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, save_graphs=False, device_id=device_id)

@moxing_wrapper()
def export_model(ckpt_path):
    if config.device_target == 'Ascend':
        network = UNet3d()
    else:
        network = UNet3d_()
    network.set_train(False)
    param_dict = load_checkpoint(ckpt_path)
    load_param_into_net(network, param_dict)
    image_shape = [config.batch_size, config.in_channels] + config.roi_size
    window_image = Tensor(np.zeros(image_shape), mstype.float32)
    export(network, window_image, file_name=config.file_name, file_format=config.file_format)

if __name__ == '__main__':
    export_model(ckpt_path=config.ckpt_file)
