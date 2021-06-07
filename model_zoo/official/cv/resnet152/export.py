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
"""Export DPN
suggest run as python export.py --file_name [filename] --file_format [file format] --checkpoint_path [ckpt path]
"""

import numpy as np
from mindspore import Tensor, context, load_checkpoint, load_param_into_net, export
from src.dpn import dpns
from src.model_utils.config import config


context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
if config.device_target == "Ascend":
    context.set_context(device_id=config.device_id)

if __name__ == "__main__":
    # define net
    backbone = config.backbone
    num_classes = config.num_classes
    net = dpns[backbone](num_classes=num_classes)

    # load checkpoint
    param_dict = load_checkpoint(config.checkpoint_path)
    load_param_into_net(net, param_dict)
    net.set_train(False)

    image = Tensor(np.zeros([1, 3, config.height, config.width], np.float32))
    export(net, image, file_name=config.file_name, file_format=config.file_format)
