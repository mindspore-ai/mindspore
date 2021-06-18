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
import numpy as np

from mindspore import Tensor, float32, context
from mindspore.train.serialization import load_checkpoint, load_param_into_net, export

from src.model import get_pose_net
from src.model_utils.config import config
from src.model_utils.device_adapter import get_device_id

if __name__ == '__main__':
    # set context
    device_id = get_device_id()
    context.set_context(mode=context.GRAPH_MODE,
                        device_target="Ascend", save_graphs=False, device_id=device_id)

    # init model
    model = get_pose_net(config, is_train=False)
    model.set_train(False)

    # load parameters
    ckpt_file = config.TEST.MODEL_FILE
    print('loading model ckpt from {}'.format(ckpt_file))
    load_param_into_net(model, load_checkpoint(ckpt_file))

    input_shape = [config.TEST.BATCH_SIZE, 3, config.MODEL.IMAGE_SIZE[1], config.MODEL.IMAGE_SIZE[0]]
    input_ids = Tensor(np.zeros(input_shape), float32)
    export(model, input_ids, file_name=config.EXPORT.FILE_NAME, file_format=config.EXPORT.FILE_FORMAT)
