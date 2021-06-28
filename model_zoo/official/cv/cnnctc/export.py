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
"""export checkpoint file into air, onnx, mindir models
   suggest run as python export.py --filename cnnctc --file_format MINDIR --ckpt_file [ckpt file path]
"""
import os
import numpy as np
from mindspore import Tensor, context, load_checkpoint, export
import mindspore.common.dtype as mstype
from src.cnn_ctc import CNNCTC_Model
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper


context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
if config.device_target == "Ascend":
    context.set_context(device_id=config.device_id)


def modelarts_pre_process():
    config.file_name = os.path.join(config.output_path, config.file_name)


@moxing_wrapper(pre_process=modelarts_pre_process)
def model_export():
    net = CNNCTC_Model(config.NUM_CLASS, config.HIDDEN_SIZE, config.FINAL_FEATURE_WIDTH)

    load_checkpoint(config.ckpt_file, net=net)

    bs = config.TEST_BATCH_SIZE

    input_data = Tensor(np.zeros([bs, 3, config.IMG_H, config.IMG_W]), mstype.float32)

    export(net, input_data, file_name=config.file_name, file_format=config.file_format)


if __name__ == '__main__':
    model_export()
