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
"""export for retinanet"""
import os
import numpy as np
import mindspore.common.dtype as mstype
from mindspore import context, Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net, export
from src.retinanet import retinanet50, resnet50, retinanetInferWithDecoder
from src.model_utils.config import config
from src.box_utils import default_boxes
from src.model_utils.moxing_adapter import moxing_wrapper


def modelarts_pre_process():
    config.file_name = os.path.join(config.output_path, config.file_name)


@moxing_wrapper(pre_process=modelarts_pre_process)
def model_export():
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, device_id=config.device_id)

    backbone = resnet50(config.num_classes)
    net = retinanet50(backbone, config)
    net = retinanetInferWithDecoder(net, Tensor(default_boxes), config)
    param_dict = load_checkpoint(config.checkpoint_path)
    net.init_parameters_data()
    load_param_into_net(net, param_dict)
    net.set_train(False)
    shape = [config.export_batch_size, 3] + config.img_shape
    input_data = Tensor(np.zeros(shape), mstype.float32)
    export(net, input_data, file_name=config.file_name, file_format=config.file_format)


if __name__ == '__main__':
    model_export()
