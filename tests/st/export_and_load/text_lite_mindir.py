# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
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
"""mobilenetv2 export mindir."""
import os
import numpy as np
import pytest

from mindspore import Tensor
from mindspore.train.serialization import export, load_checkpoint
from mindspore import context
from model_zoo.official.cv.mobilenetv2.src.mobilenetV2 import MobileNetV2Backbone, MobileNetV2Head, mobilenet_v2


context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
ckpt_path = '/home/workspace/mindspore_dataset/checkpoint/mobilenetv2/mobilenetv2_gpu.ckpt'

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_export_mobilenetv2_gpu_mindir():
    backbone_net = MobileNetV2Backbone()
    head_net = MobileNetV2Head(input_channel=backbone_net.out_channels, num_classes=1000)
    net = mobilenet_v2(backbone_net, head_net)
    load_checkpoint(ckpt_path, net)
    input_tensor = Tensor(np.ones([1, 3, 224, 224]).astype(np.float32))
    export(net, input_tensor, file_name="mobilenetv2_gpu", file_format="MINDIR")
    output = net(input_tensor).asnumpy().tolist()
    file_obj = open('mobilenetv2_gpu_output.txt', 'w')
    file_obj.write("output 1 3 224 224\n")
    for num in output[0]:
        file_obj.write(str(num))
    file_obj.close()
    assert os.path.exists("mobilenetv2_gpu.mindir")
    assert os.path.exists("mobilenetv2_gpu_output.txt")
