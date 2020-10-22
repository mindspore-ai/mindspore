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
"""hub config."""
from mindspore.compression.quant import QuantizationAwareTraining
from src.yolo import YOLOV3DarkNet53
from src.config import ConfigYOLOV3DarkNet53


def create_network(name, *args, **kwargs):
    if name == "yolov3_darknet53_quant":
        yolov3_darknet53_quant = YOLOV3DarkNet53(is_training=False)
        config = ConfigYOLOV3DarkNet53()
        # convert fusion network to quantization aware network
        if config.quantization_aware:
            quantizer = QuantizationAwareTraining(bn_fold=True,
                                                  per_channel=[True, False],
                                                  symmetric=[True, False])
            yolov3_darknet53_quant = quantizer.quantize(yolov3_darknet53_quant)
        return yolov3_darknet53_quant
    raise NotImplementedError(f"{name} is not implemented in the repo")
