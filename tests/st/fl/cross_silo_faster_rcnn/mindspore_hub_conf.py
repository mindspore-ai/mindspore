# Copyright 2020-2021 Huawei Technologies Co., Ltd
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

from src.model_utils.config import config


if config.backbone in ("resnet_v1.5_50", "resnet_v1_101", "resnet_v1_152"):
    from src.FasterRcnn.faster_rcnn_resnet import Faster_Rcnn_Resnet
elif config.backbone == "resnet_v1_50":
    from src.FasterRcnn.faster_rcnn_resnet50v1 import Faster_Rcnn_Resnet

def create_network(name, *args, **kwargs):
    if name == "faster_rcnn":
        return Faster_Rcnn_Resnet(config=config)
    raise NotImplementedError(f"{name} is not implemented in the repo")
