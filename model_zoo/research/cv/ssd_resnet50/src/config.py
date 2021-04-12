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

"""Config parameters for SSD models."""

from .config_ssd_resnet50 import config as config_ssd_resnet50

using_model = "ssd_resnet50"

config_map = {
    "ssd_resnet50": config_ssd_resnet50
}


print("...............using "+using_model+" model................")
config = config_map[using_model]


if config.num_ssd_boxes == -1:
    num = 0
    h, w = config.img_shape
    for i in range(len(config.steps)):
        num += (h // config.steps[i]) * (w // config.steps[i]) * config.num_default[i]
    config.num_ssd_boxes = num
