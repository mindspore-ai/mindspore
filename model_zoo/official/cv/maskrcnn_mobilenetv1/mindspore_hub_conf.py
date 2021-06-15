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
import re
from src.maskrcnn_mobilenetv1.mask_rcnn_mobilenetv1 import Mask_Rcnn_Mobilenetv1
from src.model_utils.config import config

def config_(cfg):
    train_cls = [i for i in re.findall(r'[a-zA-Z\s]+', cfg.coco_classes) if i != ' ']
    cfg.coco_classes = np.array(train_cls)
    lss = [int(re.findall(r'[0-9]+', i)[0]) for i in cfg.feature_shapes]
    cfg.feature_shapes = [(lss[2*i], lss[2*i+1]) for i in range(int(len(lss)/2))]
    cfg.roi_layer = dict(type='RoIAlign', out_size=7, mask_out_size=14, sample_num=2)
    cfg.warmup_ratio = 1/3.0
    cfg.mask_shape = (28, 28)
    return cfg
config = config_(config)

def create_network(name, *args, **kwargs):
    if name == "maskrcnn_mobilenetv1":
        return Mask_Rcnn_Mobilenetv1(config=config)
    raise NotImplementedError(f"{name} is not implemented in the repo")
