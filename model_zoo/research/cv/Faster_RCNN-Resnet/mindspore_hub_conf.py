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
"""hub config."""
import argparse
import src.config as cfg

parser = argparse.ArgumentParser(description="FasterRcnn")
parser.add_argument("--backbone", type=str, required=True, \
                    help="backbone network name, options:resnet50v1.0, resnet50v1.5, resnet101 ,resnet152")
args_opt = parser.parse_args()

if args_opt.backbone in ("resnet50v1.5", "resnet101", "resnet152"):
    from src.FasterRcnn.faster_rcnn_resnet import Faster_Rcnn_Resnet
    if args_opt.backbone == "resnet50v1.5":
        config = cfg.get_config("./src/config_50.yaml")
    elif args_opt.backbone == "resnet101":
        config = cfg.get_config("./src/config_101.yaml")
    elif args_opt.backbone == "resnet152":
        config = cfg.get_config("./src/config_152.yaml")

elif args_opt.backbone == "resnet50v1.0":
    config = cfg.get_config("./src/config_50.yaml")
    from src.FasterRcnn.faster_rcnn_resnet50v1 import Faster_Rcnn_Resnet

def create_network(name, *args, **kwargs):
    if name == "faster_rcnn":
        return Faster_Rcnn_Resnet(config=config)
    raise NotImplementedError(f"{name} is not implemented in the repo")
