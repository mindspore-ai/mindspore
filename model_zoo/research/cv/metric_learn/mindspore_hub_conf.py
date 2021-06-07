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
from src.resnet import resnet50
from src.config import config1, config2

def create_network(name, *args, **kwargs):
    """create net which should give the param if wants to use task_name, default is qudaruplet"""
    if name == "metric_learn":
        if "task_name" in kwargs:
            taskName = kwargs.get("task_name")

        else:
            taskName = "qudaruplet"

        if taskName == "qudaruplet":
            config = config2

        elif taskName == "triplet":
            config = config1

        net = resnet50(class_num=config.class_num)
        return net
    raise NotImplementedError(f"{name} is not implemented in the repo")
