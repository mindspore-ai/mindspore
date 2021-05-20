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
from src.models.cycle_gan import get_generator, get_discriminator
from src.utils.args import get_args

def create_network(name, *args, **kwargs):
    """create net which should give the params like trainable, style"""
    if name == "cyclegan":
        if "trainable" in kwargs:
            isTrained = kwargs.get("trainable")

        else:
            isTrained = False

        if isTrained:
            args_param = get_args("trainable")
        else:
            args_param = get_args("predict")

        if "style" in kwargs:
            style = kwargs.get("style")
        else:
            style = "G"

        if style == "G":
            net = get_generator(args_param)
        elif style == "D":
            net = get_discriminator(args_param)
        return net

    raise NotImplementedError(f"{name} is not implemented in the repo")
