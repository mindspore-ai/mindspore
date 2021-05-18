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

"""Create dataset"""

from .cityscapes import CityscapesDataSet

__img_factory = {
    'Cityscapes': CityscapesDataSet,
}

__vid_factory = {
    'Cityscapes': CityscapesDataSet,
}

def get_names():
    return list(__img_factory.keys()) + list(__vid_factory.keys())

def init_img_dataset(name, **kwargs):
    if name not in __img_factory.keys():
        raise KeyError("Invalid dataset, got '{}', but expected to be one of {}".format(name, __img_factory.keys()))
    return __img_factory[name](**kwargs)

def init_vid_dataset(name, **kwargs):
    if name not in __vid_factory.keys():
        raise KeyError("Invalid dataset, got '{}', but expected to be one of {}".format(name, __vid_factory.keys()))
    return __vid_factory[name](**kwargs)
