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
"""
This module is to support vision augmentations. It includes two parts:
c_transforms and py_transforms. C_transforms is a high performance
image augmentation module which is developed with c++ opencv. Py_transforms
provide more kinds of image augmentations which is developed with Python PIL.
"""
from . import c_transforms
from . import py_transforms
from .utils import Inter, Border, ImageBatchFormat
