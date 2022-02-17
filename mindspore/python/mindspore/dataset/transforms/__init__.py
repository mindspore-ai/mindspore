# Copyright 2019 Huawei Technologies Co., Ltd
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
This module is to support common augmentations. C_transforms is a high performance
augmentation module which is developed by C++. Py_transforms provides an optional
implementation which is developed by Python & NumPy.

Common imported modules in corresponding API examples are as follows:

.. code-block::

    import mindspore.dataset as ds
    import mindspore.dataset.vision.c_transforms as c_vision
    import mindspore.dataset.vision.py_transforms as py_vision
    from mindspore.dataset.transforms import c_transforms
    from mindspore.dataset.transforms import py_transforms
"""
from .. import vision
from . import c_transforms
from . import py_transforms
