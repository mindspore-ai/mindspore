# Copyright 2019-2022 Huawei Technologies Co., Ltd
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
This module is to support common data augmentations. Some operations are implemented in C++
to provide high performance.
Other operations are implemented in Python including using NumPy.

Common imported modules in corresponding API examples are as follows:

.. code-block::

    import mindspore.dataset as ds
    import mindspore.dataset.transforms as transforms

Note: Legacy c_transforms and py_transforms are deprecated but can still be imported as follows:

.. code-block::

    from mindspore.dataset.transforms import c_transforms
    from mindspore.dataset.transforms import py_transforms

See `Common Transforms
<https://www.mindspore.cn/tutorials/en/master/beginner/transforms.html#common-transforms>`_ tutorial for more details.

Descriptions of common data processing terms are as follows:

- TensorOperation, the base class of all data processing operations implemented in C++.
- PyTensorOperation, the base class of all data processing operations implemented in Python.

Note: In eager mode, non-NumPy input is implicitly converted to NumPy format and sent to MindSpore.
"""
from .. import vision
from . import c_transforms
from . import py_transforms
from . import transforms
from .transforms import Compose, Concatenate, Duplicate, Fill, Mask, OneHot, PadEnd, Plugin, RandomApply, \
    RandomChoice, RandomOrder, Relational, Slice, TypeCast, Unique, not_random
