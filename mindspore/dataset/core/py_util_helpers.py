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
# ==============================================================================
"""
General py_transforms_utils functions.
"""
import numpy as np


def is_numpy(img):
    """
    Check if the input image is Numpy format.

    Args:
        img: Image to be checked.

    Returns:
        Bool, True if input is Numpy image.
    """
    return isinstance(img, np.ndarray)
