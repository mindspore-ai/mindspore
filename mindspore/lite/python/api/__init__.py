# Copyright 2022 Huawei Technologies Co., Ltd
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
"""
MindSpore Lite Python API.
"""
from __future__ import absolute_import

from mindspore_lite.version import __version__
from mindspore_lite.context import Context, DeviceInfo, CPUDeviceInfo, GPUDeviceInfo, AscendDeviceInfo
from mindspore_lite.converter import FmkType, Converter
from mindspore_lite.model import ModelType, Model, RunnerConfig, ModelParallelRunner
from mindspore_lite.tensor import DataType, Format, Tensor

__all__ = []
__all__.extend(__version__)
__all__.extend(context.__all__)
__all__.extend(converter.__all__)
__all__.extend(model.__all__)
__all__.extend(tensor.__all__)
