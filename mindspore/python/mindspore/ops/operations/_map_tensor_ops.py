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

"""MapTensor operators."""

from mindspore.ops import signature as sig
from mindspore.ops.primitive import Primitive, prim_attr_register


class MapTensorGet(Primitive):
    """
    Get or create value according the key tensor and default value.
    """
    __mindspore_signature__ = (
        sig.make_sig('map_tensor'),
        sig.make_sig('key_tensor'),
        sig.make_sig('default_value'))

    @prim_attr_register
    def __init__(self):
        """Initialize MapTensorGet"""
        self.init_prim_io_names(inputs=['map_tensor', 'key_tensor', 'default_value'], outputs=['output'])
        self.add_prim_attr('side_effect_mem', True)


class MapTensorPut(Primitive):
    """
    Insert or update key value tensor pairs.
    """
    __mindspore_signature__ = (
        sig.make_sig('map_tensor'),
        sig.make_sig('key_tensor'),
        sig.make_sig('value_tensor'))

    @prim_attr_register
    def __init__(self):
        """Initialize MapTensorPut"""
        self.init_prim_io_names(inputs=['map_tensor', 'key_tensor', 'value_tensor'], outputs=['output'])
        self.add_prim_attr('side_effect_mem', True)


class MapTensorErase(Primitive):
    """
    Remove records according the key tensor.
    """
    __mindspore_signature__ = (
        sig.make_sig('map_tensor'),
        sig.make_sig('key_tensor'))

    @prim_attr_register
    def __init__(self):
        """Initialize MapTensorErase"""
        self.init_prim_io_names(inputs=['map_tensor', 'key_tensor'], outputs=['output'])
        self.add_prim_attr('side_effect_mem', True)

get = MapTensorGet()
put = MapTensorPut()
erase = MapTensorErase()
