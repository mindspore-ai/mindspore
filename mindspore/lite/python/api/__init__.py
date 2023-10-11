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
import os
import sys
import logging
import importlib
from importlib.abc import MetaPathFinder

from mindspore_lite.version import __version__
from mindspore_lite.context import Context
from mindspore_lite.converter import FmkType, Converter
from mindspore_lite.model import ModelType, Model, ModelParallelRunner, ModelGroup, ModelGroupFlag
from mindspore_lite.tensor import DataType, Format, Tensor
from mindspore_lite.lite_split import split_network, split_ir
from mindspore_lite.llm_engine import LLMReq, LLMEngineStatus, LLMRole, LLMEngine
from mindspore_lite import lite_infer

def mslite_add_path():
    """mslite add path."""
    pwd = os.path.dirname(os.path.realpath(__file__))
    akg_path = os.path.realpath(pwd)
    if akg_path not in sys.path:
        sys.path.insert(0, akg_path)
    else:
        sys.path.remove(akg_path)
        sys.path.insert(0, akg_path)


class MSLiteMetaPathFinder(MetaPathFinder):
    """class MSLiteMetaPath finder."""

    def find_module(self, fullname, path=None):
        """mslite find module."""
        akg_idx = fullname.find("mindspore_lite.akg")
        if akg_idx != -1:
            rname = fullname[akg_idx + 15:]
            return MSLiteMetaPathLoader(rname)
        return None


class MSLiteMetaPathLoader:
    """class MSLiteMetaPathLoader loader."""

    def __init__(self, rname):
        self.__rname = rname

    def load_module(self, fullname):
        """mslite load module."""
        if self.__rname in sys.modules:
            sys.modules.pop(self.__rname)
        mslite_add_path()
        importlib.import_module(self.__rname)
        if sys.modules.get(self.__rname) is None:
            raise ImportError("Can not find module: %s " % self.__rname)
        target_module = sys.modules.get(self.__rname)
        sys.modules[fullname] = target_module
        return target_module


sys.meta_path.insert(0, MSLiteMetaPathFinder())


__all__ = []
__all__.extend(__version__)
__all__.extend(context.__all__)
__all__.extend(converter.__all__)
__all__.extend(model.__all__)
__all__.extend(tensor.__all__)
__all__.extend(lite_split.__all__)
__all__.extend(llm_engine.__all__)
