# Copyright 2023 Huawei Technologies Co., Ltd
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
"""Context of offload"""
from __future__ import absolute_import
from __future__ import division

import threading

from mindspore._c_expression import OffloadContext
from mindspore._checkparam import args_type_check
from mindspore import _checkparam as Validator

K_RE_PATTERN = r'[1-9][0-9]*(\.)?[0-9]*GB|0\.[0-9]*GB'
K_GBTOBYTE = 1 << 30


class _OffloadConfig:
    """
    The key of the Offload Config.
    """
    OFFLOAD_PARAM = "offload_param"
    OFFLOAD_PATH = "offload_path"
    OFFLOAD_CHECKPOINT = "offload_checkpoint"
    OFFLOAD_DDR_SIZE = "offload_ddr_size"
    OFFLOAD_DISK_SIZE = "offload_disk_size"
    ENABLE_AIO = "enable_aio"
    AIO_BLOCK_SIZE = "aio_block_size"
    AIO_QUEUE_DEPTH = "aio_queue_depth"
    ENABLE_PINNED_MEM = "enable_pinned_mem"
    AUTO_OFFLOAD = "auto_offload"
    HOST_MEM_BLOCk_SIZE = "host_mem_block_size"


class _OffloadContext:
    """
    _OffloadContext is the configuration for offload.

    Note:
        Create a context through instantiating Context object is not recommended.
        Should use offload_context() to get the context since Context is singleton.
    """
    _instance = None
    _instance_lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance_lock.acquire()
            cls._instance = object.__new__(cls)
            cls._instance_lock.release()
        return cls._instance

    def __init__(self):
        self._context_handle = OffloadContext.get_instance()

    def check_context_handle(self):
        """
        Check context handle.

        Raises:
            ValueError: If the context handle is none.
        """
        if self._context_handle is None:
            raise ValueError("Context handle is none in context!!!")

    def set_offload_param(self, offload_param):
        Validator.check_string(offload_param.lower(), ["cpu", "disk"])
        self._context_handle.set_offload_param(offload_param.lower())

    def set_offload_path(self, offload_path):
        if not isinstance(offload_path, str):
            raise TypeError("For 'set_offload_path', "
                            "the argument 'offload_path' must be str, but got the type : {}."
                            .format(type(offload_path)))
        self._context_handle.set_offload_path(offload_path)

    def set_offload_checkpoint(self, offload_checkpoint):
        Validator.check_string(offload_checkpoint.lower(), ["cpu", "disk"])
        self._context_handle.set_offload_checkpoint(offload_checkpoint.lower())

    def set_offload_ddr_size(self, offload_ddr_size):
        if not Validator.check_str_by_regular(offload_ddr_size, K_RE_PATTERN):
            raise ValueError("The argument 'offload_ddr_size' should be in correct "
                             " format! It must be a string ending with 'GB', in addition to that, it must contain "
                             "only numbers or decimal points, such as \"5GB\" or \"3.5GB\", but got {}."
                             .format(offload_ddr_size))
        ddr_size = float(offload_ddr_size[:-2])
        self._context_handle.set_offload_ddr_size(int(ddr_size * K_GBTOBYTE))

    def set_offload_disk_size(self, offload_disk_size):
        if not Validator.check_str_by_regular(offload_disk_size, K_RE_PATTERN):
            raise ValueError("The argument 'offload_disk_size' should be in correct "
                             " format! It must be a string ending with 'GB', in addition to that, it must contain "
                             "only numbers or decimal points, such as \"5GB\" or \"3.5GB\", but got {}."
                             .format(offload_disk_size))
        disk_size = float(offload_disk_size[:-2])
        self._context_handle.set_offload_disk_size(int(disk_size * K_GBTOBYTE))

    def set_enable_aio(self, enable_aio):
        Validator.check_bool(enable_aio)
        self._context_handle.set_enable_aio(enable_aio)

    def set_aio_block_size(self, aio_block_size):
        if not Validator.check_str_by_regular(aio_block_size, K_RE_PATTERN):
            raise ValueError("The argument 'aio_block_size' should be in correct "
                             " format! It must be a string ending with 'GB', in addition to that, it must contain "
                             "only numbers or decimal points, such as \"5GB\" or \"3.5GB\", but got {}."
                             .format(aio_block_size))
        aio_size = float(aio_block_size[:-2])
        self._context_handle.set_aio_block_size(int(aio_size * K_GBTOBYTE))

    def set_aio_queue_depth(self, aio_queue_depth):
        Validator.check_positive_int(aio_queue_depth)
        self._context_handle.set_aio_queue_depth(aio_queue_depth)

    def set_enable_pinned_mem(self, enable_pinned_mem):
        Validator.check_bool(
            enable_pinned_mem, enable_pinned_mem, enable_pinned_mem)
        self._context_handle.set_enable_pinned_mem(enable_pinned_mem)

    def set_auto_offload(self, auto_offload):
        Validator.check_bool(auto_offload)
        self._context_handle.set_auto_offload(auto_offload)

    def set_host_mem_block_size(self, host_mem_block_size):
        if not Validator.check_str_by_regular(host_mem_block_size, K_RE_PATTERN):
            raise ValueError("The argument 'host_mem_block_size' should be in correct "
                             " format! It must be a string ending with 'GB', in addition to that, it must contain "
                             "only numbers or decimal points, such as \"5GB\" or \"3.5GB\", but got {}."
                             .format(host_mem_block_size))
        block_size = float(host_mem_block_size[:-2])
        self._context_handle.set_host_mem_block_size(
            int(block_size * K_GBTOBYTE))

    def set_offload_config(self, offload_config):
        """Set offfload context"""
        self.check_context_handle()
        offload_param = _OffloadConfig.OFFLOAD_PARAM
        offload_path = _OffloadConfig.OFFLOAD_PATH
        offload_checkpoint = _OffloadConfig.OFFLOAD_CHECKPOINT
        offload_ddr_size = _OffloadConfig.OFFLOAD_DDR_SIZE
        offload_disk_size = _OffloadConfig.OFFLOAD_DISK_SIZE
        enable_aio = _OffloadConfig.ENABLE_AIO
        aio_block_size = _OffloadConfig.AIO_BLOCK_SIZE
        aio_queue_depth = _OffloadConfig.AIO_QUEUE_DEPTH
        enable_pinned_mem = _OffloadConfig.ENABLE_PINNED_MEM
        auto_offload = _OffloadConfig.AUTO_OFFLOAD
        host_mem_block_size = _OffloadConfig.HOST_MEM_BLOCk_SIZE

        for config_name in offload_config:
            unknown_config = []
            if config_name not in [offload_param, offload_path, offload_checkpoint,
                                   offload_ddr_size, offload_disk_size, enable_aio, aio_block_size,
                                   aio_queue_depth, enable_pinned_mem, auto_offload, host_mem_block_size]:
                unknown_config.append(config_name)

            if unknown_config:
                raise ValueError("Unknown config: {}".format(unknown_config))
            func = _set_offload_context_func_map.get(config_name, None)
            if not func:
                raise ValueError(
                    "Can not find set function: {}".format(config_name))
            func(offload_config[config_name])

    def offload_config(self):
        """Get config of offload"""
        self.check_context_handle()
        offload_config = {
            _OffloadConfig.OFFLOAD_PARAM: self._context_handle.offload_param(),
            _OffloadConfig.OFFLOAD_PATH: self._context_handle.offload_path(),
            _OffloadConfig.OFFLOAD_CHECKPOINT: self._context_handle.offload_checkpoint(),
            _OffloadConfig.OFFLOAD_DDR_SIZE: self._context_handle.offload_ddr_size(),
            _OffloadConfig.OFFLOAD_DISK_SIZE: self._context_handle.offload_disk_size(),
            _OffloadConfig.ENABLE_AIO: self._context_handle.enable_aio(),
            _OffloadConfig.AIO_BLOCK_SIZE: self._context_handle.aio_block_size(),
            _OffloadConfig.AIO_QUEUE_DEPTH: self._context_handle.aio_queue_depth(),
            _OffloadConfig.ENABLE_PINNED_MEM: self._context_handle.enable_pinned_mem(),
            _OffloadConfig.AUTO_OFFLOAD: self._context_handle.auto_offload(),
            _OffloadConfig.HOST_MEM_BLOCk_SIZE: self._context_handle.host_mem_block_size(),
        }
        return offload_config


_OFFLOAD_CONTEXT = None


def offload_context():
    """Get offload_context. if it is not created, create a new one."""
    global _OFFLOAD_CONTEXT
    if _OFFLOAD_CONTEXT is None:
        _OFFLOAD_CONTEXT = _OffloadContext()
    return _OFFLOAD_CONTEXT


@args_type_check(offload_config=dict)
def _set_offload_context(offload_config):
    offload_context().set_offload_config(offload_config)


def _get_offload_context():
    return offload_context().offload_config()


_set_offload_context_func_map = {
    _OffloadConfig.OFFLOAD_PARAM: offload_context().set_offload_param,
    _OffloadConfig.OFFLOAD_PATH: offload_context().set_offload_path,
    _OffloadConfig.OFFLOAD_CHECKPOINT: offload_context().set_offload_checkpoint,
    _OffloadConfig.OFFLOAD_DDR_SIZE: offload_context().set_offload_ddr_size,
    _OffloadConfig.OFFLOAD_DISK_SIZE: offload_context().set_offload_disk_size,
    _OffloadConfig.ENABLE_AIO: offload_context().set_enable_aio,
    _OffloadConfig.AIO_BLOCK_SIZE: offload_context().set_aio_block_size,
    _OffloadConfig.AIO_QUEUE_DEPTH: offload_context().set_aio_queue_depth,
    _OffloadConfig.ENABLE_PINNED_MEM: offload_context().set_enable_pinned_mem,
    _OffloadConfig.AUTO_OFFLOAD: offload_context().set_auto_offload,
    _OffloadConfig.HOST_MEM_BLOCk_SIZE: offload_context().set_host_mem_block_size,
}
