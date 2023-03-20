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
from mindspore._checkparam import args_type_check, Validator


class _OffloadConfig:
    """
    The key of the Offload Config.
    """
    ENABLE_OFFLOAD = "enable_offload"
    OFFLOAD_PARAM = "offload_param"
    OFFLOAD_PATH = "offload_path"
    OFFLOAD_CHECKPOINT = "offload_checkpoint"
    OFFLOAD_DDR_SIZE = "offload_ddr_size"
    OFFLOAD_DISK_SIZE = "offload_disk_size"
    ENABLE_AIO = "enable_aio"
    AIO_BLOCK_SIZE = "aio_block_size"
    AIO_QUEUE_DEPTH = "aio_queue_depth"
    ENABLE_PINNED_MEM = "enable_pinned_mem"


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

    def set_offload_config(self, offload_config):
        """Set offfload context"""
        self.check_context_handle()
        enable_offload = _OffloadConfig.ENABLE_OFFLOAD
        offload_param = _OffloadConfig.OFFLOAD_PARAM
        offload_path = _OffloadConfig.OFFLOAD_PATH
        offload_checkpoint = _OffloadConfig.OFFLOAD_CHECKPOINT
        offload_ddr_size = _OffloadConfig.OFFLOAD_DDR_SIZE
        offload_disk_size = _OffloadConfig.OFFLOAD_DISK_SIZE
        enable_aio = _OffloadConfig.ENABLE_AIO
        aio_block_size = _OffloadConfig.AIO_BLOCK_SIZE
        aio_queue_depth = _OffloadConfig.AIO_QUEUE_DEPTH
        enable_pinned_mem = _OffloadConfig.ENABLE_PINNED_MEM

        for config_name in offload_config:
            unknown_config = []
            if config_name not in [enable_offload, offload_param, offload_path, offload_checkpoint,
                                   offload_ddr_size, offload_disk_size, enable_aio, aio_block_size,
                                   aio_queue_depth, enable_pinned_mem]:
                unknown_config.append(config_name)

            if unknown_config:
                raise ValueError("Unknown config: {}".format(unknown_config))

        if enable_offload in offload_config:
            Validator.check_bool(
                offload_config[enable_offload], enable_offload, enable_offload)
            self._context_handle.set_enable_offload(
                offload_config[enable_offload])

        if offload_param in offload_config:
            Validator.check_string(
                offload_config[offload_param].lower(), ["cpu", "disk"])
            self._context_handle.set_offload_param(
                offload_config[offload_param].lower())

        if offload_path in offload_config:
            if not isinstance(offload_config[offload_path], str):
                raise TypeError("For 'set_offload_path', "
                                "the argument 'offload_path' must be str, but got the type : {}."
                                .format(type(offload_config[offload_path])))
            self._context_handle.set_offload_path(
                offload_config[offload_path])
        if offload_checkpoint in offload_config:
            Validator.check_string(
                offload_config[offload_checkpoint].lower(), ["cpu", "disk"])
            self._context_handle.set_offload_checkpoint(
                offload_config[offload_checkpoint].lower())

        if offload_ddr_size in offload_config:
            Validator.check_positive_int(offload_config[offload_ddr_size])
            self._context_handle.set_offload_ddr_size(
                offload_config[offload_ddr_size])

        if offload_disk_size in offload_config:
            Validator.check_positive_int(offload_config[offload_disk_size])
            self._context_handle.set_offload_disk_size(
                offload_config[offload_disk_size])
        if enable_aio in offload_config:
            Validator.check_bool(
                offload_config[enable_aio], enable_aio, enable_aio)
            self._context_handle.set_enable_aio(
                offload_config[enable_aio])
        if aio_block_size in offload_config:
            Validator.check_positive_int(offload_config[aio_block_size])
            self._context_handle.set_aio_block_size(
                offload_config[aio_block_size])
        if aio_queue_depth in offload_config:
            Validator.check_positive_int(offload_config[aio_queue_depth])
            self._context_handle.set_aio_queue_depth(
                offload_config[aio_queue_depth])
        if enable_pinned_mem in offload_config:
            Validator.check_bool(
                offload_config[enable_pinned_mem], enable_pinned_mem, enable_pinned_mem)
            self._context_handle.set_enable_pinned_mem(
                offload_config[enable_pinned_mem])

    def offload_config(self):
        """Get config of offload"""
        self.check_context_handle()
        offload_config = {
            _OffloadConfig.ENABLE_OFFLOAD: self._context_handle.enable_offload(),
            _OffloadConfig.OFFLOAD_PARAM: self._context_handle.offload_param(),
            _OffloadConfig.OFFLOAD_PATH: self._context_handle.offload_path(),
            _OffloadConfig.OFFLOAD_CHECKPOINT: self._context_handle.offload_checkpoint(),
            _OffloadConfig.OFFLOAD_DDR_SIZE: self._context_handle.offload_ddr_size(),
            _OffloadConfig.OFFLOAD_DISK_SIZE: self._context_handle.offload_disk_size(),
            _OffloadConfig.ENABLE_AIO: self._context_handle.enable_aio(),
            _OffloadConfig.AIO_BLOCK_SIZE: self._context_handle.aio_block_size(),
            _OffloadConfig.AIO_QUEUE_DEPTH: self._context_handle.aio_queue_depth(),
            _OffloadConfig.ENABLE_PINNED_MEM: self._context_handle.enable_pinned_mem()
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
