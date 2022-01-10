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
# ============================================================================
"""
The MPI config, used to configure the MPI environment.
"""
import threading
from mindspore._c_expression import MpiConfig
from mindspore._checkparam import args_type_check


class _MpiConfig:
    """
    _MpiConfig is the config tool for controlling MPI

    Note:
        Create a config through instantiating MpiConfig object is not recommended.
        should use MpiConfig() to get the config since MpiConfig is singleton.
    """
    _instance = None
    _instance_lock = threading.Lock()

    def __init__(self):
        self._mpiconfig_handle = MpiConfig.get_instance()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance_lock.acquire()
            cls._instance = object.__new__(cls)
            cls._instance_lock.release()
        return cls._instance

    def __getattribute__(self, attr):
        value = object.__getattribute__(self, attr)
        if attr == "_mpiconfig_handle" and value is None:
            raise ValueError("mpiconfig handle is none in MpiConfig!!!")
        return value

    @property
    def enable_mpi(self):
        """Get enable mpi."""
        return self._mpiconfig_handle.get_enable_mpi()

    @enable_mpi.setter
    def enable_mpi(self, enable_mpi):
        self._mpiconfig_handle.set_enable_mpi(enable_mpi)

_k_mpi_config = None


def _mpi_config():
    """
    Get the global mpi config, if mpi config is not created, create a new one.

    Returns:
        _MpiConfig, the global mpi config.
    """
    global _k_mpi_config
    if _k_mpi_config is None:
        _k_mpi_config = _MpiConfig()
    return _k_mpi_config


@args_type_check(enable_mpi=bool)
def _set_mpi_config(**kwargs):
    """
    Sets mpi config for running environment.

    mpi config should be configured before running your program. If there is no configuration,
    mpi module will be disabled by default.

    Note:
        Attribute name is required for setting attributes.

    Args:
        enable_mpi (bool): Whether to enable mpi. Default: False.

    Raises:
        ValueError: If input key is not an attribute in mpi config.

    Examples:
        >>> mpiconfig.set_mpi_config(enable_mpi=True)
    """
    for key, value in kwargs.items():
        if not hasattr(_mpi_config(), key):
            raise ValueError("Set mpi config keyword %s is not recognized!" % key)
        setattr(_mpi_config(), key, value)


def _get_mpi_config(attr_key):
    """
    Gets mpi config attribute value according to the input key.

    Args:
        attr_key (str): The key of the attribute.

    Returns:
        Object, The value of given attribute key.

    Raises:
        ValueError: If input key is not an attribute in config.
    """
    if not hasattr(_mpi_config(), attr_key):
        raise ValueError("Get context keyword %s is not recognized!" % attr_key)
    return getattr(_mpi_config(), attr_key)
