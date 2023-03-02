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
"""Context for parameter server training mode"""

import os
from mindspore._c_expression import PSContext
from mindspore import context
from mindspore import log as logger

_ps_context = None


def ps_context():
    """
    Get the global _ps_context, if it is not created, create a new one.

    Returns:
        _ps_context, the global parameter server training mode context.
    """
    global _ps_context
    if _ps_context is None:
        _ps_context = PSContext.get_instance()
    return _ps_context


def _need_reset_device_target_for_ps(target):
    '''
    For Ascend backend, the card can't be occupied by multiple processes in distributed traning,
    so we need to reset the device target for some roles.
    '''
    is_server = (os.getenv('MS_ROLE') in ["MS_PSERVER", "MS_SERVER", "MS_SCHED"])
    return is_server and target == "Ascend"


def set_ps_enable(enable):
    """
    Set ps enable flag.
    """
    ps_context().set_ps_enable(enable)
    # If this is Server or Scheduler and device target is Ascend, reset the target to CPU
    if _need_reset_device_target_for_ps(context.get_context("device_target")):
        logger.info("Reset device target to CPU when set_ps_enable.")
        context.set_context(device_target="CPU")

_set_ps_context_func_map = {
    "server_mode": ps_context().set_server_mode,
    "ms_role": ps_context().set_ms_role,
    "enable_ps": set_ps_enable,
    "worker_num": ps_context().set_worker_num,
    "server_num": ps_context().set_server_num,
    "scheduler_ip": ps_context().set_scheduler_ip,
    "scheduler_port": ps_context().set_scheduler_port,
    "enable_ssl": ps_context().set_enable_ssl,
    "client_password": ps_context().set_client_password,
    "server_password": ps_context().set_server_password,
    "scheduler_manage_port": ps_context().set_scheduler_manage_port,
    "config_file_path": ps_context().set_config_file_path,
}

_get_ps_context_func_map = {
    "server_mode": ps_context().server_mode,
    "ms_role": ps_context().ms_role,
    "enable_ps": ps_context().is_ps_mode,
    "worker_num": ps_context().worker_num,
    "server_num": ps_context().server_num,
    "scheduler_ip": ps_context().scheduler_ip,
    "scheduler_port": ps_context().scheduler_port,
    "enable_ssl": ps_context().enable_ssl,
    "client_password": ps_context().client_password,
    "server_password": ps_context().server_password,
    "scheduler_manage_port": ps_context().scheduler_manage_port,
    "config_file_path": ps_context().config_file_path,
}

def _get_ps_mode_rank():
    ps_rank = ps_context().ps_rank_id()
    if ps_rank == -1:
        raise RuntimeError("The parameter server mode training is not enabled yet.")
    return ps_rank


def _set_ps_context(**kwargs):
    """
    Set parameter server training mode context.

    Note:
        Some other environment variables should also be set for parameter server training mode.
        These environment variables are listed below:

        .. code-block::

            MS_SERVER_NUM  # Server number
            MS_WORKER_NUM  # Worker number
            MS_SCHED_HOST  # Scheduler IP address
            MS_SCHED_PORT  # Scheduler port
            MS_ROLE        # The role of this process:
                           # MS_SCHED represents the scheduler,
                           # MS_WORKER represents the worker,
                           # MS_PSERVER/MS_SERVER represents the Server


    Args:
        enable_ps (bool): Whether to enable parameter server training mode.
                          Only after enable_ps is set True, the environment variables will be effective.
                          Default: False.
        config_file_path (string): Configuration file path used by recovery. Default: ''.
        scheduler_manage_port (int): scheduler manage port used to scale out/in. Default: 11202.
        enable_ssl (bool): Set PS SSL mode enabled or disabled. Default: False.
        client_password (str): Password to decrypt the secret key stored in the client certificate. Default: ''.
        server_password (str): Password to decrypt the secret key stored in the server certificate. Default: ''.

    Raises:
        ValueError: If input key is not the attribute in parameter server training mode context.

    Examples:
        >>> import mindspore as ms
        >>> ms.set_ps_context(enable_ps=True, enable_ssl=True, client_password='123456', server_password='123456')
    """
    for key, value in kwargs.items():
        if key not in _set_ps_context_func_map:
            raise ValueError("Set PS context keyword %s is not recognized!" % key)
        set_func = _set_ps_context_func_map[key]
        set_func(value)


def _get_ps_context(attr_key):
    """
    Get parameter server training mode context attribute value according to the key.

    Args:
        attr_key (str): The key of the attribute.

    Returns:
        Returns attribute value according to the key.

    Raises:
        ValueError: If input key is not attribute in auto parallel context.
    """
    if attr_key not in _get_ps_context_func_map:
        raise ValueError("Get PS context keyword %s is not recognized!" % attr_key)
    get_func = _get_ps_context_func_map[attr_key]
    value = get_func()
    return value


def _reset_ps_context():
    """
    Reset parameter server training mode context attributes to the default values:

    - enable_ps: False.
    """
    ps_context().reset()


def _is_role_worker():
    return ps_context().is_worker()


def _is_role_pserver():
    return ps_context().is_server()


def _is_role_sched():
    return ps_context().is_scheduler()


def _insert_hash_table_size(name, cache_vocab_size, embedding_size, vocab_size, param_key=-1):
    ps_context().insert_hash_table_size(name, cache_vocab_size, embedding_size, vocab_size, param_key)


def _reinsert_hash_table_size(new_name, cur_name, cache_vocab_size, embedding_size):
    ps_context().reinsert_hash_table_size(new_name, cur_name, cache_vocab_size, embedding_size)


def _insert_accumu_init_info(name, init_val):
    ps_context().insert_accumu_init_info(name, init_val)


def _clone_hash_table(dest_param_name, dest_param_key, src_param_name, src_param_key):
    ps_context().clone_hash_table(dest_param_name, dest_param_key, src_param_name, src_param_key)


def _set_cache_enable(cache_enable):
    # Environment variables are used to specify a maximum number of OpenBLAS threads:
    # In ubuntu(GPU) environment, numpy will use too many threads for computing,
    if cache_enable:
        os.environ['OPENBLAS_NUM_THREADS'] = '2'
        os.environ['GOTO_NUM_THREADS'] = '2'
        os.environ['OMP_NUM_THREADS'] = '2'
    ps_context().set_cache_enable(cache_enable)


def _cache_enable():
    return ps_context().cache_enable()


def _set_cache_size(cache_size):
    ps_context().set_cache_size(cache_size)


def _set_sparse_format(sparse_format):
    ps_context().set_sparse_format(sparse_format)


def _set_rank_id(rank_id):
    ps_context().set_rank_id(rank_id)


def _is_ps_mode():
    return _get_ps_context("server_mode") == "PARAMETER_SERVER"


def _enable_distributed_mindrt():
    '''
    Whether the distributed MindRT is enabled.
    This method is used to distinguish from old distributed training mode.
    '''
    return ps_context().enable_distributed_mindrt()
