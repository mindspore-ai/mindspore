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

_set_ps_context_func_map = {
    "enable_ps": ps_context().set_ps_enable
}

_get_ps_context_func_map = {
    "enable_ps": ps_context().is_ps_mode
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
                           # MS_PSERVER represents the Server


    Args:
        enable_ps (bool): Whether to enable parameter server training mode.
                          Only after enable_ps is set True, the environment variables will be effective.
                          Default: False.

    Raises:
        ValueError: If input key is not the attribute in parameter server training mode context.

    Examples:
        >>> context.set_ps_context(enable_ps=True)
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

def _insert_hash_table_size(name, cache_vocab_size, embedding_size, vocab_size):
    ps_context().insert_hash_table_size(name, cache_vocab_size, embedding_size, vocab_size)

def _reinsert_hash_table_size(new_name, cur_name, cache_vocab_size, embedding_size):
    ps_context().reinsert_hash_table_size(new_name, cur_name, cache_vocab_size, embedding_size)

def _insert_weight_init_info(name, global_seed, op_seed):
    ps_context().insert_weight_init_info(name, global_seed, op_seed)

def _insert_accumu_init_info(name, init_val):
    ps_context().insert_accumu_init_info(name, init_val)

def _clone_hash_table(dest_param_name, src_param_name):
    ps_context().clone_hash_table(dest_param_name, src_param_name)

def _set_cache_enable(cache_enable):
    # Environment variables are used to specify a maximum number of OpenBLAS threads:
    # In ubuntu(GPU) environment, numpy will use too many threads for computing,
    if cache_enable:
        os.environ['OPENBLAS_NUM_THREADS'] = '2'
        os.environ['GOTO_NUM_THREADS'] = '2'
        os.environ['OMP_NUM_THREADS'] = '2'
    ps_context().set_cache_enable(cache_enable)

def _set_rank_id(rank_id):
    ps_context().set_rank_id(rank_id)
