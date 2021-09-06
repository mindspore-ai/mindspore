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
from mindspore._checkparam import Validator
from mindspore._c_expression import PSContext

_ps_context = None

_check_positive_int_keys = ["server_num", "scheduler_port", "fl_server_port",
                            "start_fl_job_threshold", "start_fl_job_time_window", "update_model_time_window",
                            "fl_iteration_num", "client_epoch_num", "client_batch_size", "scheduler_manage_port",
                            "cipher_time_window", "reconstruct_secrets_threshold"]

_check_non_negative_int_keys = ["worker_num"]

_check_positive_float_keys = ["update_model_ratio", "client_learning_rate"]

_check_port_keys = ["scheduler_port", "fl_server_port", "scheduler_manage_port"]


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
    "server_mode": ps_context().set_server_mode,
    "ms_role": ps_context().set_ms_role,
    "enable_ps": ps_context().set_ps_enable,
    "enable_fl": ps_context().set_ps_enable,
    "worker_num": ps_context().set_worker_num,
    "server_num": ps_context().set_server_num,
    "scheduler_ip": ps_context().set_scheduler_ip,
    "scheduler_port": ps_context().set_scheduler_port,
    "fl_server_port": ps_context().set_fl_server_port,
    "enable_fl_client": ps_context().set_fl_client_enable,
    "start_fl_job_threshold": ps_context().set_start_fl_job_threshold,
    "start_fl_job_time_window": ps_context().set_start_fl_job_time_window,
    "update_model_ratio": ps_context().set_update_model_ratio,
    "update_model_time_window": ps_context().set_update_model_time_window,
    "share_secrets_ratio": ps_context().set_share_secrets_ratio,
    "cipher_time_window": ps_context().set_cipher_time_window,
    "reconstruct_secrets_threshold": ps_context().set_reconstruct_secrets_threshold,
    "fl_name": ps_context().set_fl_name,
    "fl_iteration_num": ps_context().set_fl_iteration_num,
    "client_epoch_num": ps_context().set_client_epoch_num,
    "client_batch_size": ps_context().set_client_batch_size,
    "client_learning_rate": ps_context().set_client_learning_rate,
    "worker_step_num_per_iteration": ps_context().set_worker_step_num_per_iteration,
    "enable_ssl": ps_context().set_enable_ssl,
    "client_password": ps_context().set_client_password,
    "server_password": ps_context().set_server_password,
    "scheduler_manage_port": ps_context().set_scheduler_manage_port,
    "config_file_path": ps_context().set_config_file_path,
    "dp_eps": ps_context().set_dp_eps,
    "dp_delta": ps_context().set_dp_delta,
    "dp_norm_clip": ps_context().set_dp_norm_clip,
    "encrypt_type": ps_context().set_encrypt_type
}

_get_ps_context_func_map = {
    "server_mode": ps_context().server_mode,
    "ms_role": ps_context().ms_role,
    "enable_ps": ps_context().is_ps_mode,
    "enable_fl": ps_context().is_ps_mode,
    "worker_num": ps_context().worker_num,
    "server_num": ps_context().server_num,
    "scheduler_ip": ps_context().scheduler_ip,
    "scheduler_port": ps_context().scheduler_port,
    "fl_server_port": ps_context().fl_server_port,
    "enable_fl_client": ps_context().fl_client_enable,
    "start_fl_job_threshold": ps_context().start_fl_job_threshold,
    "start_fl_job_time_window": ps_context().start_fl_job_time_window,
    "update_model_ratio": ps_context().update_model_ratio,
    "update_model_time_window": ps_context().update_model_time_window,
    "share_secrets_ratio": ps_context().share_secrets_ratio,
    "cipher_time_window": ps_context().set_cipher_time_window,
    "reconstruct_secrets_threshold": ps_context().reconstruct_secrets_threshold,
    "fl_name": ps_context().fl_name,
    "fl_iteration_num": ps_context().fl_iteration_num,
    "client_epoch_num": ps_context().client_epoch_num,
    "client_batch_size": ps_context().client_batch_size,
    "client_learning_rate": ps_context().client_learning_rate,
    "worker_step_num_per_iteration": ps_context().worker_step_num_per_iteration,
    "enable_ssl": ps_context().enable_ssl,
    "client_password": ps_context().client_password,
    "server_password": ps_context().server_password,
    "scheduler_manage_port": ps_context().scheduler_manage_port,
    "config_file_path": ps_context().config_file_path
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
        _check_value(key, value)
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


def _check_value(key, value):
    """
    Validate the value for parameter server context keys.
    """
    if key in _check_positive_int_keys:
        Validator.check_positive_int(value, key)

    if key in _check_non_negative_int_keys:
        Validator.check_non_negative_int(value, key)

    if key in _check_positive_float_keys:
        Validator.check_positive_float(value, key)

    if key in _check_port_keys:
        if value < 1 or value > 65535:
            raise ValueError("The range of %s must be 1 to 65535, but got %d." % (key, value))
