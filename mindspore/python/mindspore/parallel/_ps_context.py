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
from mindspore._checkparam import Validator, Rel
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
    is_server = (_get_ps_context("ms_role") in ["MS_PSERVER", "MS_SERVER", "MS_SCHED"])
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
    "root_first_ca_path": ps_context().set_root_first_ca_path,
    "root_second_ca_path": ps_context().set_root_second_ca_path,
    "pki_verify": ps_context().set_pki_verify,
    "equip_crl_path": ps_context().set_equip_crl_path,
    "replay_attack_time_diff": ps_context().set_replay_attack_time_diff,
    "enable_ssl": ps_context().set_enable_ssl,
    "client_password": ps_context().set_client_password,
    "server_password": ps_context().set_server_password,
    "scheduler_manage_port": ps_context().set_scheduler_manage_port,
    "config_file_path": ps_context().set_config_file_path,
    "dp_eps": ps_context().set_dp_eps,
    "dp_delta": ps_context().set_dp_delta,
    "dp_norm_clip": ps_context().set_dp_norm_clip,
    "encrypt_type": ps_context().set_encrypt_type,
    "http_url_prefix": ps_context().set_http_url_prefix,
    "global_iteration_time_window": ps_context().set_global_iteration_time_window,
    "sign_k": ps_context().set_sign_k,
    "sign_eps": ps_context().set_sign_eps,
    "sign_thr_ratio": ps_context().set_sign_thr_ratio,
    "sign_global_lr": ps_context().set_sign_global_lr,
    "sign_dim_out": ps_context().set_sign_dim_out,
    "checkpoint_dir": ps_context().set_checkpoint_dir,
    "upload_compress_type": ps_context().set_upload_compress_type,
    "upload_sparse_rate": ps_context().set_upload_sparse_rate,
    "download_compress_type": ps_context().set_download_compress_type,
    "instance_name": ps_context().set_instance_name,
    "participation_time_level": ps_context().set_participation_time_level,
    "continuous_failure_times": ps_context().set_continuous_failure_times,
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
    "cipher_time_window": ps_context().cipher_time_window,
    "reconstruct_secrets_threshold": ps_context().reconstruct_secrets_threshold,
    "fl_name": ps_context().fl_name,
    "fl_iteration_num": ps_context().fl_iteration_num,
    "client_epoch_num": ps_context().client_epoch_num,
    "client_batch_size": ps_context().client_batch_size,
    "client_learning_rate": ps_context().client_learning_rate,
    "worker_step_num_per_iteration": ps_context().worker_step_num_per_iteration,
    "dp_eps": ps_context().dp_eps,
    "dp_delta": ps_context().dp_delta,
    "dp_norm_clip": ps_context().dp_norm_clip,
    "encrypt_type": ps_context().encrypt_type,
    "root_first_ca_path": ps_context().root_first_ca_path,
    "root_second_ca_path": ps_context().root_second_ca_path,
    "pki_verify": ps_context().pki_verify,
    "equip_crl_path": ps_context().equip_crl_path,
    "replay_attack_time_diff": ps_context().replay_attack_time_diff,
    "enable_ssl": ps_context().enable_ssl,
    "client_password": ps_context().client_password,
    "server_password": ps_context().server_password,
    "scheduler_manage_port": ps_context().scheduler_manage_port,
    "config_file_path": ps_context().config_file_path,
    "http_url_prefix": ps_context().http_url_prefix,
    "global_iteration_time_window": ps_context().global_iteration_time_window,
    "sign_k": ps_context().sign_k,
    "sign_eps": ps_context().sign_eps,
    "sign_thr_ratio": ps_context().sign_thr_ratio,
    "sign_global_lr": ps_context().sign_global_lr,
    "sign_dim_out": ps_context().sign_dim_out,
    "checkpoint_dir": ps_context().checkpoint_dir,
    "upload_compress_type": ps_context().upload_compress_type,
    "upload_sparse_rate": ps_context().upload_sparse_rate,
    "download_compress_type": ps_context().download_compress_type,
    "instance_name": ps_context().instance_name,
    "participation_time_level": ps_context().participation_time_level,
    "continuous_failure_times": ps_context().continuous_failure_times,
}

_check_positive_int_keys = ["server_num", "scheduler_port", "fl_server_port",
                            "start_fl_job_threshold", "start_fl_job_time_window", "update_model_time_window",
                            "fl_iteration_num", "client_epoch_num", "client_batch_size", "cipher_time_window",
                            "reconstruct_secrets_threshold"]

_check_non_negative_int_keys = ["worker_num"]

_check_positive_float_keys = ["update_model_ratio", "client_learning_rate"]

_check_port_keys = ["scheduler_port", "fl_server_port"]

_check_string_keys = {
    "upload_compress_type": ["NO_COMPRESS", "DIFF_SPARSE_QUANT"],
    "download_compress_type": ["NO_COMPRESS", "QUANT"],
}

_check_float_range_keys = {
    "upload_sparse_rate": {"lower_limit": 0.0, "upper_limit": 1.0, "rel": Rel.INC_RIGHT},
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
    kwargs = _check_conflict_value(kwargs)
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


def _insert_hash_table_size(name, cache_vocab_size, embedding_size, vocab_size, param_key=-1):
    ps_context().insert_hash_table_size(name, cache_vocab_size, embedding_size, vocab_size, param_key)


def _reinsert_hash_table_size(new_name, cur_name, cache_vocab_size, embedding_size):
    ps_context().reinsert_hash_table_size(new_name, cur_name, cache_vocab_size, embedding_size)


def _insert_weight_init_info(name, global_seed, op_seed):
    ps_context().insert_weight_init_info(name, global_seed, op_seed)


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


def _set_rank_id(rank_id):
    ps_context().set_rank_id(rank_id)


def _is_ps_mode():
    return _get_ps_context("server_mode") == "PARAMETER_SERVER"


def _is_fl_mode():
    return _get_ps_context("server_mode") in ("FEDERATED_LEARNING", "HYBRID_TRAINING")


def _enable_distributed_mindrt():
    '''
    Whether the distributed MindRT is enabled.
    This method is used to distinguish from old distributed training mode.
    '''
    return ps_context().enable_distributed_mindrt()


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

    if key in _check_string_keys:
        string_keys = _check_string_keys[key]
        Validator.check_string(value, string_keys)

    if key in _check_float_range_keys:
        range_keys = _check_float_range_keys[key]
        Validator.check_float_range(value, **range_keys)

    if key in _check_port_keys:
        if value < 1 or value > 65535:
            raise ValueError("The range of %s must be 1 to 65535, but got %d." % (key, value))


def _check_conflict_value(kwargs):
    if "upload_compress_type" in kwargs and "encrypt_type" in kwargs:
        if kwargs["upload_compress_type"] != "NO_COMPRESS" and kwargs["encrypt_type"] in ("SIGNDS", "PW_ENCRYPT"):
            logger.warning("The '{}' and '{}' are conflicted, and in '{}' mode the"
                           " 'upload_compress_type' will be 'NO_COMPRESS'".format(kwargs["encrypt_type"],
                                                                                  kwargs["upload_compress_type"],
                                                                                  kwargs["encrypt_type"]))
            kwargs["upload_compress_type"] = "NO_COMPRESS"
    return kwargs
