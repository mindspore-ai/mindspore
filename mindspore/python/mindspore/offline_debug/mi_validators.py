# Copyright 2021 Huawei Technologies Co., Ltd
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
# ==============================================================================
"""
Validator Functions for Offline Debugger APIs.
"""
from functools import wraps

import mindspore.offline_debug.dbg_services as cds
from mindspore.offline_debug.mi_validator_helpers import parse_user_args, type_check, \
    type_check_list, check_dir, check_uint32, check_uint64, check_iteration, check_param_id


def check_init(method):
    """Wrapper method to check the parameters of DbgServices init."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [dump_file_path], _ = parse_user_args(method, *args, **kwargs)

        type_check(dump_file_path, (str,), "dump_file_path")
        check_dir(dump_file_path)

        return method(self, *args, **kwargs)

    return new_method


def check_initialize(method):
    """Wrapper method to check the parameters of DbgServices Initialize method."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [net_name, is_sync_mode, max_mem_usage], _ = parse_user_args(method, *args, **kwargs)

        type_check(net_name, (str,), "net_name")
        type_check(is_sync_mode, (bool,), "is_sync_mode")
        check_uint32(max_mem_usage, "max_mem_usage")

        return method(self, *args, **kwargs)

    return new_method


def check_add_watchpoint(method):
    """Wrapper method to check the parameters of DbgServices AddWatchpoint."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [id_value, watch_condition, check_node_list, parameter_list], _ = parse_user_args(method, *args, **kwargs)

        check_uint32(id_value, "id")
        check_uint32(watch_condition, "watch_condition")
        type_check(check_node_list, (dict,), "check_node_list")
        for node_name, node_info in check_node_list.items():
            type_check(node_name, (str,), "node_name")
            type_check(node_info, (dict,), "node_info")
            for info_name, info_param in node_info.items():
                type_check(info_name, (str,), "node parameter name")
                if info_name in ["rank_id"]:
                    check_param_id(info_param, info_name="rank_id")
                elif info_name in ["root_graph_id"]:
                    check_param_id(info_param, info_name="root_graph_id")
                elif info_name in ["is_output"]:
                    type_check(info_param, (bool,), "is_output")
                else:
                    raise ValueError("Node parameter {} is not defined.".format(info_name))
        param_names = ["param_{0}".format(i) for i in range(len(parameter_list))]
        type_check_list(parameter_list, (cds.Parameter,), param_names)

        return method(self, *args, **kwargs)

    return new_method


def check_remove_watchpoint(method):
    """Wrapper method to check the parameters of DbgServices RemoveWatchpoint."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [id_value], _ = parse_user_args(method, *args, **kwargs)

        check_uint32(id_value, "id")

        return method(self, *args, **kwargs)

    return new_method


def check_check_watchpoints(method):
    """Wrapper method to check the parameters of DbgServices CheckWatchpoint."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [iteration, error_on_no_value], _ = parse_user_args(method, *args, **kwargs)

        check_iteration(iteration, "iteration")
        type_check(error_on_no_value, (bool,), "error_on_no_value")

        return method(self, *args, **kwargs)

    return new_method


def check_read_tensor_info(method):
    """Wrapper method to check the parameters of DbgServices ReadTensors."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [info_list], _ = parse_user_args(method, *args, **kwargs)

        info_names = ["info_{0}".format(i) for i in range(len(info_list))]
        type_check_list(info_list, (cds.TensorInfo,), info_names)

        return method(self, *args, **kwargs)

    return new_method


def check_initialize_done(method):
    """Wrapper method to check if initlize is done for DbgServices."""

    @wraps(method)
    def new_method(self, *args, **kwargs):

        if not self.initialized:
            raise RuntimeError("Inilize should be called before any other methods of DbgServices!")
        return method(self, *args, **kwargs)

    return new_method


def check_tensor_info_init(method):
    """Wrapper method to check the parameters of DbgServices TensorInfo init."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [node_name, slot, iteration, rank_id, root_graph_id,
         is_output], _ = parse_user_args(method, *args, **kwargs)

        type_check(node_name, (str,), "node_name")
        check_uint32(slot, "slot")
        check_iteration(iteration, "iteration")
        check_uint32(rank_id, "rank_id")
        check_uint32(root_graph_id, "root_graph_id")
        type_check(is_output, (bool,), "is_output")

        return method(self, *args, **kwargs)

    return new_method


def check_tensor_data_init(method):
    """Wrapper method to check the parameters of DbgServices TensorData init."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [data_ptr, data_size, dtype, shape], _ = parse_user_args(method, *args, **kwargs)

        type_check(data_ptr, (bytes,), "data_ptr")
        check_uint64(data_size, "data_size")
        type_check(dtype, (int,), "dtype")
        shape_names = ["shape_{0}".format(i) for i in range(len(shape))]
        type_check_list(shape, (int,), shape_names)

        if len(data_ptr) != data_size:
            raise ValueError("data_ptr length ({0}) is not equal to data_size ({1}).".format(len(data_ptr), data_size))

        return method(self, *args, **kwargs)

    return new_method


def check_tensor_base_data_init(method):
    """Wrapper method to check the parameters of DbgServices TensorBaseData init."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [data_size, dtype, shape], _ = parse_user_args(method, *args, **kwargs)

        check_uint64(data_size, "data_size")
        type_check(dtype, (int,), "dtype")
        shape_names = ["shape_{0}".format(i) for i in range(len(shape))]
        type_check_list(shape, (int,), shape_names)

        return method(self, *args, **kwargs)

    return new_method


def check_tensor_stat_data_init(method):
    """Wrapper method to check the parameters of DbgServices TensorBaseData init."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [data_size, dtype, shape, is_bool, max_value, min_value,
         avg_value, count, neg_zero_count, pos_zero_count,
         nan_count, neg_inf_count, pos_inf_count,
         zero_count], _ = parse_user_args(method, *args, **kwargs)

        check_uint64(data_size, "data_size")
        type_check(dtype, (int,), "dtype")
        shape_names = ["shape_{0}".format(i) for i in range(len(shape))]
        type_check_list(shape, (int,), shape_names)
        type_check(is_bool, (bool,), "is_bool")
        type_check(max_value, (float,), "max_value")
        type_check(min_value, (float,), "min_value")
        type_check(avg_value, (float,), "avg_value")
        type_check(count, (int,), "count")
        type_check(neg_zero_count, (int,), "neg_zero_count")
        type_check(pos_zero_count, (int,), "pos_zero_count")
        type_check(nan_count, (int,), "nan_count")
        type_check(neg_inf_count, (int,), "neg_inf_count")
        type_check(pos_inf_count, (int,), "pos_inf_count")
        type_check(zero_count, (int,), "zero_count")


        return method(self, *args, **kwargs)

    return new_method


def check_watchpoint_hit_init(method):
    """Wrapper method to check the parameters of DbgServices WatchpointHit init."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [name, slot, condition, watchpoint_id,
         parameters, error_code, rank_id, root_graph_id], _ = parse_user_args(method, *args, **kwargs)

        type_check(name, (str,), "name")
        check_uint32(slot, "slot")
        type_check(condition, (int,), "condition")
        check_uint32(watchpoint_id, "watchpoint_id")
        param_names = ["param_{0}".format(i) for i in range(len(parameters))]
        type_check_list(parameters, (cds.Parameter,), param_names)
        type_check(error_code, (int,), "error_code")
        check_uint32(rank_id, "rank_id")
        check_uint32(root_graph_id, "root_graph_id")

        return method(self, *args, **kwargs)

    return new_method


def check_parameter_init(method):
    """Wrapper method to check the parameters of DbgServices Parameter init."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [name, disabled, value, hit, actual_value], _ = parse_user_args(method, *args, **kwargs)

        type_check(name, (str,), "name")
        type_check(disabled, (bool,), "disabled")
        type_check(value, (float,), "value")
        type_check(hit, (bool,), "hit")
        type_check(actual_value, (float,), "actual_value")

        return method(self, *args, **kwargs)

    return new_method
