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
The module DbgServices provides offline debugger APIs.
"""

from mindspore._c_expression import security
from mindspore.offline_debug.mi_validators import check_init, check_initialize, check_add_watchpoint,\
     check_remove_watchpoint, check_check_watchpoints, check_read_tensor_info, check_initialize_done, \
         check_tensor_info_init, check_tensor_data_init, check_tensor_base_data_init, check_tensor_stat_data_init,\
              check_watchpoint_hit_init, check_parameter_init
from mindspore.offline_debug.mi_validator_helpers import replace_minus_one
from mindspore import log as logger
if not security.enable_security():
    import mindspore._mindspore_offline_debug as cds


def get_version():
    """
    Function to return offline Debug Services version.

    Returns:
        version (str): DbgServices version.

    Examples:
        >>> from mindspore.ccsrc.debug.debugger.offline_debug import dbg_services
        >>> version = dbg_services.get_version()
    """
    if security.enable_security():
        raise ValueError("Offline debugger is not supported in security mode. "
                         "Please recompile mindspore without `-s on`.")
    return cds.DbgServices().GetVersion()


class DbgServices:
    """
    Offline Debug Services class.

    Args:
        dump_file_path (str): Directory where the dump files are saved.

    Examples:
        >>> from mindspore.ccsrc.debug.debugger.offline_debug import dbg_services
        >>> d = dbg_services.DbgServices(dump_file_path="dump_file_path")
    """

    @check_init
    def __init__(self, dump_file_path):
        if security.enable_security():
            raise ValueError("Offline debugger is not supported in security mode. "
                             "Please recompile mindspore without `-s on`.")
        logger.info("in Python __init__, file path is %s", dump_file_path)
        self.dump_file_path = dump_file_path
        self.dbg_instance = cds.DbgServices()
        self.version = self.dbg_instance.GetVersion()
        self.initialized = False

    @staticmethod
    def transform_check_node_list(info_name, info_param, node_name, check_node_list):
        """
        Transforming check_node_list based on info_name and info_param.

        Args:
            info_name (str): Info name of check_node_list, either 'rank_id', 'root_graph_id' or 'is_output'
            info_param (list[int]): Info parameters of check_node_list, mapped to info_name.
            node_name (str): Node name as key of check_node_list.
            check_node_list (dict): Dictionary of node names (str or '*' to check all nodes) as key,
                                    mapping to rank_id (list of ints or '*' to check all devices),
                                    root_graph_id (list of ints or '*' to check all graphs) and is_output (bool).

        Returns:
            Transformed check_node_list.

        Examples:
        >>> from mindspore.offline_debug import dbg_services
        >>> d = dbg_services.DbgServices(dump_file_path="dump_file_path")
        >>> d_init = d.initialize(is_sync_mode=True)
        >>> d_wp = d_init.transform_check_node_list(info_name="rank_id",
        >>>                                         info_param=[0],
        >>>                                         node_name="conv2.bias",
        >>>                                         check_node_list={"conv2.bias" : {"rank_id": [0],
        >>>                                                                         root_graph_id: [0],
        >>>                                                                         "is_output": True}})
        """
        if info_name in ["rank_id", "root_graph_id"]:
            if info_param in ["*"]:
                check_node_list[node_name][info_name] = ["*"]
            else:
                check_node_list[node_name][info_name] = list(map(str, info_param))
        return check_node_list

    @check_initialize
    def initialize(self, net_name, is_sync_mode=True, max_mem_usage=0):
        """
        Initialize Debug Service.

        Args:
            net_name (str): Network name.
            is_sync_mode (bool): Whether to process synchronous or asynchronous dump files mode
                                 (default: True (synchronous)).
            max_mem_usage (int): Maximum memory size of the debugger internal tensor cache in Megabytes(MB),
                                 (default: 0 (disable memory restriction feature)).

        Returns:
            Initialized Debug Service instance.

        Examples:
        >>> from mindspore.ccsrc.debug.debugger.offline_debug import dbg_services
        >>> d = dbg_services.DbgServices(dump_file_path="dump_file_path")
        >>> d_init = d.initialize(net_name="network name", is_sync_mode=True, max_mem_usage=4096)
        """
        logger.info("in Python Initialize dump_file_path %s", self.dump_file_path)
        self.initialized = True
        return self.dbg_instance.Initialize(net_name, self.dump_file_path, is_sync_mode, max_mem_usage)

    @check_initialize_done
    @check_add_watchpoint
    def add_watchpoint(self, watchpoint_id, watch_condition, check_node_list, parameter_list):
        """
        Adding watchpoint to Debug Service instance.

        Args:
            watchpoint_id (int): Watchpoint id
            watch_condition (int): A representation of the condition to be checked.
            check_node_list (dict): Dictionary of node names (str or '*' to check all nodes) as key,
                                    mapping to rank_id (list of ints or '*' to check all devices),
                                    root_graph_id (list of ints or '*' to check all graphs) and is_output (bool).
            parameter_list (list): List of parameters in watchpoint. Parameters should be instances of Parameter class.
                                   Each parameter describes the value to be checked in watchpoint.

        Returns:
            Debug Service instance with added watchpoint.

        Examples:
        >>> from mindspore.ccsrc.debug.debugger.offline_debug import dbg_services
        >>> d = dbg_services.DbgServices(dump_file_path="dump_file_path")
        >>> d_init = d.initialize(is_sync_mode=True)
        >>> d_wp = d_init.add_watchpoint(watchpoint_id=1,
        ...                              watch_condition=6,
        ...                              check_node_list={"conv2.bias" : {"rank_id": [0],
        ...                                                               root_graph_id: [0], "is_output": True}},
        ...                              parameter_list=[dbg_services.Parameter(name="param",
        ...                                                                     disabled=False,
        ...                                                                     value=0.0,
        ...                                                                     hit=False,
        ...                                                                     actual_value=0.0)])
        """
        logger.info("in Python AddWatchpoint")
        for node_name, node_info in check_node_list.items():
            for info_name, info_param in node_info.items():
                check_node_list = self.transform_check_node_list(info_name, info_param, node_name, check_node_list)
        parameter_list_inst = []
        for elem in parameter_list:
            parameter_list_inst.append(elem.instance)
        return self.dbg_instance.AddWatchpoint(watchpoint_id, watch_condition, check_node_list, parameter_list_inst)

    @check_initialize_done
    @check_remove_watchpoint
    def remove_watchpoint(self, watchpoint_id):
        """
        Removing watchpoint from Debug Service instance.

        Args:
            watchpoint_id (int): Watchpoint id.

        Returns:
            Debug Service instance with removed watchpoint.

        Examples:
        >>> from mindspore.ccsrc.debug.debugger.offline_debug import dbg_services
        >>> d = dbg_services.DbgServices(dump_file_path="dump_file_path")
        >>> d_init = d.initialize(is_sync_mode=True)
        >>> d_wp = d_init.add_watchpoint(watchpoint_id=1,
        ...                              watch_condition=6,
        ...                              check_node_list={"conv2.bias" : {"rank_id": [0],
        ...                                                               root_graph_id: [0], "is_output": True}},
        ...                              parameter_list=[dbg_services.Parameter(name="param",
        ...                                                                     disabled=False,
        ...                                                                     value=0.0,
        ...                                                                     hit=False,
        ...                                                                     actual_value=0.0)])
        >>> d_wp = d_wp.remove_watchpoint(watchpoint_id=1)
        """
        logger.info("in Python Remove Watchpoint id %d", watchpoint_id)
        return self.dbg_instance.RemoveWatchpoint(watchpoint_id)

    @check_initialize_done
    @check_check_watchpoints
    def check_watchpoints(self, iteration, error_on_no_value=False):
        """
        Checking watchpoint at given iteration.

        Args:
            iteration (int): Watchpoint check iteration.
            error_on_no_value (bool): Whether report error when the tensor has
                no value. Default: False.

        Returns:
            Watchpoint hit list.

        Examples:
        >>> from mindspore.ccsrc.debug.debugger.offline_debug import dbg_services
        >>> d = dbg_services.DbgServices(dump_file_path="dump_file_path")
        >>> d_init = d.initialize(is_sync_mode=True)
        >>> d_wp = d_init.add_watchpoint(id=1,
        ...                              watch_condition=6,
        ...                              check_node_list={"conv2.bias" : {"rank_id": [5],
        ...                                                               root_graph_id: [0], "is_output": True}},
        ...                              parameter_list=[dbg_services.Parameter(name="param",
        ...                                                                     disabled=False,
        ...                                                                     value=0.0,
        ...                                                                     hit=False,
        ...                                                                     actual_value=0.0)])
        >>> watchpoints = d_wp.check_watchpoints(iteration=8)
        """
        logger.info("in Python CheckWatchpoints iteration %d", iteration)
        iteration = replace_minus_one(iteration)
        watchpoint_list = self.dbg_instance.CheckWatchpoints(iteration, error_on_no_value)
        watchpoint_hit_list = []
        for watchpoint in watchpoint_list:
            name = watchpoint.get_name()
            slot = watchpoint.get_slot()
            condition = watchpoint.get_condition()
            watchpoint_id = watchpoint.get_watchpoint_id()
            parameters = watchpoint.get_parameters()
            error_code = watchpoint.get_error_code()
            rank_id = watchpoint.get_rank_id()
            root_graph_id = watchpoint.get_root_graph_id()
            param_list = []
            for param in parameters:
                p_name = param.get_name()
                disabled = param.get_disabled()
                value = param.get_value()
                hit = param.get_hit()
                actual_value = param.get_actual_value()
                param_list.append(Parameter(p_name, disabled, value, hit, actual_value))
            watchpoint_hit_list.append(WatchpointHit(name, slot, condition, watchpoint_id,
                                                     param_list, error_code, rank_id, root_graph_id))
        return watchpoint_hit_list

    @check_initialize_done
    def check_watchpoint_progress(self):
        """
        Returning the progress percentage of checking watchpoint.

        Returns:
            float, progress percentage.
        """
        progress_percentage = self.dbg_instance.CheckWatchpointProgress()
        return progress_percentage

    @check_initialize_done
    @check_read_tensor_info
    def read_tensors(self, info):
        """
        Returning tensor data object describing the tensor requested tensor.

        Args:
            info (list): List of TensorInfo objects.

        Returns:
            TensorData list (list).

        Examples:
        >>> from mindspore.ccsrc.debug.debugger.offline_debug import dbg_services
        >>> d = dbg_services.DbgServices(dump_file_path="dump_file_path")
        >>> d_init = d.initialize(is_sync_mode=True)
        >>> tensor_data_list = d_init.read_tensors([dbg_services.TensorInfo(node_name="conv2.bias",
        ...                                                                 slot=0,
        ...                                                                 iteration=8,
        ...                                                                 rank_id=5,
        ...                                                                 root_graph_id=0,
        ...                                                                 is_output=True)])
        """
        logger.info("in Python ReadTensors info:")
        logger.info(info)
        info_list_inst = []
        for elem in info:
            logger.info("in Python ReadTensors info:")
            logger.info(info)
            info_list_inst.append(elem.instance)
        tensor_data_list = self.dbg_instance.ReadTensors(info_list_inst)
        tensor_data_list_ret = []
        for elem in tensor_data_list:
            if elem.get_data_size() == 0:
                tensor_data = TensorData(b'', elem.get_data_size(), elem.get_dtype(), elem.get_shape())
            else:
                tensor_data = TensorData(elem.get_data_ptr(), elem.get_data_size(), elem.get_dtype(), elem.get_shape())
            tensor_data_list_ret.append(tensor_data)
        return tensor_data_list_ret

    @check_initialize_done
    @check_read_tensor_info
    def read_tensor_base(self, info):
        """
        Returning tensor base data object describing the requested tensor.

        Args:
            info (list): List of TensorInfo objects.

        Returns:
            list, TensorBaseData list.

        Examples:
        >>> from mindspore.ccsrc.debug.debugger.offline_debug import dbg_services
        >>> d = dbg_services.DbgServices(dump_file_path="dump_file_path")
        >>> d_init = d.initialize(is_sync_mode=True)
        >>> tensor_base_data_list = d_init.read_tensor_base([dbg_services.TensorInfo(node_name="conv2.bias",
        ...                                                                          slot=0,
        ...                                                                          iteration=8,
        ...                                                                          rank_id=5,
        ...                                                                          root_graph_id=0,
        ...                                                                          is_output=True)])
        """
        logger.info("in Python ReadTensorsBase info:")
        logger.info(info)
        info_list_inst = []
        for elem in info:
            logger.info("in Python ReadTensorsBase info:")
            logger.info(info)
            info_list_inst.append(elem.instance)
        tensor_base_data_list = self.dbg_instance.ReadTensorsBase(info_list_inst)
        tensor_base_data_list_ret = []
        for elem in tensor_base_data_list:
            tensor_base_data = TensorBaseData(elem.data_size(), elem.dtype(), elem.shape())
            tensor_base_data_list_ret.append(tensor_base_data)
        return tensor_base_data_list_ret

    @check_initialize_done
    @check_read_tensor_info
    def read_tensor_stats(self, info):
        """
        Returning tensor statistics object describing the requested tensor.

        Args:
            info (list): List of TensorInfo objects.

        Returns:
            list, TensorStatData list.

        Examples:
        >>> from mindspore.ccsrc.debug.debugger.offline_debug import dbg_services
        >>> d = dbg_services.DbgServices(dump_file_path="dump_file_path")
        >>> d_init = d.initialize(is_sync_mode=True)
        >>> tensor_stat_data_list = d_init.read_tensor_stats([dbg_services.TensorInfo(node_name="conv2.bias",
        ...                                                                           slot=0,
        ...                                                                           iteration=8,
        ...                                                                           rank_id=5,
        ...                                                                           root_graph_id=0,
        ...                                                                           is_output=True)])
        """
        logger.info("in Python ReadTensorsStat info:")
        logger.info(info)
        info_list_inst = []
        for elem in info:
            logger.info("in Python ReadTensorsStat info:")
            logger.info(info)
            info_list_inst.append(elem.instance)
        tensor_stat_data_list = self.dbg_instance.ReadTensorsStat(info_list_inst)
        tensor_stat_data_list_ret = []
        for elem in tensor_stat_data_list:
            tensor_stat_data = TensorStatData(elem.data_size(), elem.dtype(),
                                              elem.shape(), elem.is_bool(),
                                              elem.max_value(), elem.min_value(),
                                              elem.avg_value(), elem.count(), elem.neg_zero_count(),
                                              elem.pos_zero_count(), elem.nan_count(), elem.neg_inf_count(),
                                              elem.pos_inf_count(), elem.zero_count())
            tensor_stat_data_list_ret.append(tensor_stat_data)
        return tensor_stat_data_list_ret


class TensorInfo:
    """
    Tensor Information class.

    Args:
        node_name (str): Fully qualified name of the desired node.
        slot (int): The particular output for the requested node.
        iteration (int): The desired itraretion to gather tensor information.
        rank_id (int): The desired rank id to gather tensor information.
        is_output (bool): Whether node is an output or input.

    Examples:
        >>> from mindspore.ccsrc.debug.debugger.offline_debug import dbg_services
        >>> tensor_info = dbg_services.TensorInfo(node_name="conv2.bias",
        ...                                       slot=0,
        ...                                       iteration=8,
        ...                                       rank_id=5,
        ...                                       root_graph_id=0,
        ...                                       is_output=True)
    """
    @check_tensor_info_init
    def __init__(self, node_name, slot, iteration, rank_id, root_graph_id, is_output=True):
        if security.enable_security():
            raise ValueError("Offline debugger is not supported in security mode. "
                             "Please recompile mindspore without `-s on`.")
        iteration = replace_minus_one(iteration)
        self.instance = cds.tensor_info(node_name, slot, iteration, rank_id, root_graph_id, is_output)

    @property
    def node_name(self):
        """
        Function to receive TensorInfo node_name.

        Returns:
            node_name of TensorInfo instance (str).

        Examples:
            >>> from mindspore.ccsrc.debug.debugger.offline_debug import dbg_services
            >>> tensor_info = dbg_services.TensorInfo(node_name="conv2.bias",
            ...                                       slot=0,
            ...                                       iteration=8,
            ...                                       rank_id=5,
            ...                                       root_graph_id=0,
            ...                                       is_output=True)
            >>> name = tensor_info.node_name
        """
        return self.instance.get_node_name()

    @property
    def slot(self):
        """
        Function to receive TensorInfo slot.

        Returns:
            slot of TensorInfo instance (int).

        Examples:
            >>> from mindspore.ccsrc.debug.debugger.offline_debug import dbg_services
            >>> tensor_info = dbg_services.TensorInfo(node_name="conv2.bias",
            ...                                       slot=0,
            ...                                       iteration=8,
            ...                                       rank_id=5,
            ...                                       root_graph_id=0,
            ...                                       is_output=True)
            >>> slot = tensor_info.slot
        """
        return self.instance.get_slot()

    @property
    def iteration(self):
        """
        Function to receive TensorInfo iteration.

        Returns:
            iteration of TensorInfo instance (int).

        Examples:
            >>> from mindspore.ccsrc.debug.debugger.offline_debug import dbg_services
            >>> tensor_info = dbg_services.TensorInfo(node_name="conv2.bias",
            ...                                       slot=0,
            ...                                       iteration=8,
            ...                                       rank_id=5,
            ...                                       root_graph_id=0,
            ...                                       is_output=True)
            >>> iteration = tensor_info.iteration
        """
        return self.instance.get_iteration()

    @property
    def rank_id(self):
        """
        Function to receive TensorInfo rank_id.

        Returns:
            rank_id of TensorInfo instance (int).

        Examples:
            >>> from mindspore.ccsrc.debug.debugger.offline_debug import dbg_services
            >>> tensor_info = dbg_services.TensorInfo(node_name="conv2.bias",
            ...                                       slot=0,
            ...                                       iteration=8,
            ...                                       rank_id=5,
            ...                                       root_graph_id=0,
            ...                                       is_output=True)
            >>> rank_id = tensor_info.rank_id
        """
        return self.instance.get_rank_id()

    @property
    def root_graph_id(self):
        """
        Function to receive TensorInfo root_graph_id.

        Returns:
            root_graph_id of TensorInfo instance (int).

        Examples:
            >>> from mindspore.ccsrc.debug.debugger.offline_debug import dbg_services
            >>> tensor_info = dbg_services.TensorInfo(node_name="conv2.bias",
            ...                                       slot=0,
            ...                                       iteration=8,
            ...                                       rank_id=5,
            ...                                       root_graph_id=0,
            ...                                       is_output=True)
            >>> rank_id = tensor_info.root_graph_id
        """
        return self.instance.get_root_graph_id()

    @property
    def is_output(self):
        """
        Function to receive TensorInfo is_output.

        Returns:
            is_output of TensorInfo instance (bool).

        Examples:
            >>> from mindspore.ccsrc.debug.debugger.offline_debug import dbg_services
            >>> tensor_info = dbg_services.TensorInfo(node_name="conv2.bias",
            ...                                       slot=0,
            ...                                       iteration=8,
            ...                                       rank_id=5,
            ...                                       root_graph_id=0,
            ...                                       is_output=True)
            >>> is_output = tensor_info.is_output
        """
        return self.instance.get_is_output()


class TensorData:
    """
    TensorData class.

    Args:
        data_ptr (byte): Data pointer.
        data_size (int): Size of data in bytes.
        dtype (int): An encoding representing the type of TensorData.
        shape (list): Shape of tensor.

    Examples:
        >>> from mindspore.ccsrc.debug.debugger.offline_debug import dbg_services
        >>> tensor_data = dbg_services.TensorData(data_ptr=b'\xba\xd0\xba\xd0',
        ...                                       data_size=4,
        ...                                       dtype=0,
        ...                                       shape=[2, 2])
    """
    @check_tensor_data_init
    def __init__(self, data_ptr, data_size, dtype, shape):
        if security.enable_security():
            raise ValueError("Offline debugger is not supported in security mode."
                             "Please recompile mindspore without `-s on`.")
        self.instance = cds.tensor_data(data_ptr, data_size, dtype, shape)

    @property
    def data_ptr(self):
        """
        Function to receive TensorData data_ptr.

        Returns:
            data_ptr of TensorData instance (byte).

        Examples:
            >>> from mindspore.ccsrc.debug.debugger.offline_debug import dbg_services
            >>> tensor_data = dbg_services.TensorData(data_ptr=b'\xba\xd0\xba\xd0',
            ...                                       data_size=4,
            ...                                       dtype=0,
            ...                                       shape=[2, 2])
            >>> data_ptr = tensor_data.data_ptr
        """
        return self.instance.get_data_ptr()

    @property
    def data_size(self):
        """
        Function to receive TensorData data_size.

        Returns:
            data_size of TensorData instance (int).

        Examples:
            >>> from mindspore.ccsrc.debug.debugger.offline_debug import dbg_services
            >>> tensor_data = dbg_services.TensorData(data_ptr=b'\xba\xd0\xba\xd0',
            ...                                       data_size=4,
            ...                                       dtype=0,
            ...                                       shape=[2, 2])
            >>> data_size = tensor_data.data_size
        """
        return self.instance.get_data_size()

    @property
    def dtype(self):
        """
        Function to receive TensorData dtype.

        Returns:
            dtype of TensorData instance (int).

        Examples:
            >>> from mindspore.ccsrc.debug.debugger.offline_debug import dbg_services
            >>> tensor_data = dbg_services.TensorData(data_ptr=b'\xba\xd0\xba\xd0',
            ...                                       data_size=4,
            ...                                       dtype=0,
            ...                                       shape=[2, 2])
            >>> dtype = tensor_data.dtype
        """
        return self.instance.get_dtype()

    @property
    def shape(self):
        """
        Function to receive TensorData shape.

        Returns:
            shape of TensorData instance (list).

        Examples:
            >>> from mindspore.ccsrc.debug.debugger.offline_debug import dbg_services
            >>> tensor_data = dbg_services.TensorData(data_ptr=b'\xba\xd0\xba\xd0',
            ...                                       data_size=4,
            ...                                       dtype=0,
            ...                                       shape=[2, 2])
            >>> shape = tensor_data.shape
        """
        return self.instance.get_shape()


class TensorBaseData:

    """
    TensorBaseData class.

    Args:
        data_size (int): Size of data in bytes.
        dtype (int): An encoding representing the type of TensorData.
        shape (list): Shape of tensor.

    Examples:
        >>> from mindspore.ccsrc.debug.debugger.offline_debug import dbg_services
        >>> tensor_base_data = dbg_services.TensorBaseData(data_size=4,
        ...                                                dtype=0,
        ...                                                shape=[2, 2])
    """
    @check_tensor_base_data_init
    def __init__(self, data_size, dtype, shape):
        if security.enable_security():
            raise ValueError("Offline debugger is not supported in security mode. "
                             "Please recompile mindspore without `-s on`.")
        self.instance = cds.TensorBaseData(data_size, dtype, shape)

    def __str__(self):
        tensor_base_info = (
            f'size in bytes = {self.data_size}\n'
            f'debugger dtype = {self.dtype}\n'
            f'shape = {self.shape}'
        )
        return tensor_base_info

    @property
    def data_size(self):
        """
        Function to receive TensorBaseData data_size.

        Returns:
            int, data_size of TensorBaseData instance.

        Examples:
            >>> from mindspore.ccsrc.debug.debugger.offline_debug import dbg_services
            >>> tensor_base_data = dbg_services.TensorBaseData(data_size=4,
            ...                                       dtype=0,
            ...                                       shape=[2, 2])
            >>> data_size = tensor_base_data.data_size
        """
        return self.instance.data_size()

    @property
    def dtype(self):
        """
        Function to receive TensorBaseData dtype.

        Returns:
            int, dtype of TensorBaseData instance.

        Examples:
            >>> from mindspore.ccsrc.debug.debugger.offline_debug import dbg_services
            >>> tensor_base_data = dbg_services.TensorBaseData(data_size=4,
            ...                                                dtype=0,
            ...                                                shape=[2, 2])
            >>> dtype = tensor_base_data.dtype
        """

        return self.instance.dtype()

    @property
    def shape(self):
        """
        Function to receive TensorBaseData shape.

        Returns:
            list, shape of TensorBaseData instance.

        Examples:
            >>> from mindspore.ccsrc.debug.debugger.offline_debug import dbg_services
            >>> tensor_base_data = dbg_services.TensorBaseData(data_size=4,
            ...                                       dtype=0,
            ...                                       shape=[2, 2])
            >>> shape = tensor_base_data.shape
        """
        return self.instance.shape()


class TensorStatData:

    """
    TensorStatData class.

    Args:
        data_size (int): Size of data in bytes.
        dtype (int): An encoding representing the type of TensorData.
        shape (list): Shape of tensor.
        is_bool (bool): Whether the data type is bool.
        max_value (float): Maximum value in tensor's elements.
        min_value (float): Minimum value in tensor's elements.
        avg_value (float): Average value of all tensor's elements.
        count (int): Number of elements in tensor.
        neg_zero_count (int): Number of negative elements in tensor.
        pos_zero_count (int): Number of positive elements in tensor.
        nan_cout (int): Number of nan elements in tensor.
        neg_inf_count (int): Number of negative infinity elements in tensor.
        pos_inf_count (int): Number of positive infinity elements in tensor.
        zero_count (int): Total number of zero elements in tensor.

    Examples:
        >>> from mindspore.ccsrc.debug.debugger.offline_debug import dbg_services
        >>> tensor_stat_data = dbg_services.TensorStatData(data_size=4, dtype=0, shape=[2, 2], is_bool = false,
        ...                                                max_value = 10.0, min_value = 0.0, avg_value = 5.0,
        ...                                                count = 4, neg_zero_count = 0, pos_zero_count = 4,
        ...                                                nan_count = 0, neg_inf_count = 0, pos_inf_count = 0,
        ...                                                zero_count = 1)
    """
    @check_tensor_stat_data_init
    def __init__(self, data_size, dtype, shape, is_bool, max_value, min_value, avg_value, count,
                 neg_zero_count, pos_zero_count, nan_count, neg_inf_count, pos_inf_count, zero_count):
        if security.enable_security():
            raise ValueError("Offline debugger is not supported in security mode. "
                             "Please recompile mindspore without `-s on`.")
        self.instance = cds.TensorStatData(data_size, dtype, shape, is_bool, max_value,
                                           min_value, avg_value, count, neg_zero_count,
                                           pos_zero_count, nan_count, neg_inf_count,
                                           pos_inf_count, zero_count)

    def __str__(self):
        tensor_stats_info = (
            f'size in bytes = {self.data_size}\n'
            f'debugger dtype = {self.dtype}\n'
            f'shape = {self.shape}\n'
            f'is_bool = {self.is_bool}\n'
            f'max_value = {self.max_value}\n'
            f'min_value = {self.min_value}\n'
            f'avg_value = {self.avg_value}\n'
            f'count = {self.count}\n'
            f'neg_zero_count = {self.neg_zero_count}\n'
            f'pos_zero_count = {self.pos_zero_count}\n'
            f'nan_count = {self.nan_count}\n'
            f'neg_inf_count = {self.neg_inf_count}\n'
            f'pos_inf_count = {self.pos_inf_count}\n'
            f'zero_count = {self.zero_count}\n'
            )
        return tensor_stats_info

    @property
    def data_size(self):
        """
        Function to receive TensorStatData data_size.

        Returns:
            int, data_size of TensorStatData instance.

        Examples:
            >>> from mindspore.ccsrc.debug.debugger.offline_debug import dbg_services
            >>> tensor_stat_data = dbg_services.TensorStatData(data_size=4, dtype=0, shape=[2, 2], is_bool = false,
            ...                                                max_value = 10.0, min_value = 0.0, avg_value = 5.0,
            ...                                                count = 4, neg_zero_count = 0, pos_zero_count = 4,
            ...                                                nan_count = 0, neg_inf_count = 0, pos_inf_count = 0,
            ...                                                zero_count = 1)
            >>> data_size = tensor_stat_data.data_size
        """

        return self.instance.data_size()

    @property
    def dtype(self):
        """
        Function to receive TensorStatData dtype.

        Returns:
            int, dtype of TensorStatData instance.

        Examples:
            >>> from mindspore.ccsrc.debug.debugger.offline_debug import dbg_services
            >>> tensor_stat_data = dbg_services.TensorStatData(data_size=4, dtype=0, shape=[2, 2], is_bool = false,
            ...                                                max_value = 10.0, min_value = 0.0, avg_value = 5.0,
            ...                                                count = 4, neg_zero_count = 0, pos_zero_count = 4,
            ...                                                nan_count = 0, neg_inf_count = 0, pos_inf_count = 0,
            ...                                                zero_count = 1)
            >>> dtype = tensor_stat_data.dtype
        """
        return self.instance.dtype()

    @property
    def shape(self):
        """
        Function to receive TensorStatData shape.

        Returns:
            list, shape of TensorStatData instance.

        Examples:
            >>> from mindspore.ccsrc.debug.debugger.offline_debug import dbg_services
            >>> tensor_stat_data = dbg_services.TensorStatData(data_size=4, dtype=0, shape=[2, 2], is_bool = false,
            ...                                                max_value = 10.0, min_value = 0.0, avg_value = 5.0,
            ...                                                count = 4, neg_zero_count = 0, pos_zero_count = 4,
            ...                                                nan_count = 0, neg_inf_count = 0, pos_inf_count = 0,
            ...                                                zero_count = 1)
            >>> shape = tensor_stat_data.shape
        """
        return self.instance.shape()

    @property
    def is_bool(self):
        """
        Function to receive TensorStatData is_bool.

        Returns:
            bool, Whether the tensor elements are bool.

        Examples:
            >>> from mindspore.ccsrc.debug.debugger.offline_debug import dbg_services
            >>> tensor_stat_data = dbg_services.TensorStatData(data_size=4, dtype=0, shape=[2, 2], is_bool = false,
            ...                                                max_value = 10.0, min_value = 0.0, avg_value = 5.0,
            ...                                                count = 4, neg_zero_count = 0, pos_zero_count = 4,
            ...                                                nan_count = 0, neg_inf_count = 0, pos_inf_count = 0,
            ...                                                zero_count = 1)
            >>> is_bool = tensor_stat_data.is_bool
        """
        return self.instance.is_bool()

    @property
    def max_value(self):
        """
        Function to receive TensorStatData max_value.

        Returns:
            float, max_value of TensorStatData instance.

        Examples:
            >>> from mindspore.ccsrc.debug.debugger.offline_debug import dbg_services
            >>> tensor_stat_data = dbg_services.TensorStatData(data_size=4, dtype=0, shape=[2, 2], is_bool = false,
            ...                                                max_value = 10.0, min_value = 0.0, avg_value = 5.0,
            ...                                                count = 4, neg_zero_count = 0, pos_zero_count = 4,
            ...                                                nan_count = 0, neg_inf_count = 0, pos_inf_count = 0,
            ...                                                zero_count = 1)
            >>> max_value = tensor_stat_data.max_value
        """
        return self.instance.max_value()

    @property
    def min_value(self):
        """
        Function to receive TensorStatData min_value.

        Returns:
            float, min_value of TensorStatData instance.

        Examples:
            >>> from mindspore.ccsrc.debug.debugger.offline_debug import dbg_services
            >>> tensor_stat_data = dbg_services.TensorStatData(data_size=4, dtype=0, shape=[2, 2], is_bool = false,
            ...                                                max_value = 10.0, min_value = 0.0, avg_value = 5.0,
            ...                                                count = 4, neg_zero_count = 0, pos_zero_count = 4,
            ...                                                nan_count = 0, neg_inf_count = 0, pos_inf_count = 0,
            ...                                                zero_count = 1)
            >>> min_value = tensor_stat_data.min_value
        """
        return self.instance.min_value()

    @property
    def avg_value(self):
        """
        Function to receive TensorStatData avg_value.

        Returns:
            float, avg_value of TensorStatData instance.

        Examples:
            >>> from mindspore.ccsrc.debug.debugger.offline_debug import dbg_services
            >>> tensor_stat_data = dbg_services.TensorStatData(data_size=4, dtype=0, shape=[2, 2], is_bool = false,
            ...                                                max_value = 10.0, min_value = 0.0, avg_value = 5.0,
            ...                                                count = 4, neg_zero_count = 0, pos_zero_count = 4,
            ...                                                nan_count = 0, neg_inf_count = 0, pos_inf_count = 0,
            ...                                                zero_count = 1)
            >>> avg_value = tensor_stat_data.avg_value
        """
        return self.instance.avg_value()

    @property
    def count(self):
        """
        Function to receive TensorStatData count.

        Returns:
            int, count of TensorStatData instance.

        Examples:
            >>> from mindspore.ccsrc.debug.debugger.offline_debug import dbg_services
            >>> tensor_stat_data = dbg_services.TensorStatData(data_size=4, dtype=0, shape=[2, 2], is_bool = false,
            ...                                                max_value = 10.0, min_value = 0.0, avg_value = 5.0,
            ...                                                count = 4, neg_zero_count = 0, pos_zero_count = 4,
            ...                                                nan_count = 0, neg_inf_count = 0, pos_inf_count = 0,
            ...                                                zero_count = 1)
            >>> count = tensor_stat_data.count
        """
        return self.instance.count()

    @property
    def neg_zero_count(self):
        """
        Function to receive TensorStatData neg_zero_count.

        Returns:
            int, neg_zero_count of TensorStatData instance.

        Examples:
            >>> from mindspore.ccsrc.debug.debugger.offline_debug import dbg_services
            >>> tensor_stat_data = dbg_services.TensorStatData(data_size=4, dtype=0, shape=[2, 2], is_bool = false,
            ...                                                max_value = 10.0, min_value = 0.0, avg_value = 5.0,
            ...                                                count = 4, neg_zero_count = 0, pos_zero_count = 4,
            ...                                                nan_count = 0, neg_inf_count = 0, pos_inf_count = 0,
            ...                                                zero_count = 1)
            >>> neg_zero_count = tensor_stat_data.neg_zero_count
        """
        return self.instance.neg_zero_count()

    @property
    def pos_zero_count(self):
        """
        Function to receive TensorStatData pos_zero_count.

        Returns:
            int, pos_zero_count of TensorStatData instance.

        Examples:
            >>> from mindspore.ccsrc.debug.debugger.offline_debug import dbg_services
            >>> tensor_stat_data = dbg_services.TensorStatData(data_size=4, dtype=0, shape=[2, 2], is_bool = false,
            ...                                                max_value = 10.0, min_value = 0.0, avg_value = 5.0,
            ...                                                count = 4, neg_zero_count = 0, pos_zero_count = 4,
            ...                                                nan_count = 0, neg_inf_count = 0, pos_inf_count = 0,
            ...                                                zero_count = 1)
            >>> pos_zero_count = tensor_stat_data.pos_zero_count
        """
        return self.instance.pos_zero_count()

    @property
    def zero_count(self):
        """
        Function to receive TensorStatData zero_count.

        Returns:
            int, zero_count of TensorStatData instance.

        Examples:
            >>> from mindspore.ccsrc.debug.debugger.offline_debug import dbg_services
            >>> tensor_stat_data = dbg_services.TensorStatData(data_size=4, dtype=0, shape=[2, 2], is_bool = false,
            ...                                                max_value = 10.0, min_value = 0.0, avg_value = 5.0,
            ...                                                count = 4, neg_zero_count = 0, pos_zero_count = 4,
            ...                                                nan_count = 0, neg_inf_count = 0, pos_inf_count = 0,
            ...                                                zero_count = 1)
            >>> zero_count = tensor_stat_data.zero_count
        """
        return self.instance.zero_count()

    @property
    def nan_count(self):
        """
        Function to receive TensorStatData nan_count.

        Returns:
            int, nan_count of TensorStatData instance.

        Examples:
            >>> from mindspore.ccsrc.debug.debugger.offline_debug import dbg_services
            >>> tensor_stat_data = dbg_services.TensorStatData(data_size=4, dtype=0, shape=[2, 2], is_bool = false,
            ...                                                max_value = 10.0, min_value = 0.0, avg_value = 5.0,
            ...                                                count = 4, neg_zero_count = 0, pos_zero_count = 4,
            ...                                                nan_count = 0, neg_inf_count = 0, pos_inf_count = 0,
            ...                                                zero_count = 1)
            >>> nan_count = tensor_stat_data.nan_count
        """
        return self.instance.nan_count()

    @property
    def neg_inf_count(self):
        """
        Function to receive TensorStatData shape.

        Returns:
            int, neg_inf_count of TensorStatData instance.

        Examples:
            >>> from mindspore.ccsrc.debug.debugger.offline_debug import dbg_services
            >>> tensor_stat_data = dbg_services.TensorStatData(data_size=4, dtype=0, shape=[2, 2], is_bool = false,
            ...                                                max_value = 10.0, min_value = 0.0, avg_value = 5.0,
            ...                                                count = 4, neg_zero_count = 0, pos_zero_count = 4,
            ...                                                nan_count = 0, neg_inf_count = 0, pos_inf_count = 0,
            ...                                                zero_count = 1)
            >>> neg_inf_count = tensor_stat_data.neg_inf_count
        """
        return self.instance.neg_inf_count()

    @property
    def pos_inf_count(self):
        """
        Function to receive TensorStatData pos_inf_count.

        Returns:
            pos_inf_count of TensorStatData instance (int).

        Examples:
            >>> from mindspore.ccsrc.debug.debugger.offline_debug import dbg_services
            >>> tensor_stat_data = dbg_services.TensorStatData(data_size=4, dtype=0, shape=[2, 2], is_bool = false,
            ...                                                max_value = 10.0, min_value = 0.0, avg_value = 5.0,
            ...                                                count = 4, neg_zero_count = 0, pos_zero_count = 4,
            ...                                                nan_count = 0, neg_inf_count = 0, pos_inf_count = 0,
            ...                                                zero_count = 1)
            >>> pos_inf_count = tensor_stat_data.pos_inf_count
        """
        return self.instance.pos_inf_count()


class WatchpointHit:
    """
    WatchpointHit class.

    Args:
        name (str): Name of WatchpointHit instance.
        slot (int): The numerical label of an output.
        condition (int): A representation of the condition to be checked.
        watchpoint_id (int): Watchpoint id.
        parameters (list): A list of all parameters for WatchpointHit instance.
                           Parameters have to be instances of Parameter class.
        error_code (int): An explanation of certain scenarios where watchpoint could not be checked.
        rank_id (int): Rank id where the watchpoint is hit.
        root_graph_id (int): Root graph id where the watchpoint is hit.

    Examples:
        >>> from mindspore.ccsrc.debug.debugger.offline_debug import dbg_services
        >>> watchpoint_hit = dbg_services.WatchpointHit(name="hit1",
        ...                                             slot=1,
        ...                                             condition=2,
        ...                                             watchpoint_id=3,
        ...                                             parameters=[param1, param2],
        ...                                             error_code=0,
        ...                                             rank_id=1,
        ...                                             root_graph_id=1)
    """

    @check_watchpoint_hit_init
    def __init__(self, name, slot, condition, watchpoint_id, parameters, error_code, rank_id, root_graph_id):
        if security.enable_security():
            raise ValueError("Offline debugger is not supported in security mode. "
                             "Please recompile mindspore without `-s on`.")
        parameter_list_inst = []
        for elem in parameters:
            parameter_list_inst.append(elem.instance)
        self.instance = cds.watchpoint_hit(name, slot, condition, watchpoint_id,
                                           parameter_list_inst, error_code, rank_id, root_graph_id)

    @property
    def name(self):
        """
        Function to receive WatchpointHit name.

        Returns:
            name of WatchpointHit instance (str).

        Examples:
            >>> from mindspore.ccsrc.debug.debugger.offline_debug import dbg_services
            >>> watchpoint_hit = dbg_services.WatchpointHit(name="hit1",
            ...                                             slot=1,
            ...                                             condition=2,
            ...                                             watchpoint_id=3,
            ...                                             parameters=[param1, param2],
            ...                                             error_code=0,
            ...                                             rank_id=1,
            ...                                             root_graph_id=1)
            >>> name = watchpoint_hit.name
        """
        return self.instance.get_name()

    @property
    def slot(self):
        """
        Function to receive WatchpointHit slot.

        Returns:
            slot of WatchpointHit instance (int).

        Examples:
            >>> from mindspore.ccsrc.debug.debugger.offline_debug import dbg_services
            >>> watchpoint_hit = dbg_services.WatchpointHit(name="hit1",
            ...                                             slot=1,
            ...                                             condition=2,
            ...                                             watchpoint_id=3,
            ...                                             parameters=[param1, param2],
            ...                                             error_code=0,
            ...                                             rank_id=1,
            ...                                             root_graph_id=1)
            >>> slot = watchpoint_hit.slot
        """
        return self.instance.get_slot()

    @property
    def condition(self):
        """
        Function to receive WatchpointHit condition.

        Returns:
            condition of WatchpointHit instance (int).

        Examples:
            >>> from mindspore.ccsrc.debug.debugger.offline_debug import dbg_services
            >>> watchpoint_hit = dbg_services.WatchpointHit(name="hit1",
            ...                                             slot=1,
            ...                                             condition=2,
            ...                                             watchpoint_id=3,
            ...                                             parameters=[param1, param2],
            ...                                             error_code=0,
            ...                                             rank_id=1,
            ...                                             root_graph_id=1)
            >>> condition = watchpoint_hit.condition
        """
        return self.instance.get_condition()

    @property
    def watchpoint_id(self):
        """
        Function to receive WatchpointHit watchpoint_id.

        Returns:
            watchpoint_id of WatchpointHit instance (int).

        Examples:
            >>> from mindspore.ccsrc.debug.debugger.offline_debug import dbg_services
            >>> watchpoint_hit = dbg_services.WatchpointHit(name="hit1",
            ...                                             slot=1,
            ...                                             condition=2,
            ...                                             watchpoint_id=3,
            ...                                             parameters=[param1, param2],
            ...                                             error_code=0,
            ...                                             rank_id=1,
            ...                                             root_graph_id=1)
            >>> watchpoint_id = watchpoint_hit.watchpoint_id
        """

        return self.instance.get_watchpoint_id()

    @property
    def parameters(self):
        """
        Function to receive WatchpointHit parameters.

        Returns:
            List of parameters of WatchpointHit instance (list).

        Examples:
            >>> from mindspore.ccsrc.debug.debugger.offline_debug import dbg_services
            >>> watchpoint_hit = dbg_services.WatchpointHit(name="hit1",
            ...                                             slot=1,
            ...                                             condition=2,
            ...                                             watchpoint_id=3,
            ...                                             parameters=[param1, param2],
            ...                                             error_code=0,
            ...                                             rank_id=1,
            ...                                             root_graph_id=1)
            >>> parameters = watchpoint_hit.parameters
        """
        params = self.instance.get_parameters()
        param_list = []
        for elem in params:
            tmp = Parameter(elem.get_name(),
                            elem.get_disabled(),
                            elem.get_value(),
                            elem.get_hit(),
                            elem.get_actual_value())
            param_list.append(tmp)
        return param_list

    @property
    def error_code(self):
        """
        Function to receive WatchpointHit error_code.

        Returns:
            error_code of WatchpointHit instance (int).

        Examples:
            >>> from mindspore.ccsrc.debug.debugger.offline_debug import dbg_services
            >>> watchpoint_hit = dbg_services.WatchpointHit(name="hit1",
            ...                                             slot=1,
            ...                                             condition=2,
            ...                                             watchpoint_id=3,
            ...                                             parameters=[param1, param2],
            ...                                             error_code=0,
            ...                                             rank_id=1,
            ...                                             root_graph_id=1)
            >>> error_code = watchpoint_hit.error_code
        """
        return self.instance.get_error_code()

    @property
    def rank_id(self):
        """
        Function to receive WatchpointHit rank_id.

        Returns:
            rank_id of WatchpointHit instance (int).

        Examples:
            >>> from mindspore.ccsrc.debug.debugger.offline_debug import dbg_services
            >>> watchpoint_hit = dbg_services.WatchpointHit(name="hit1",
            ...                                             slot=1,
            ...                                             condition=2,
            ...                                             watchpoint_id=3,
            ...                                             parameters=[param1, param2],
            ...                                             error_code=0,
            ...                                             rank_id=1,
            ...                                             root_graph_id=1)
            >>> rank_id = watchpoint_hit.rank_id
        """
        return self.instance.get_rank_id()

    @property
    def root_graph_id(self):
        """
        Function to receive WatchpointHit root_graph_id.

        Returns:
            root_graph_id of WatchpointHit instance (int).

        Examples:
            >>> from mindspore.ccsrc.debug.debugger.offline_debug import dbg_services
            >>> watchpoint_hit = dbg_services.WatchpointHit(name="hit1",
            ...                                             slot=1,
            ...                                             condition=2,
            ...                                             watchpoint_id=3,
            ...                                             parameters=[param1, param2],
            ...                                             error_code=0,
            ...                                             rank_id=1,
            ...                                             root_graph_id=1)
            >>> root_graph_id = watchpoint_hit.root_graph_id
        """
        return self.instance.get_root_graph_id()


class Parameter:
    """
    Parameter class.

    Args:
        name (str): Name of the parameter.
        disabled (bool): Whether parameter is used in backend.
        value (float): Threshold value of the parameter.
        hit (bool): Whether this parameter triggered watchpoint (default is False).
        actual_value (float): Actual value of the parameter (default is 0.0).

    Examples:
        >>> from mindspore.ccsrc.debug.debugger.offline_debug import dbg_services
        >>> parameter = dbg_services.Parameter(name="param",
        ...                                    disabled=False,
        ...                                    value=0.0,
        ...                                    hit=False,
        ...                                    actual_value=0.0)
    """
    @check_parameter_init
    def __init__(self, name, disabled, value, hit=False, actual_value=0.0):
        if security.enable_security():
            raise ValueError("Offline debugger is not supported in security mode. "
                             "Please recompile mindspore without `-s on`.")
        self.instance = cds.parameter(name, disabled, value, hit, actual_value)

    @property
    def name(self):
        """
        Function to receive Parameter name.

        Returns:
            name of Parameter instance (str).

        Examples:
            >>> from mindspore.ccsrc.debug.debugger.offline_debug import dbg_services
            >>> parameter = dbg_services.Parameter(name="param",
            ...                                    disabled=False,
            ...                                    value=0.0,
            ...                                    hit=False,
            ...                                    actual_value=0.0)
            >>> name = watchpoint_hit.name
        """

        return self.instance.get_name()

    @property
    def disabled(self):
        """
        Function to receive Parameter disabled value.

        Returns:
            disabled of Parameter instance (bool).

        Examples:
            >>> from mindspore.ccsrc.debug.debugger.offline_debug import dbg_services
            >>> parameter = dbg_services.Parameter(name="param",
            ...                                    disabled=False,
            ...                                    value=0.0,
            ...                                    hit=False,
            ...                                    actual_value=0.0)
            >>> disabled = watchpoint_hit.disabled
        """
        return self.instance.get_disabled()

    @property
    def value(self):
        """
        Function to receive Parameter value.

        Returns:
            value of Parameter instance (float).

        Examples:
            >>> from mindspore.ccsrc.debug.debugger.offline_debug import dbg_services
            >>> parameter = dbg_services.Parameter(name="param",
            ...                                    disabled=False,
            ...                                    value=0.0,
            ...                                    hit=False,
            ...                                    actual_value=0.0)
            >>> value = watchpoint_hit.value
        """
        return self.instance.get_value()

    @property
    def hit(self):
        """
        Function to receive Parameter hit value.

        Returns:
            hit of Parameter instance (bool).

        Examples:
            >>> from mindspore.ccsrc.debug.debugger.offline_debug import dbg_services
            >>> parameter = dbg_services.Parameter(name="param",
            ...                                    disabled=False,
            ...                                    value=0.0,
            ...                                    hit=False,
            ...                                    actual_value=0.0)
            >>> hit = watchpoint_hit.hit
        """
        return self.instance.get_hit()

    @property
    def actual_value(self):
        """
        Function to receive Parameter actual_value value.

        Returns:
            actual_value of Parameter instance (float).

        Examples:
            >>> from mindspore.ccsrc.debug.debugger.offline_debug import dbg_services
            >>> parameter = dbg_services.Parameter(name="param",
            ...                                    disabled=False,
            ...                                    value=0.0,
            ...                                    hit=False,
            ...                                    actual_value=0.0)
            >>> actual_value = watchpoint_hit.actual_value
        """
        return self.instance.get_actual_value()
