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

import mindspore._mindspore_offline_debug as cds
from mi_validators import check_init, check_initialize, check_add_watchpoint, check_remove_watchpoint, check_check_watchpoints, check_read_tensors, check_initialize_done, check_tensor_info_init, check_tensor_data_init, check_watchpoint_hit_init, check_parameter_init


def get_version():
    """
    Function to return offline Debug Services version.

    Returns:
        version (str): dbgServices version.

    Examples:
        >>> from mindspore.ccsrc.debug.debugger.offline_debug import dbg_services
        >>> version = dbg_services.get_version()
    """
    return cds.DbgServices(False).GetVersion()

class DbgLogger:
    """
    Offline Debug Services Logger

    Args:
        verbose (bool): whether to print logs.

    Examples:
        >>> from mindspore.ccsrc.debug.debugger.offline_debug import dbg_services
        >>> version = dbg_services.DbgLogger(verbose=False)
    """
    def __init__(self, verbose):
        self.verbose = verbose

    def __call__(self, *logs):
        if self.verbose:
            print(logs)


log = DbgLogger(False)


class DbgServices():
    """
    Offline Debug Services class.

    Args:
        dump_file_path (str): directory where the dump files are saved.
        verbose (bool): whether to print logs (default: False)..

    Examples:
        >>> from mindspore.ccsrc.debug.debugger.offline_debug import dbg_services
        >>> d = dbg_services.DbgServices(dump_file_path="dump_file_path",
        >>>                              verbose=True)
    """

    @check_init
    def __init__(self, dump_file_path, verbose=False):
        log.verbose = verbose
        log("in Python __init__, file path is ", dump_file_path)
        self.dump_file_path = dump_file_path
        self.dbg_instance = cds.DbgServices(verbose)
        self.version = self.dbg_instance.GetVersion()
        self.verbose = verbose
        self.initialized = False

    @check_initialize
    def initialize(self, net_name, is_sync_mode=True):
        """
        Initialize Debug Service.

        Args:
            net_name (str): Network name.
            is_sync_mode (bool): Whether to process synchronous or asynchronous dump files mode
                                 (default: True (synchronous)).

        Returns:
            Initialized Debug Service instance.

        Examples:
        >>> from mindspore.ccsrc.debug.debugger.offline_debug import dbg_services
        >>> d = dbg_services.DbgServices(dump_file_path="dump_file_path",
        >>>                              verbose=True)
        >>> d_init = d.initialize(net_name="network name", is_sync_mode=True)
        """

        log("in Python Initialize dump_file_path ", self.dump_file_path)
        self.initialized = True
        return self.dbg_instance.Initialize(net_name, self.dump_file_path, is_sync_mode)

    @check_initialize_done
    @check_add_watchpoint
    def add_watchpoint(self, watchpoint_id, watch_condition, check_node_list, parameter_list):
        """
        Adding watchpoint to Debug Service instance.

        Args:
            watchpoint_id (int): Watchpoint id
            watch_condition (int): A representation of the condition to be checked.
            check_node_list (dict): Dictionary of node names (str) as key,
                                    mapping to device_id (list of ints), root_graph_id (list of ints) and is_parameter
                                    (bool).
            parameter_list (list): List of parameters in watchpoint. Parameters should be instances of Parameter class.
                                   Each parameter describes the value to be checked in watchpoint.

        Returns:
            Debug Service instance with added watchpoint.

        Examples:
        >>> from mindspore.ccsrc.debug.debugger.offline_debug import dbg_services
        >>> d = dbg_services.DbgServices(dump_file_path="dump_file_path",
        >>>                              verbose=True)
        >>> d_init = d.initialize(is_sync_mode=True)
        >>> d_wp = d_init.add_watchpoint(watchpoint_id=1,
        >>>                              watch_condition=6,
        >>>                              check_node_list={"conv2.bias" : {"device_id": [0],
                                                                          root_graph_id: [0], "is_parameter": True}},
        >>>                              parameter_list=[dbg_services.Parameter(name="param",
        >>>                                                                     disabled=False,
        >>>                                                                     value=0.0,
        >>>                                                                     hit=False,
        >>>                                                                     actual_value=0.0)])
        """

        print("Amir: ", check_node_list)

        log("in Python AddWatchpoint")
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
            watchpoint_id (int): Watchpoint id

        Returns:
            Debug Service instance with removed watchpoint.

        Examples:
        >>> from mindspore.ccsrc.debug.debugger.offline_debug import dbg_services
        >>> d = dbg_services.DbgServices(dump_file_path="dump_file_path",
        >>>                              verbose=True)
        >>> d_init = d.initialize(is_sync_mode=True)
        >>> d_wp = d_init.add_watchpoint(watchpoint_id=1,
        >>>                              watch_condition=6,
        >>>                              check_node_list={"conv2.bias" : {"device_id": [5],
                                                                          root_graph_id: [0], "is_parameter": True}},
        >>>                              parameter_list=[dbg_services.Parameter(name="param",
        >>>                                                                     disabled=False,
        >>>                                                                     value=0.0,
        >>>                                                                     hit=False,
        >>>                                                                     actual_value=0.0)])
        >>> d_wp = d_wp.remove_watchpoint(watchpoint_id=1)
        """

        log("in Python Remove Watchpoint id ", watchpoint_id)
        return self.dbg_instance.RemoveWatchpoint(watchpoint_id)

    @check_initialize_done
    @check_check_watchpoints
    def check_watchpoints(self, iteration):
        """
        Checking watchpoint at given iteration.

        Args:
            iteration (int): Watchpoint check iteration.

        Returns:
            Watchpoint hit list.

        Examples:
        >>> from mindspore.ccsrc.debug.debugger.offline_debug import dbg_services
        >>> d = dbg_services.DbgServices(dump_file_path="dump_file_path",
        >>>                              verbose=True)
        >>> d_init = d.initialize(is_sync_mode=True)
        >>> d_wp = d_init.add_watchpoint(id=1,
        >>>                              watch_condition=6,
        >>>                              check_node_list={"conv2.bias" : {"device_id": [5],
                                                                          root_graph_id: [0], "is_parameter": True}},
        >>>                              parameter_list=[dbg_services.Parameter(name="param",
        >>>                                                                     disabled=False,
        >>>                                                                     value=0.0,
        >>>                                                                     hit=False,
        >>>                                                                     actual_value=0.0)])
        >>> watchpoints = d_wp.check_watchpoints(iteration=8)
        """

        log("in Python CheckWatchpoints iteration ", iteration)
        watchpoint_list = self.dbg_instance.CheckWatchpoints(iteration)
        watchpoint_hit_list = []
        for watchpoint in watchpoint_list:
            name = watchpoint.get_name()
            slot = watchpoint.get_slot()
            condition = watchpoint.get_condition()
            watchpoint_id = watchpoint.get_watchpoint_id()
            parameters = watchpoint.get_parameters()
            error_code = watchpoint.get_error_code()
            device_id = watchpoint.get_device_id()
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
                                                     param_list, error_code, device_id, root_graph_id))
        return watchpoint_hit_list

    @check_initialize_done
    @check_read_tensors
    def read_tensors(self, info):
        """
        Returning tensor data object describing the tensor requested tensor.

        Args:
            info (list): List of TensorInfo objects.

        Returns:
            TensorData list (list).

        Examples:
        >>> from mindspore.ccsrc.debug.debugger.offline_debug import dbg_services
        >>> d = dbg_services.DbgServices(dump_file_path="dump_file_path",
        >>>                              verbose=True)
        >>> d_init = d.initialize(is_sync_mode=True)
        >>> tensor_data_list = d_init.read_tensors([dbg_services.TensorInfo(node_name="conv2.bias",
        >>>                                                                 slot=0,
        >>>                                                                 iteration=8,
        >>>                                                                 device_id=5,
        >>>                                                                 root_graph_id=0,
        >>>                                                                 is_parameter=True)])
        """

        log("in Python ReadTensors info ", info)
        info_list_inst = []
        for elem in info:
            log("in Python ReadTensors info ", info)
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

class TensorInfo():
    """
    Tensor Information class.

    Args:
        node_name (str): Fully qualified name of the desired node.
        slot (int): The particular output for the requested node.
        iteration (int): The desired itraretion to gather tensor information.
        device_id (int): The desired device id to gather tensor information.
        is_parameter (bool): Whether node is a parameter (input, constant, bias, parameter).

    Examples:
        >>> from mindspore.ccsrc.debug.debugger.offline_debug import dbg_services
        >>> tensor_info = dbg_services.TensorInfo(node_name="conv2.bias",
        >>>                                       slot=0,
        >>>                                       iteration=8,
        >>>                                       device_id=5,
        >>>                                       root_graph_id=0,
        >>>                                       is_parameter=True)
    """

    @check_tensor_info_init
    def __init__(self, node_name, slot, iteration, device_id, root_graph_id, is_parameter):
        self.instance = cds.tensor_info(node_name, slot, iteration, device_id, root_graph_id, is_parameter)

    @property
    def node_name(self):
        """
        Function to receive TensorInfo node_name.

        Returns:
            node_name of TensorInfo instance (str).

        Examples:
            >>> from mindspore.ccsrc.debug.debugger.offline_debug import dbg_services
            >>> tensor_info = dbg_services.TensorInfo(node_name="conv2.bias",
            >>>                                       slot=0,
            >>>                                       iteration=8,
            >>>                                       device_id=5,
            >>>                                       root_graph_id=0,
            >>>                                       is_parameter=True)
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
            >>>                                       slot=0,
            >>>                                       iteration=8,
            >>>                                       device_id=5,
            >>>                                       root_graph_id=0,
            >>>                                       is_parameter=True)
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
            >>>                                       slot=0,
            >>>                                       iteration=8,
            >>>                                       device_id=5,
            >>>                                       root_graph_id=0,
            >>>                                       is_parameter=True)
            >>> iteration = tensor_info.iteration
        """

        return self.instance.get_iteration()

    @property
    def device_id(self):
        """
        Function to receive TensorInfo device_id.

        Returns:
            device_id of TensorInfo instance (int).

        Examples:
            >>> from mindspore.ccsrc.debug.debugger.offline_debug import dbg_services
            >>> tensor_info = dbg_services.TensorInfo(node_name="conv2.bias",
            >>>                                       slot=0,
            >>>                                       iteration=8,
            >>>                                       device_id=5,
            >>>                                       root_graph_id=0,
            >>>                                       is_parameter=True)
            >>> device_id = tensor_info.device_id
        """

    @property
    def root_graph_id(self):
        """
        Function to receive TensorInfo root_graph_id.

        Returns:
            root_graph_id of TensorInfo instance (int).

        Examples:
            >>> from mindspore.ccsrc.debug.debugger.offline_debug import dbg_services
            >>> tensor_info = dbg_services.TensorInfo(node_name="conv2.bias",
            >>>                                       slot=0,
            >>>                                       iteration=8,
            >>>                                       device_id=5,
            >>>                                       root_graph_id=0,
            >>>                                       is_parameter=True)
            >>> device_id = tensor_info.root_graph_id
        """

        return self.instance.get_root_graph_id()

    @property
    def is_parameter(self):
        """
        Function to receive TensorInfo is_parameter.

        Returns:
            is_parameter of TensorInfo instance (bool).

        Examples:
            >>> from mindspore.ccsrc.debug.debugger.offline_debug import dbg_services
            >>> tensor_info = dbg_services.TensorInfo(node_name="conv2.bias",
            >>>                                       slot=0,
            >>>                                       iteration=8,
            >>>                                       device_id=5,
            >>>                                       root_graph_id=0,
            >>>                                       is_parameter=True)
            >>> is_parameter = tensor_info.is_parameter
        """

        return self.instance.get_is_parameter()

class TensorData():
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
        >>>                                       data_size=4,
        >>>                                       dtype=0,
        >>>                                       shape=[2, 2])
    """

    @check_tensor_data_init
    def __init__(self, data_ptr, data_size, dtype, shape):
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
            >>>                                       data_size=4,
            >>>                                       dtype=0,
            >>>                                       shape=[2, 2])
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
            >>>                                       data_size=4,
            >>>                                       dtype=0,
            >>>                                       shape=[2, 2])
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
            >>>                                       data_size=4,
            >>>                                       dtype=0,
            >>>                                       shape=[2, 2])
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
            >>>                                       data_size=4,
            >>>                                       dtype=0,
            >>>                                       shape=[2, 2])
            >>> shape = tensor_data.shape
        """

        return self.instance.get_shape()

class WatchpointHit():
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
        device_id (int): Device id where the watchpoint is hit.
        root_graph_id (int): Root graph id where the watchpoint is hit.

    Examples:
        >>> from mindspore.ccsrc.debug.debugger.offline_debug import dbg_services
        >>> watchpoint_hit = dbg_services.WatchpointHit(name="hit1",
        >>>                                             slot=1,
        >>>                                             condition=2,
        >>>                                             watchpoint_id=3,
        >>>                                             parameters=[param1, param2],
        >>>                                             error_code=0,
        >>>                                             device_id=1,
        >>>                                             root_graph_id=1)
    """

    @check_watchpoint_hit_init
    def __init__(self, name, slot, condition, watchpoint_id, parameters, error_code, device_id, root_graph_id):
        parameter_list_inst = []
        for elem in parameters:
            parameter_list_inst.append(elem.instance)
        self.instance = cds.watchpoint_hit(name, slot, condition, watchpoint_id,
                                           parameter_list_inst, error_code, device_id, root_graph_id)

    @property
    def name(self):
        """
        Function to receive WatchpointHit name.

        Returns:
            name of WatchpointHit instance (str).

        Examples:
            >>> from mindspore.ccsrc.debug.debugger.offline_debug import dbg_services
            >>> watchpoint_hit = dbg_services.WatchpointHit(name="hit1",
            >>>                                             slot=1,
            >>>                                             condition=2,
            >>>                                             watchpoint_id=3,
            >>>                                             parameters=[param1, param2],
            >>>                                             error_code=0,
            >>>                                             device_id=1,
            >>>                                             root_graph_id=1)
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
            >>>                                             slot=1,
            >>>                                             condition=2,
            >>>                                             watchpoint_id=3,
            >>>                                             parameters=[param1, param2],
            >>>                                             error_code=0,
            >>>                                             device_id=1,
            >>>                                             root_graph_id=1)
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
            >>>                                             slot=1,
            >>>                                             condition=2,
            >>>                                             watchpoint_id=3,
            >>>                                             parameters=[param1, param2],
            >>>                                             error_code=0,
            >>>                                             device_id=1,
            >>>                                             root_graph_id=1)
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
            >>>                                             slot=1,
            >>>                                             condition=2,
            >>>                                             watchpoint_id=3,
            >>>                                             parameters=[param1, param2],
            >>>                                             error_code=0,
            >>>                                             device_id=1,
            >>>                                             root_graph_id=1)
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
            >>>                                             slot=1,
            >>>                                             condition=2,
            >>>                                             watchpoint_id=3,
            >>>                                             parameters=[param1, param2],
            >>>                                             error_code=0,
            >>>                                             device_id=1,
            >>>                                             root_graph_id=1)
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
            >>>                                             slot=1,
            >>>                                             condition=2,
            >>>                                             watchpoint_id=3,
            >>>                                             parameters=[param1, param2],
            >>>                                             error_code=0,
            >>>                                             device_id=1,
            >>>                                             root_graph_id=1)
            >>> error_code = watchpoint_hit.error_code
        """

        return self.instance.get_error_code()

    @property
    def device_id(self):
        """
        Function to receive WatchpointHit device_id.

        Returns:
            device_id of WatchpointHit instance (int).

        Examples:
            >>> from mindspore.ccsrc.debug.debugger.offline_debug import dbg_services
            >>> watchpoint_hit = dbg_services.WatchpointHit(name="hit1",
            >>>                                             slot=1,
            >>>                                             condition=2,
            >>>                                             watchpoint_id=3,
            >>>                                             parameters=[param1, param2],
            >>>                                             error_code=0,
            >>>                                             device_id=1,
            >>>                                             root_graph_id=1)
            >>> device_id = watchpoint_hit.device_id
        """

        return self.instance.get_device_id()

    @property
    def root_graph_id(self):
        """
        Function to receive WatchpointHit root_graph_id.

        Returns:
            root_graph_id of WatchpointHit instance (int).

        Examples:
            >>> from mindspore.ccsrc.debug.debugger.offline_debug import dbg_services
            >>> watchpoint_hit = dbg_services.WatchpointHit(name="hit1",
            >>>                                             slot=1,
            >>>                                             condition=2,
            >>>                                             watchpoint_id=3,
            >>>                                             parameters=[param1, param2],
            >>>                                             error_code=0,
            >>>                                             device_id=1,
            >>>                                             root_graph_id=1)
            >>> root_graph_id = watchpoint_hit.root_graph_id
        """

        return self.instance.get_root_graph_id()

class Parameter():
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
        >>>                                    disabled=False,
        >>>                                    value=0.0,
        >>>                                    hit=False,
        >>>                                    actual_value=0.0)
    """

    @check_parameter_init
    def __init__(self, name, disabled, value, hit=False, actual_value=0.0):
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
            >>>                                    disabled=False,
            >>>                                    value=0.0,
            >>>                                    hit=False,
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
            >>>                                    disabled=False,
            >>>                                    value=0.0,
            >>>                                    hit=False,
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
            >>>                                    disabled=False,
            >>>                                    value=0.0,
            >>>                                    hit=False,
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
            >>>                                    disabled=False,
            >>>                                    value=0.0,
            >>>                                    hit=False,
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
            >>>                                    disabled=False,
            >>>                                    value=0.0,
            >>>                                    hit=False,
            >>> actual_value = watchpoint_hit.actual_value
        """

        return self.instance.get_actual_value()
