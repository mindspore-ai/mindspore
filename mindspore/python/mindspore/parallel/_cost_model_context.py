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
"""Context of cost_model in auto_parallel"""
from __future__ import absolute_import
from __future__ import division

import threading

from mindspore._c_expression import CostModelContext
from mindspore._checkparam import args_type_check


class _CostModelContext:
    """
    _CostModelContext is the environment in which operations are executed

    Note:
        Creating a context through instantiating Context object is not recommended.
        Use cost_model_context() to get the context since Context is singleton.
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
        self._context_handle = CostModelContext.get_instance()

    def set_device_memory_capacity(self, dev_mem_cap):
        """
        Set device memory capacity.

        Args:
            dev_mem_cap (float): The memory capacity for each device.

        Raises:
            ValueError: If context handle is none.
        """
        if self._context_handle is None:
            raise ValueError("Context handle is none in context!!!")
        self._context_handle.set_device_memory_capacity(dev_mem_cap)

    def get_device_memory_capacity(self):
        """
        Get device memory capacity.

        Raises:
            ValueError: If context handle is none.
        """
        if self._context_handle is None:
            raise ValueError("Context handle is none in context!!!")
        return self._context_handle.get_device_memory_capacity()

    def set_costmodel_alpha(self, alpha):
        """
        Set costmodel alpha.

        Args:
            alpha (float): The parameter costmodel_alpha used in strategy-searching algorithm.

        Raises:
            ValueError: If context handle is none.
        """
        if self._context_handle is None:
            raise ValueError("Context handle is none in context!!!")
        self._context_handle.set_costmodel_alpha(alpha)

    def get_costmodel_alpha(self):
        """
        Get costmodel alpha.

        Raises:
            ValueError: If context handle is none.
        """
        if self._context_handle is None:
            raise ValueError("Context handle is none in context!!!")
        return self._context_handle.get_costmodel_alpha()

    def set_costmodel_beta(self, beta):
        """
        Set costmodel beta.

        Args:
            beta (float): The parameter costmodel_beta used in strategy-searching algorithm.

        Raises:
            ValueError: If context handle is none.
        """
        if self._context_handle is None:
            raise ValueError("Context handle is none in context!!!")
        self._context_handle.set_costmodel_beta(beta)

    def get_costmodel_beta(self):
        """
        Get costmodel beta.

        Raises:
            ValueError: If context handle is none.
        """
        if self._context_handle is None:
            raise ValueError("Context handle is none in context!!!")
        return self._context_handle.get_costmodel_beta()

    def set_costmodel_gamma(self, gamma):
        """
        Set costmodel gamma.

        Args:
            gamma (float): The parameter costmodel_gamma used in strategy-searching algorithm.

        Raises:
            ValueError: If context handle is none.
        """
        if self._context_handle is None:
            raise ValueError("Context handle is none in context!!!")
        self._context_handle.set_costmodel_gamma(gamma)

    def get_costmodel_gamma(self):
        """
        Get costmodel gamma.

        Raises:
            ValueError: If context handle is none.
        """
        if self._context_handle is None:
            raise ValueError("Context handle is none in context!!!")
        return self._context_handle.get_costmodel_gamma()

    def set_costmodel_communi_threshold(self, threshold):
        """
        Set costmodel communication threshold.

        Args:
            threshold (float): A parameter used in adjusting communication calculation for practice.

        Raises:
            ValueError: If context handle is none.
        """
        if self._context_handle is None:
            raise ValueError("Context handle is none in context!!!")
        self._context_handle.set_costmodel_communi_threshold(threshold)

    def get_costmodel_communi_threshold(self):
        """
        Get costmodel communication threshold.

        Raises:
            ValueError: If context handle is none.
        """
        if self._context_handle is None:
            raise ValueError("Context handle is none in context!!!")
        return self._context_handle.get_costmodel_communi_threshold()

    def set_costmodel_communi_const(self, communi_const):
        """
        Set costmodel communication const.

        Args:
            const (float): A parameter used in adjusting communication calculation for practice.

        Raises:
            ValueError: If context handle is none.
        """
        if self._context_handle is None:
            raise ValueError("Context handle is none in context!!!")
        self._context_handle.set_costmodel_communi_const(communi_const)

    def get_costmodel_communi_const(self):
        """
        Get costmodel communication const.

        Raises:
            ValueError: If context handle is none.
        """
        if self._context_handle is None:
            raise ValueError("Context handle is none in context!!!")
        return self._context_handle.get_costmodel_communi_const()

    def set_costmodel_communi_bias(self, communi_bias):
        """
        Set costmodel communication bias.

        Args:
            communi_bias (float): A parameter used in adjusting communication calculation for practice.

        Raises:
            ValueError: If context handle is none.
        """
        if self._context_handle is None:
            raise ValueError("Context handle is none in context!!!")
        self._context_handle.set_costmodel_communi_bias(communi_bias)

    def get_costmodel_communi_bias(self):
        """
        Get costmodel communication bias.

        Raises:
            ValueError: If context handle is none.
        """
        if self._context_handle is None:
            raise ValueError("Context handle is none in context!!!")
        return self._context_handle.get_costmodel_communi_bias()

    def set_multi_subgraphs(self, multi_subgraph):
        """
        Set the flag of ANF graph containing multiple subgraphs.

        Args:
            multi_subgraph (bool): A parameter used in marking the multi-subgraphs flag.

        Raises:
            ValueError: If context handle is none.
        """
        if self._context_handle is None:
            raise ValueError("Context handle is none in context!!!")
        self._context_handle.set_multi_subgraphs(multi_subgraph)

    def get_multi_subgraphs(self):
        """
        Get the flag of ANF graph containing multiple subgraphs.

        Raises:
            ValueError: If context handle is none.
        """
        if self._context_handle is None:
            raise ValueError("Context handle is none in context!!!")
        return self._context_handle.get_multi_subgraphs()

    def set_run_phase(self, phase):
        """
        Set the flag of running phase: training (0) or inference (1)

        Args:
            phase (int): A parameter indicating which phase is running.

        Raises:
            ValueError: If context handle is none, or phase is not in {0, 1}.
        """
        if not isinstance(phase, int) or isinstance(phase, bool):
            raise TypeError(f"For 'set_run_phase', the argument 'communi_const' must be int, but got {type(phase)}.")
        if self._context_handle is None:
            raise ValueError("Context handle is none in context!!!")
        if phase not in (0, 1):
            raise ValueError("For 'set_run_phase', the argument 'phase' must be '0' or '1', but got {}".format(phase))
        self._context_handle.set_run_phase(phase)

    def get_run_phase(self):
        """
        Get the flag of running phase.

        Raises:
            ValueError: If context handle is none.
        """
        if self._context_handle is None:
            raise ValueError("Context handle is none in context!!!")
        return self._context_handle.get_run_phase()

    def set_dp_algo_single_loop(self, single_loop):
        """
        Set the flag of generating a single suite of OperatorInfos in for-loop.

        Args:
            single_loop (bool): The parameter for the single loop flag.

        Raises:
            ValueError: If context handle is none.
        """
        if not isinstance(single_loop, bool):
            raise TypeError("For 'set_dp_algo_single_loop', the argument 'single_loop' must be bool, "
                            "but got the type : {}".format(type(single_loop)))
        if self._context_handle is None:
            raise ValueError("Context handle is none in context!!!")
        self._context_handle.set_dp_algo_single_loop(single_loop)

    def get_dp_algo_single_loop(self):
        """
        Get the flag of whether or not generating a single suite of OperatorInfos in for-loop.

        Raises:
            ValueError: If context handle is none.
        """
        if self._context_handle is None:
            raise ValueError("Context handle is none in context!!!")
        return self._context_handle.get_dp_algo_single_loop()

    def set_costmodel_allreduce_fusion_algorithm(self, algorithm):
        """
        Set costmodel allreduce fusion algorithm.

        Args:
            algorithm (int): The AllReduce fusion algorithm of parameter gradients.

        Raises:
            ValueError: If context handle is none.
        """
        if self._context_handle is None:
            raise ValueError("Context handle is none in context!!!")
        self._context_handle.set_costmodel_allreduce_fusion_algorithm(algorithm)

    def get_costmodel_allreduce_fusion_algorithm(self):
        """
        Get costmodel allreduce fusion algorithm.

        Raises:
            ValueError: If context handle is none.
        """
        if self._context_handle is None:
            raise ValueError("Context handle is none in context!!!")
        return self._context_handle.get_costmodel_allreduce_fusion_algorithm()

    def set_costmodel_allreduce_fusion_times(self, allreduce_fusion_times):
        """
        Set costmodel allreduce fusion times.

        Args:
            allreduce_fusion_times (int): The AllReduce fusion times of parameter gradients.

        Raises:
            ValueError: If context handle is none.
        """
        if self._context_handle is None:
            raise ValueError("Context handle is none in context!!!")
        self._context_handle.set_costmodel_allreduce_fusion_times(allreduce_fusion_times)

    def get_costmodel_allreduce_fusion_times(self):
        """
        Get costmodel allreduce fusion times.

        Raises:
            ValueError: If context handle is none.
        """
        if self._context_handle is None:
            raise ValueError("Context handle is none in context!!!")
        return self._context_handle.get_costmodel_allreduce_fusion_times()

    def set_costmodel_allreduce_fusion_tail_percent(self, tail_percent):
        """
        Set costmodel allreduce fusion tail percent.

        Args:
            tail_percent (int): The percentage of backward computing time corresponding to the last parameter gradients
                AllReduce in the whole backward computing time.

        Raises:
            ValueError: If context handle is none.
        """
        if self._context_handle is None:
            raise ValueError("Context handle is none in context!!!")
        self._context_handle.set_costmodel_allreduce_fusion_tail_percent(tail_percent)

    def get_costmodel_allreduce_fusion_tail_percent(self):
        """
        Get costmodel allreduce fusion tail percent.

        Raises:
            ValueError: If context handle is none.
        """
        if self._context_handle is None:
            raise ValueError("Context handle is none in context!!!")
        return self._context_handle.get_costmodel_allreduce_fusion_tail_percent()

    def set_costmodel_allreduce_fusion_tail_time(self, tail_time):
        """
        Set costmodel allreduce fusion tail time.

        Args:
            tail_time (int): The tail time of the last parameter gradients AllReduce after the end of backward
                computation.

        Raises:
            ValueError: If context handle is none.
        """
        if self._context_handle is None:
            raise ValueError("Context handle is none in context!!!")
        self._context_handle.set_costmodel_allreduce_fusion_tail_time(tail_time)

    def get_costmodel_allreduce_fusion_tail_time(self):
        """
        Get costmodel allreduce fusion tail time.

        Raises:
            ValueError: If context handle is none.
        """
        if self._context_handle is None:
            raise ValueError("Context handle is none in context!!!")
        return self._context_handle.get_costmodel_allreduce_fusion_tail_time()

    def set_costmodel_allreduce_fusion_allreduce_inherent_time(self, allreduce_inherent_time):
        """
        Set costmodel allreduce fusion allreduce inherent time.

        Args:
            allreduce_inherent_time (int): The inherent cost time of AllReduce.

        Raises:
            ValueError: If context handle is none.
        """
        if self._context_handle is None:
            raise ValueError("Context handle is none in context!!!")
        self._context_handle.set_costmodel_allreduce_fusion_allreduce_inherent_time(allreduce_inherent_time)

    def get_costmodel_allreduce_fusion_allreduce_inherent_time(self):
        """
        Get costmodel allreduce fusion allreduce inherent time.

        Raises:
            ValueError: If context handle is none.
        """
        if self._context_handle is None:
            raise ValueError("Context handle is none in context!!!")
        return self._context_handle.get_costmodel_allreduce_fusion_allreduce_inherent_time()

    def set_costmodel_allreduce_fusion_allreduce_bandwidth(self, allreduce_bandwidth):
        """
        Set costmodel allreduce fusion allreduce bandwidth.

        Args:
            allreduce_bandwidth (int): The bandwidth of AllReduce.

        Raises:
            ValueError: If context handle is none.
        """
        if self._context_handle is None:
            raise ValueError("Context handle is none in context!!!")
        self._context_handle.set_costmodel_allreduce_fusion_allreduce_bandwidth(allreduce_bandwidth)

    def get_costmodel_allreduce_fusion_allreduce_bandwidth(self):
        """
        Get costmodel allreduce fusion allreduce bandwidth.

        Raises:
            ValueError: If context handle is none.
        """
        if self._context_handle is None:
            raise ValueError("Context handle is none in context!!!")
        return self._context_handle.get_costmodel_allreduce_fusion_allreduce_bandwidth()

    def set_rp_matmul_mem_coef(self, coef):
        """
        Set the matmul memory coef which is used in the RP algorithm.

        Args:
            coef (int): The coefficient of memory cost in RP.

        Raises:
            ValueError: If context handle is none.
        """
        if self._context_handle is None:
            raise ValueError("Context handle is none in context!!!")
        self._context_handle.set_rp_matmul_mem_coef(coef)

    def get_rp_matmul_mem_coef(self):
        """
        Get the matmul memory coef which is used in the RP algorithm.

        Raises:
            ValueError: If context handle is none.
        """
        if self._context_handle is None:
            raise ValueError("Context handle is none in context!!!")
        return self._context_handle.get_rp_matmul_mem_coef()

    def set_costmodel_allreduce_fusion_computation_time_parameter(self, computation_time_parameter):
        """
        Set costmodel allreduce fusion computation time parameter.

        Args:
            computation_time_parameter (int): The parameter used to compute backward computation time.

        Raises:
            ValueError: If context handle is none.
        """
        if self._context_handle is None:
            raise ValueError("Context handle is none in context!!!")
        self._context_handle.set_costmodel_allreduce_fusion_computation_time_parameter(computation_time_parameter)

    def get_costmodel_allreduce_fusion_computation_time_parameter(self):
        """
        Get costmodel allreduce fusion computation time parameter.

        Raises:
            ValueError: If context handle is none.
        """
        if self._context_handle is None:
            raise ValueError("Context handle is none in context!!!")
        return self._context_handle.get_costmodel_allreduce_fusion_computation_time_parameter()

    def reset_cost_model(self):
        """
        Reset cost model settings.

        Raises:
            ValueError: If context handle is none.
        """
        if self._context_handle is None:
            raise ValueError("Context handle is none in context!!!")
        self._context_handle.reset_cost_model()


_COST_MODEL_CONTEXT = None


def cost_model_context():
    """
    Get the global _COST_MODEL_CONTEXT. If it is not created, create a new one.

    Returns:
        The global cost_model context.
    """
    global _COST_MODEL_CONTEXT
    if _COST_MODEL_CONTEXT is None:
        _COST_MODEL_CONTEXT = _CostModelContext()
    return _COST_MODEL_CONTEXT


set_cost_model_context_func_map = {
    "device_memory_capacity": cost_model_context().set_device_memory_capacity,
    "costmodel_alpha": cost_model_context().set_costmodel_alpha,
    "costmodel_beta": cost_model_context().set_costmodel_beta,
    "costmodel_gamma": cost_model_context().set_costmodel_gamma,
    "costmodel_communi_threshold": cost_model_context().set_costmodel_communi_threshold,
    "costmodel_communi_const": cost_model_context().set_costmodel_communi_const,
    "costmodel_communi_bias": cost_model_context().set_costmodel_communi_bias,
    "run_phase": cost_model_context().set_run_phase,
    "costmodel_allreduce_fusion_algorithm": cost_model_context().set_costmodel_allreduce_fusion_algorithm,
    "costmodel_allreduce_fusion_times": cost_model_context().set_costmodel_allreduce_fusion_times,
    "costmodel_allreduce_fusion_tail_percent": cost_model_context().set_costmodel_allreduce_fusion_tail_percent,
    "costmodel_allreduce_fusion_tail_time": cost_model_context().set_costmodel_allreduce_fusion_tail_time,
    "costmodel_allreduce_fusion_allreduce_inherent_time":
        cost_model_context().set_costmodel_allreduce_fusion_allreduce_inherent_time,
    "costmodel_allreduce_fusion_allreduce_bandwidth":
        cost_model_context().set_costmodel_allreduce_fusion_allreduce_bandwidth,
    "costmodel_allreduce_fusion_computation_time_parameter":
        cost_model_context().set_costmodel_allreduce_fusion_computation_time_parameter}


get_cost_model_context_func_map = {
    "device_memory_capacity": cost_model_context().get_device_memory_capacity,
    "costmodel_alpha": cost_model_context().get_costmodel_alpha,
    "costmodel_beta": cost_model_context().get_costmodel_beta,
    "costmodel_gamma": cost_model_context().get_costmodel_gamma,
    "costmodel_communi_threshold": cost_model_context().get_costmodel_communi_threshold,
    "costmodel_communi_const": cost_model_context().get_costmodel_communi_const,
    "costmodel_communi_bias": cost_model_context().get_costmodel_communi_bias,
    "run_phase": cost_model_context().get_run_phase,
    "costmodel_allreduce_fusion_algorithm": cost_model_context().get_costmodel_allreduce_fusion_algorithm,
    "costmodel_allreduce_fusion_times": cost_model_context().get_costmodel_allreduce_fusion_times,
    "costmodel_allreduce_fusion_tail_percent": cost_model_context().get_costmodel_allreduce_fusion_tail_percent,
    "costmodel_allreduce_fusion_tail_time": cost_model_context().get_costmodel_allreduce_fusion_tail_time,
    "costmodel_allreduce_fusion_allreduce_inherent_time":
        cost_model_context().get_costmodel_allreduce_fusion_allreduce_inherent_time,
    "costmodel_allreduce_fusion_allreduce_bandwidth":
        cost_model_context().get_costmodel_allreduce_fusion_allreduce_bandwidth,
    "costmodel_allreduce_fusion_computation_time_parameter":
        cost_model_context().get_costmodel_allreduce_fusion_computation_time_parameter}


@args_type_check(device_memory_capacity=float, costmodel_alpha=float, costmodel_beta=float, costmodel_gamma=float,
                 costmodel_communi_threshold=float, costmodel_communi_const=float, costmodel_communi_bias=float,
                 multi_subgraphs=bool, run_phase=int,
                 costmodel_allreduce_fusion_algorithm=int, costmodel_allreduce_fusion_times=int,
                 costmodel_allreduce_fusion_tail_percent=float, costmodel_allreduce_fusion_tail_time=float,
                 costmodel_allreduce_fusion_allreduce_inherent_time=float,
                 costmodel_allreduce_fusion_allreduce_bandwidth=float,
                 costmodel_allreduce_fusion_computation_time_parameter=float)
def set_cost_model_context(**kwargs):
    """
    Set cost model context.

    Note:
        Attribute name is needed.

    Args:
        device_memory_capacity (float): The memory capacity for each device.
        costmodel_alpha (float): The parameter costmodel_alpha used in strategy-searching algorithm.
        costmodel_beta (float): The parameter costmodel_beta used in strategy-searching algorithm.
        costmodel_gamma (float): The parameter costmodel_gamma used in strategy-searching algorithm.
        costmodel_communi_threshold (float): A parameter used in adjusting communication calculation for practice.
        costmodel_communi_const (float): A parameter used in adjusting communication calculation for practice.
        costmodel_communi_bias (float): A parameter used in adjusting communication calculation for practice.
        run_phase (int): A parameter indicating which phase is running: training (0) or inference (1). Default: 0.
        costmodel_allreduce_fusion_algorithm (int): The allreduce fusion algorithm.
            0: bypass allreduce fusion;
            1: only use backward computation time to group allreduce;
            2: use backward computation time and parameter gradient allreduce time to group allreduce.
        costmodel_allreduce_fusion_times (int): The AllReduce fusion times of parameter gradients.
        costmodel_allreduce_fusion_tail_percent (float): A parameter used in allreduce fusion algorithm. The percentage
            of backward computing time corresponding to the last parameter gradients AllReduce in the whole backward
            computing time.
        costmodel_allreduce_fusion_tail_time (float): A parameter used in allreduce fusion algorithm. The tail time of
            the last parameter gradients AllReduce after the end of backward computation.
        costmodel_allreduce_fusion_allreduce_inherent_time (float): A parameter used in allreduce fusion algorithm. The
            inherent cost time of AllReduce.
        costmodel_allreduce_fusion_allreduce_bandwidth (float): A parameter used in allreduce fusion algorithm. The
            bandwidth of AllReduce.
        costmodel_allreduce_fusion_computation_time_parameter (float): A parameter used in allreduce fusion algorithm.
            The parameter used to compute backward computation time.



    Raises:
        ValueError: If context keyword is not recognized.
    """
    for key, value in kwargs.items():
        if key not in set_cost_model_context_func_map:
            raise ValueError("Set context keyword %s is not recognized!" % key)
        set_func = set_cost_model_context_func_map[key]
        set_func(value)


def get_cost_model_context(attr_key):
    """
    Get cost model context attributes.

    Note:
        Return value according to the attribute value.

    Args:
        attr_key (str): The key of the attribute.

    Raises:
        ValueError: If context keyword is not recognized.
    """
    if attr_key not in get_cost_model_context_func_map:
        raise ValueError("Get context keyword %s is not recognized!" % attr_key)
    get_func = get_cost_model_context_func_map[attr_key]
    return get_func()


def reset_cost_model_context():
    """Reset cost model context attributes."""
    cost_model_context().reset_cost_model()


def _set_multi_subgraphs(multi_subgraph=True):
    """
    Set the flag of ANF graph containing multiple subgraphs.

    Args:
        multi_subgraph (bool): A parameter used in marking the multi-subgraphs flag.
    """
    cost_model_context().set_multi_subgraphs(multi_subgraph)


def _get_multi_subgraphs():
    """
        Get the flag of ANF graph containing multiple subgraphs.
    """
    return cost_model_context().get_multi_subgraphs()


def _set_algo_single_loop(single_loop=True):
    """
    Set the flag of generating a single suite of OperatorInfos in for-loop.

    Args:
        single_loop (bool): The parameter for the single loop flag.
    """
    cost_model_context().set_dp_algo_single_loop(single_loop)


def _get_algo_single_loop():
    """
    Get the flag of whether or not generating a single suite of OperatorInfos in for-loop.
    """
    return cost_model_context().get_dp_algo_single_loop()


def _set_rp_matmul_mem_coef(coef):
    """
    Set the matmul memory coef which is used in the RP algorithm.

    Args:
        coef (int): The coefficient of memory cost in RP.
    """
    cost_model_context().set_rp_matmul_mem_coef(coef)


def _get_rp_matmul_mem_coef():
    """
    Get the matmul memory coef which is used in the RP algorithm.
    """
    return cost_model_context().get_rp_matmul_mem_coef()
