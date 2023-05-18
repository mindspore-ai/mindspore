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
# ============================================================================
"""base process"""
from __future__ import absolute_import

import os
import time
import math
import copy
import numpy as np
from scipy import linalg as la
from mindspore.context import ParallelMode
import mindspore.nn as nn
from mindspore.nn.optim import LARS
from mindspore import log as logger
from mindspore.common import Parameter
from mindspore.communication.management import get_group_size
from mindspore.train.serialization import load_checkpoint
from mindspore.parallel._utils import _get_global_rank
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.boost.less_batch_normalization import CommonHeadLastFN


__all__ = ["OptimizerProcess", "ParameterProcess"]


class OptimizerProcess:
    r"""
    Process optimizer for Boost. Currently, this class supports adding GC(grad centralization) tags
    and creating new optimizers.

    Args:
       opt (Cell): Optimizer used.

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor, Parameter, nn
        >>> from mindspore import ops
        >>> from mindspore.boost import OptimizerProcess
        >>>
        >>> class Net(nn.Cell):
        ...     def __init__(self, in_features, out_features):
        ...         super(Net, self).__init__()
        ...         self.weight = Parameter(Tensor(np.ones([in_features, out_features]).astype(np.float32)),
        ...                                 name='weight')
        ...         self.matmul = ops.MatMul()
        ...
        ...     def construct(self, x):
        ...         output = self.matmul(x, self.weight)
        ...         return output
        ...
        >>> size, in_features, out_features = 16, 16, 10
        >>> network = Net(in_features, out_features)
        >>> optimizer = nn.Momentum(network.trainable_params(), learning_rate=0.1, momentum=0.9)
        >>> optimizer_process = OptimizerProcess(optimizer)
        >>> optimizer_process.add_grad_centralization(network)
        >>> optimizer = optimizer_process.generate_new_optimizer()
    """
    def __init__(self, opt):
        if isinstance(opt, LARS):
            self.is_lars = True
            self.single_opt = opt.opt
            self.opt_class = type(opt.opt)
            self.opt_init_args = opt.opt.init_args
            self.lars_init_args = opt.init_args
            self.learning_rate = opt.opt.init_learning_rate
        else:
            self.is_lars = False
            self.single_opt = opt
            self.opt_class = type(opt)
            self.opt_init_args = opt.init_args
            self.learning_rate = opt.init_learning_rate
        self.origin_params = opt.init_params["params"]

    @staticmethod
    def build_params_dict(network):
        r"""
        Build the parameter's dict of the network.

        Args:
            network (Cell): The training network.
        """
        cells = network.cells_and_names()
        params_dict = {}
        for _, cell in cells:
            for par in cell.get_parameters(expand=False):
                params_dict[id(par)] = cell
        return params_dict

    @staticmethod
    def build_gc_params_group(params_dict, parameters):
        r"""
        Build the parameter's group with grad centralization.

        Args:
            params_dict (dict): The network's parameter dict.
            parameters (list): The network's parameter list.
        """
        group_params = []
        for group_param in parameters:
            if 'order_params' in group_param.keys():
                group_params.append(group_param)
                continue
            params_gc_value = []
            params_value = []
            for param in group_param['params']:
                if 'beta' not in param.name and 'gamma' not in param.name and 'bias' not in param.name:
                    param_cell = params_dict[id(param)]
                    if (isinstance(param_cell, nn.Conv2d) and param_cell.group > 1) or \
                        isinstance(param_cell, CommonHeadLastFN):
                        params_value.append(param)
                    else:
                        params_gc_value.append(param)
                else:
                    params_value.append(param)
            if params_gc_value:
                new_group_param = copy.deepcopy(group_param)
                new_group_param['params'] = params_gc_value
                new_group_param['grad_centralization'] = True
                group_params.append(new_group_param)
            if params_value:
                new_group_param = copy.deepcopy(group_param)
                new_group_param['params'] = params_value
                group_params.append(new_group_param)
        return group_params

    def add_grad_centralization(self, network):
        r"""
        Add gradient centralization.

        Args:
            network (Cell): The training network.
        """
        params_dict = OptimizerProcess.build_params_dict(network)

        if self.origin_params is not None and not isinstance(self.origin_params, list):
            parameters = list(self.origin_params)
        else:
            parameters = self.origin_params

        if not parameters:
            raise ValueError("Optimizer got an empty parameter list.")

        if not isinstance(parameters[0], (dict, Parameter)):
            raise TypeError("Only a list of Parameter or dict can be supported.")

        if isinstance(parameters[0], Parameter):
            logger.warning("Only group parameters support gradient centralization.")
            return

        self.origin_params = OptimizerProcess.build_gc_params_group(params_dict, parameters)

    def generate_new_optimizer(self):
        """Generate new optimizer."""
        if self.learning_rate is None:
            self.learning_rate = self.single_opt.learning_rate
        if not self.is_lars:
            opt = self.opt_class(params=self.origin_params, learning_rate=self.learning_rate, **self.opt_init_args)
        else:
            opt = LARS(self.opt_class(params=self.origin_params, learning_rate=self.learning_rate, \
                                      **self.opt_init_args), **self.lars_init_args)

        return opt


class ParameterProcess:
    r"""
    Process parameter for Boost. Currently, this class supports creating group parameters
    and automatically setting gradient segmentation point.

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor, Parameter, nn
        >>> import mindspore.ops as ops
        >>> from mindspore.boost import ParameterProcess
        >>>
        >>> class Net(nn.Cell):
        ...     def __init__(self, in_features, out_features):
        ...         super(Net, self).__init__()
        ...         self.weight = Parameter(Tensor(np.ones([in_features, out_features]).astype(np.float32)),
        ...                                 name='weight')
        ...         self.weight2 = Parameter(Tensor(np.ones([in_features, out_features]).astype(np.float32)),
        ...                                 name='weight2')
        ...         self.matmul = ops.MatMul()
        ...         self.matmul2 = ops.MatMul()
        ...
        ...     def construct(self, x):
        ...         output = self.matmul(x, self.weight)
        ...         output2 = self.matmul2(x, self.weight2)
        ...         return output + output2
        ...
        >>> size, in_features, out_features = 16, 16, 10
        >>> network = Net(in_features, out_features)
        >>> new_parameter = network.trainable_params()[:1]
        >>> group_params = ParameterProcess.generate_group_params(new_parameter, network.trainable_params())
    """
    def __init__(self):
        self._parameter_indices = 1

    @staticmethod
    def generate_group_params(parameters, origin_params):
        r"""
        Generate group parameters.

        Args:
            parameters (list): The network's parameter list.
            origin_params (list): The network's origin parameter list.
        """
        origin_params_copy = origin_params
        if origin_params_copy is not None:
            if not isinstance(origin_params_copy, list):
                origin_params_copy = list(origin_params_copy)

        if not origin_params_copy:
            raise ValueError("Optimizer got an empty parameter list.")

        if not isinstance(origin_params_copy[0], (dict, Parameter)):
            raise TypeError("Only a list of Parameter or dict can be supported.")

        if isinstance(origin_params_copy[0], Parameter):
            group_params = [{"params": parameters}]
            return group_params

        return ParameterProcess._generate_new_group_params(parameters, origin_params_copy)

    @staticmethod
    def _generate_new_group_params(parameters, origin_params_copy):
        r"""
        Generate new group parameters.

        Args:
            parameters (list): The network's parameter list.
            origin_params_copy (list): Copy from origin parameter list.
        """
        group_params = []
        params_name = [param.name for param in parameters]
        new_params_count = copy.deepcopy(params_name)
        new_params_clone = {}
        max_key_number = 0
        for group_param in origin_params_copy:
            if 'order_params' in group_param.keys():
                new_group_param = copy.deepcopy(group_param)
                new_group_param['order_params'] = parameters
                group_params.append(new_group_param)
                continue
            params_value = []
            for param in group_param['params']:
                if param.name in params_name:
                    index = params_name.index(param.name)
                    params_value.append(parameters[index])
                    new_params_count.remove(param.name)
            new_group_param = copy.deepcopy(group_param)
            new_group_param['params'] = params_value
            group_params.append(new_group_param)
            if len(group_param.keys()) > max_key_number:
                max_key_number = len(group_param.keys())
                new_params_clone = copy.deepcopy(group_param)
        if new_params_count:
            params_value = []
            for param in new_params_count:
                index = params_name.index(param)
                params_value.append(parameters[index])
            if new_params_clone:
                new_params_clone['params'] = params_value
                group_params.append(new_params_clone)
            else:
                group_params.append({"params": params_value})
        return group_params

    def assign_parameter_group(self, parameters, split_point=None):
        r"""
        Assign parameter group.

        Args:
            parameters (list): The network's parameter list.
            split_point (list): The gradient split point of this network. Default: ``None``.
        """
        if not isinstance(parameters, (list, tuple)) or not parameters:
            return parameters

        parameter_len = len(parameters)
        if split_point:
            split_parameter_index = split_point
        else:
            split_parameter_index = [parameter_len // 2]
        for i in range(parameter_len):
            if i in split_parameter_index:
                self._parameter_indices += 1
            parameters[i].comm_fusion = self._parameter_indices
        return parameters


def _get_local_pca_mat_path(weight_load_dir, pca_mat_path, n_component, device_number, network):
    """
    get local pca mat path.

    Args:
        weight_load_dir (str): The weight(ckpt) file directory to be load.
        pca_mat_path (str): the path to load pca mat. Default: ``None``.
        n_component (int): pca component.
        device_number (int): device number.
        network (Cell): The network.
    """
    if pca_mat_path is not None and os.path.exists(pca_mat_path) and os.path.isfile(pca_mat_path) and \
            pca_mat_path.endswith(".npy"):
        full_pca_mat_path = pca_mat_path
        pca_mat_exist = True

    else:
        if weight_load_dir is None or not os.path.exists(weight_load_dir) or not os.path.isdir(weight_load_dir):
            raise ValueError("The weight_load_dir: {} is None / not exists / not directory.".format(weight_load_dir))

        full_pca_mat_path = os.path.join(weight_load_dir, "pca_mat_temp.npy")
        pca_mat_exist = False

    save_pca_end_path = os.path.join(os.path.dirname(full_pca_mat_path), "save_pca_end.txt")
    if os.path.exists(save_pca_end_path):
        os.remove(save_pca_end_path)

    rank = _get_global_rank()
    local_pca_mat_path = full_pca_mat_path[:-4] + "_rank_" + str(rank) + ".npy"
    if os.path.exists(local_pca_mat_path):
        os.remove(local_pca_mat_path)
    if rank % device_number != 0:
        return local_pca_mat_path

    if pca_mat_exist:
        pca_mat = np.load(full_pca_mat_path)
    else:
        data = _load_weights(weight_load_dir, network)
        pca_mat = _compute_pca_mat(data, n_component)
        np.save(full_pca_mat_path, pca_mat)
    _save_local_pca_mat(pca_mat, full_pca_mat_path, n_component)
    return local_pca_mat_path


def _load_weights(weight_load_dir, network):
    """
    load weights.

    Args:
        weight_load_dir (str): The weight(ckpt) file directory to be load.
        network (Cell): The network.
    """
    param_requires_grad_list = []
    for param in network.trainable_params():
        param_requires_grad_list.append(param.name)

    param_mat_tuple = ()
    weight_file_list = os.listdir(weight_load_dir)
    for file in weight_file_list:
        if not file.endswith('.ckpt'):
            continue
        file_path = os.path.join(weight_load_dir, file)
        param_dict = load_checkpoint(file_path)
        param_tuple = ()
        for key, value in param_dict.items():
            if key in param_requires_grad_list:
                param_tuple += (value.asnumpy().reshape((1, -1)),)
        param = np.concatenate(param_tuple, axis=1)
        param_mat_tuple += (param,)
    param_mat = np.concatenate(param_mat_tuple, axis=0)
    return param_mat


def _compute_pca_mat(data, n_component, randomized=True):
    """
    compute pca mat.

    Args:
        data (array): array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        n_component (int): pca component.
        randomized (bool) if use randomized svd.
    """
    if data.shape[0] < n_component:
        raise ValueError("The samples: {} is less than: n_component {}.".format(data.shape[0], n_component))

    if randomized:
        components = _randomized_svd(data, n_component)
    else:
        components = _full_svd(data, n_component)

    return components


def _randomized_svd(data, n_component, n_oversample=10, n_iter=1):
    """
    compute pca mat use randomized svd.

    Args:
        data (array): array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        n_component (int): pca component.
        n_oversample (int): oversample num
        n_iter (int): iteration count
    """
    data -= np.mean(data, axis=0)
    n_random = n_component + n_oversample
    n_samples, n_features = data.shape
    transpose = n_samples < n_features
    if transpose:
        data = data.T
    q_mat = _randomized_range_finder(data, n_random, n_iter)
    b_mat = q_mat.T @ data
    u_hat, _, vt_mat = la.svd(b_mat, full_matrices=False)
    del b_mat
    u_mat = np.dot(q_mat, u_hat)
    u_mat, vt_mat = _svd_flip(u_mat, vt_mat, transpose)
    if transpose:
        components = u_mat[:, :n_component].T
    else:
        components = vt_mat[:n_component, :]
    return components


def _full_svd(data, n_component):
    """
    compute pca mat use full svd.

    Args:
        data (array): array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        n_component (int): pca component.
    """
    mean = np.mean(data, axis=0)
    data -= mean
    u, _, v = la.svd(data, full_matrices=False)
    _, v = _svd_flip(u, v)
    components = v[:n_component]
    return components


def _randomized_range_finder(data, size, n_iter=1):
    """
    compute pca mat use randomized svd.

    Args:
        data (array): array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        size (int): n_component + n_oversample.
        n_iter (int): iteration count
    """
    q_mat = np.random.normal(size=(data.shape[1], size))

    for _ in range(n_iter):
        q_mat, _ = la.lu(data @ q_mat, permute_l=True)
        q_mat, _ = la.lu(data.T @ q_mat, permute_l=True)

    q_mat, _ = la.qr(data @ q_mat, mode="economic")
    return q_mat


def _svd_flip(u, v, transpose=True):
    """
    svd flip.

    Args:
        u (ndarray): the output of `linalg.svd`.
        v (ndarray): the output of `linalg.svd`.
        transpose (bool): if data is transposed.
    """
    if not transpose:
        max_abs_cols = np.argmax(np.abs(u), axis=0)
        signs = np.sign(u[max_abs_cols, range(u.shape[1])])
        u *= signs
        v *= signs[:, np.newaxis]
    else:
        max_abs_rows = np.argmax(np.abs(v), axis=1)
        signs = np.sign(v[range(v.shape[0]), max_abs_rows])
        u *= signs
        v *= signs[:, np.newaxis]
    return u, v


def _save_local_pca_mat(pca_mat, full_pca_mat_path, n_component):
    """
    save pca mat.

    Args:
        pca_mat (numpy.ndarray): pca mat to be saved.
        full_pca_mat_path (str): the path of full pca mat.
        n_component (int): pca component.
    """
    parallel_mode = auto_parallel_context().get_parallel_mode()
    rank_size = 1 if parallel_mode == ParallelMode.STAND_ALONE else get_group_size()
    local_dim = math.ceil(n_component // rank_size)
    for rank_id in range(rank_size):
        start_index = rank_id * local_dim
        end_index = (rank_id + 1) * local_dim
        pca_start_index = min(n_component, start_index)
        pca_end_index = min(n_component, end_index)
        p_local = np.zeros([local_dim, pca_mat.shape[1]])
        if pca_start_index != pca_end_index:
            p_local[0: pca_end_index - pca_start_index, :] = pca_mat[pca_start_index: pca_end_index, :]
        local_pca_mat_path = "{}_rank_{}.npy".format(full_pca_mat_path[:-4], str(rank_id))
        np.save(local_pca_mat_path, p_local)
    save_pca_end_path = os.path.join(os.path.dirname(full_pca_mat_path), "save_pca_end.txt")
    os.mknod(save_pca_end_path)


def _load_local_pca_mat(local_pca_mat_path, timeout):
    """
    load pca mat.

    Args:
        local_pca_mat_path (str): local pca mat file path.
    """
    save_pca_end_path = os.path.join(os.path.dirname(local_pca_mat_path), "save_pca_end.txt")
    start_time = time.time()
    while True:
        current_time = time.time()
        if (current_time - start_time) > timeout:
            raise RuntimeError("the time of waiting to load local pca mat is larger than {} second.".format(timeout))
        if os.path.exists(save_pca_end_path):
            break
        time.sleep(5)
    pca_mat = np.load(local_pca_mat_path)
    return pca_mat
