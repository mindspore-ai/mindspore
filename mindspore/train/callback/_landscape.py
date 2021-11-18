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
"""Process data and Calc loss landscape."""
import os
import time
import json
import shutil

from collections import defaultdict, namedtuple
from concurrent.futures import wait, ALL_COMPLETED, ProcessPoolExecutor

import numpy as np
from sklearn.decomposition import PCA

from mindspore import log as logger
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
from mindspore.train.serialization import save_checkpoint, load_checkpoint, load_param_into_net
from mindspore.train.summary_pb2 import LossLandscape
from mindspore.train.summary import SummaryRecord
from mindspore.train.summary.enums import PluginEnum
from mindspore.train.anf_ir_pb2 import DataType
from mindspore.train._utils import check_value_type, _make_directory
from mindspore.train.dataset_helper import DatasetHelper, connect_network_with_dataset
from mindspore.nn.metrics import get_metrics
from mindspore import context

# if there is no path, you need to set to empty list
Points = namedtuple("Points", ["x", "y", "z"])


def nptype_to_prototype(np_value):
    """
    Transform the np type to proto type.

    Args:
        np_value (Type): Numpy data type.

    Returns:
        Type, proto data type.
    """
    np2pt_tbl = {
        np.bool_: 'DT_BOOL',
        np.int8: 'DT_INT8',
        np.int16: 'DT_INT16',
        np.int32: 'DT_INT32',
        np.int64: 'DT_INT64',
        np.uint8: 'DT_UINT8',
        np.uint16: 'DT_UINT16',
        np.uint32: 'DT_UINT32',
        np.uint64: 'DT_UINT64',
        np.float16: 'DT_FLOAT16',
        np.float: 'DT_FLOAT64',
        np.float32: 'DT_FLOAT32',
        np.float64: 'DT_FLOAT64',
        None: 'DT_UNDEFINED'
    }
    if np_value is None:
        return None

    np_type = np_value.dtype.type
    proto = np2pt_tbl.get(np_type, None)
    if proto is None:
        raise TypeError("No match for proto data type.")

    return proto


def fill_array_to_tensor(np_value, summary_tensor):
    """
    Package the tensor summary.

    Args:
        np_value (Type): Summary data type.
        summary_tensor (Tensor): The tensor of summary.

    Returns:
        Summary, return tensor summary content.
    """
    # get tensor dtype
    tensor_dtype = nptype_to_prototype(np_value)
    summary_tensor.data_type = DataType.Value(tensor_dtype)

    # get the value list
    tensor_value_list = np_value.reshape(-1).tolist()
    summary_tensor.float_data.extend(tensor_value_list)

    # get the tensor dim
    for vector in np_value.shape:
        summary_tensor.dims.append(vector)

    return summary_tensor


def transfer_tensor_to_tuple(inputs):
    """
    If the input is a tensor, convert it to a tuple. If not, the output is unchanged.
    """
    if isinstance(inputs, Tensor):
        return (inputs,)

    return inputs


class Landscape:
    """Return loss landscape."""
    def __init__(self,
                 intervals,
                 decomposition,
                 landscape_points: Points,
                 convergence_point=None,
                 path_points=None):
        self.landscape_points = landscape_points
        self.decomposition = decomposition
        self.intervals = intervals
        self.num_samples = 2048
        self.convergence_point = convergence_point
        self.path_points = path_points
        self.unit = 'step'
        self.step_per_epoch = 1

    def set_convergence_point(self, convergence_point: Points):
        """Set the convergence point."""
        self.convergence_point = convergence_point

    def transform_to_loss_landscape_msg(self, landscape_data):
        """Transform to loss landscape_msg."""
        landscape_msg = LossLandscape()
        # only save one dim in x and y
        fill_array_to_tensor(landscape_data.landscape_points.x[0], landscape_msg.landscape.x)
        fill_array_to_tensor(landscape_data.landscape_points.y[:, 0], landscape_msg.landscape.y)
        fill_array_to_tensor(landscape_data.landscape_points.z, landscape_msg.landscape.z)

        if landscape_data.path_points:
            landscape_msg.loss_path.intervals.extend(landscape_data.intervals)
            fill_array_to_tensor(landscape_data.path_points.x, landscape_msg.loss_path.points.x)
            fill_array_to_tensor(landscape_data.path_points.y, landscape_msg.loss_path.points.y)
            fill_array_to_tensor(landscape_data.path_points.z, landscape_msg.loss_path.points.z)

        if landscape_data.convergence_point:
            fill_array_to_tensor(landscape_data.convergence_point.x, landscape_msg.convergence_point.x)
            fill_array_to_tensor(landscape_data.convergence_point.y, landscape_msg.convergence_point.y)
            fill_array_to_tensor(landscape_data.convergence_point.z, landscape_msg.convergence_point.z)

        landscape_msg.metadata.decomposition = landscape_data.decomposition
        landscape_msg.metadata.unit = self.unit
        landscape_msg.metadata.step_per_epoch = self.step_per_epoch

        return landscape_msg


class SummaryLandscape:
    """
    SummaryLandscape can help you to collect loss landscape information.
    It can create landscape in PCA direction or random direction by calculating loss.

    Note:
        1. SummaryLandscape only supports Linux systems.

    Args:
        summary_dir(str): The path of summary is used to save the model weight,
            metadata and other data required for create landscape.

    Examples:
        >>> from mindspore.train.callback import SummaryLandscape
        >>> import mindspore.nn as nn
        >>> from mindspore import Model
        >>> from mindspore.nn import Loss
        >>> if __name__ == '__main__':
        ...     def callback_fn():
        ...         # The detail of LeNet5 shown in model_zoo.official.cv.lenet.src.lenet.py
        ...         network = LeNet5(10)
        ...         net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
        ...         model = Model(network, net_loss, metrics={"Loss": Loss()})
        ...         mnist_dataset_dir = '/path/to/mnist_dataset_directory'
        ...         ds_eval = create_dataset(mnist_dataset_dir, 32)
        ...         return model, network, ds_eval
        ...
        ...     summary_landscape = SummaryLandscape('./summary/lenet_interval_1')
        ...     intervals = [1, 2, 3, 4, 5]
        ...     # parameters of collect_landscape can be modified or unchanged
        ...     summary_landscape.gen_landscapes_with_multi_process(callback_fn,
        ...                                                         collect_landscape={"landscape_size": 40,
        ...                                                                            "create_landscape":{"train":True,
        ...                                                                                               "result":True
        ...                                                                                                           },
        ...                                                                            "num_samples": 2048,
        ...                                                                            "intervals": [interval_1
        ...                                                                                          ]},
        ...                                                         device_ids=[0, 1],
        ...                                                         device_target="GPU")
    """
    def __init__(self, summary_dir, intervals=None):
        self._summary_dir = os.path.realpath(summary_dir)
        self._ckpt_dir = os.path.join(self._summary_dir, 'ckpt_dir')
        _make_directory(self._ckpt_dir)

        # save the model params file, key is epoch, value is the ckpt file path
        self._model_params_file_map = {}

        # save the loss, key is epoch, value is loss
        self._loss_map = {}
        self._epoch_group = defaultdict(list)
        if intervals:
            self._create_epoch_group(intervals)
        self._max_epoch_group = 1

    def _create_epoch_group(self, intervals):
        """Create epoch group."""
        for i, interval in enumerate(intervals):
            for j in interval:
                self._epoch_group[i].append(j)

    def save_loss_and_model_params(self, cur_num, unit, backbone, loss):
        """Save model params and loss."""
        self._save_model_params(cur_num, unit, backbone, loss)

    def _save_model_params(self, cur_num, unit, backbone, loss):
        """Save model params and loss."""
        param_list = []

        for param in backbone.get_parameters():
            param.init_data()
            param_data = param.data if isinstance(param.data, Tensor) else Tensor(param.data)

            param_list.append(dict(
                name=param.name,
                data=param_data
            ))

        ckpt_file_name = f"{type(backbone).__name__}_{cur_num}_{unit}.ckpt"
        file_path = os.path.join(self._ckpt_dir, ckpt_file_name)
        save_checkpoint(param_list, file_path)

        self._model_params_file_map[str(cur_num)] = file_path
        self._loss_map[str(cur_num)] = loss

    def _get_model_params(self, epochs):
        """Get the model params."""
        parameters = []
        for epoch in epochs:
            file_path = self._model_params_file_map[str(epoch)]
            parameters.append(load_checkpoint(file_path).values())
        return parameters

    def clean_ckpt(self):
        """Clean the checkpoint."""
        shutil.rmtree(self._ckpt_dir, ignore_errors=True)

    def save_metadata(self, step_per_epoch, unit, num_samples, landscape_size, create_landscape):
        """Save meta data to json file."""
        data = {
            "epoch_group": self._epoch_group,
            "model_params_file_map": self._model_params_file_map,
            "step_per_epoch": step_per_epoch,
            "unit": unit,
            "num_samples": num_samples,
            "landscape_size": landscape_size,
            "create_landscape": create_landscape,
            "loss_map": self._loss_map
        }
        with open(os.path.join(self._ckpt_dir, 'train_metadata.json'), 'w') as file:
            json.dump(data, file)

    def gen_landscapes_with_multi_process(self, callback_fn, collect_landscape=None,
                                          device_ids=None, device_target='Ascend', output=None):
        """
        Use the multi process to generate landscape.

        Args:
            callback_fn (python function): A python function object. User needs to write a function,
                callback_ fn, it has no input, and the return requirements are as follows.

                - mindspore.train.Model: User's model object.
                - mindspore.nn.Cell: User's network object.
                - mindspore.dataset: User's dataset object for create loss landscape.
            collect_landscape (Union[dict, None]): The meaning of the parameters
                when creating loss landscape is consistent with the fields
                with the same name in SummaryCollector. The purpose of setting here
                is to allow users to freely modify creating parameters.

                - landscape_size (int): Specify the image resolution of the generated loss landscape.
                  For example, if it is set to 128, the resolution of the landscape is 128 * 128.
                  The calculation time increases with the increase of resolution.
                  Default: 40. Optional values: between 3 and 256.
                - create_landscape (List[bool, bool]): Select how to create loss landscape.
                  Training process loss landscape(train) and Training result loss landscape(result).
                  Default: {"train": True, "result": True}. Optional: True/False.
                - num_samples (int): The size of the dataset used to create the loss landscape.
                  For example, in image dataset, You can set num_samples is 2048,
                  which means that 2048 images are used to create loss landscape.
                  Default: 2048.
                - intervals (List[List[int]): Specifies the interval
                  in which the loss landscape. For example: If the user wants to
                  crate loss landscape of two training processes, they are 1-5 epoch
                  and 6-10 epoch respectively. They can set [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]].
                  Note: Each interval have at least three epochs.
            device_ids (List(int)): Specifies which devices are used to create loss landscape.
                For example: [0, 1] refers to creating loss landscape with device 0 and device 1.
            device_target (str): Specifies the type of computing device.
                Default: Ascend. Optional: Ascend/GPU/CPU.
            output (str): Specifies the path to save the loss landscape.
                Default: None. The default save path is the same as the summary file.
        """

        output_path = os.path.realpath(output) if output is not None else self._summary_dir
        summary_record = SummaryRecord(output_path)
        check_value_type('device_target', device_target, str)
        self._check_device_ids(device_ids)
        if device_target not in ["Ascend", "GPU", "CPU"]:
            raise ValueError(f'Landscape device_target should be Ascend, GPU or CPU, but got {device_target}.')
        if collect_landscape is not None:
            self._check_collect_landscape_data(collect_landscape)
            json_path = os.path.join(self._ckpt_dir, 'train_metadata.json')
            if not os.path.exists(json_path):
                raise FileNotFoundError(f'json file path not exists.')
            with open(json_path, 'r') as file:
                data = json.load(file)
            for key, value in collect_landscape.items():
                if key in data.keys():
                    data[key] = value

            if "intervals" in collect_landscape.keys():
                epoch_group = defaultdict(list)
                for i, interval in enumerate(collect_landscape.get("intervals")):
                    for j in interval:
                        epoch_group[i].append(j)
                data["epoch_group"] = epoch_group

            with open(json_path, 'w') as file:
                json.dump(data, file)

        for interval, landscape in self.list_landscapes(callback_fn=callback_fn,
                                                        device_ids=device_ids,
                                                        device_target=device_target):
            summary_record.add_value(PluginEnum.LANDSCAPE.value, f'landscape_{str(interval)}', landscape)
            summary_record.record(0)
            summary_record.flush()
        summary_record.close()

    def list_landscapes(self, callback_fn, device_ids=None, device_target='Ascend'):
        """Create landscape with single device and list all landscape."""
        json_path = os.path.join(self._ckpt_dir, 'train_metadata.json')
        if not os.path.exists(json_path):
            raise FileNotFoundError(f'train_metadata json file path not exists,'
                                    f'please use summary_collector to collect information to create the json file')
        with open(json_path, 'r') as file:
            data = json.load(file)
        self._check_json_file_data(data)

        self._epoch_group = data['epoch_group']
        self._model_params_file_map = data['model_params_file_map']
        self._loss_map = data['loss_map']
        create_landscape = data['create_landscape']
        landscape_size = data['landscape_size']
        kwargs = dict(proz=0.2, landscape_size=landscape_size, device_ids=device_ids, callback_fn=callback_fn)

        count = len(device_ids)
        start = time.time()
        with ProcessPoolExecutor(max_workers=count) as executor:
            if count > 1:
                futures = []
                for device_id in device_ids:
                    future = executor.submit(self._set_context, device_id, device_target)
                    futures.append(future)
                wait(futures, return_when=ALL_COMPLETED)

            kwargs['executor'] = executor if count > 1 else None

            if create_landscape['train']:
                for i, epochs in enumerate(self._epoch_group.values()):
                    #Each epoch_group have at least three epochs.
                    if len(epochs) < 3:
                        logger.error(f"This group epochs(%s) length is less 3, will ignore." % (epochs))
                        continue
                    if create_landscape['result']:
                        msg = f"Start to create the {i+1}/{len(self._epoch_group)+1} landscapes," \
                              f"epochs is {epochs}, decomposition is PCA."
                    else:
                        msg = f"Start to create the {i+1}/{len(self._epoch_group)} landscapes," \
                              f"epochs is {epochs}, decomposition is PCA."
                    logger.info(msg)
                    kwargs['epochs'] = epochs
                    mid_time = time.time()
                    landscape_data = self._create_landscape_by_pca(**kwargs)
                    logger.info("Create landscape end, use time: %s s." % (round(time.time() - mid_time, 6)))
                    landscape_data.unit = data['unit']
                    landscape_data.step_per_epoch = data['step_per_epoch']
                    landscape_data.num_samples = data['num_samples']
                    landscape_msg = landscape_data.transform_to_loss_landscape_msg(landscape_data)
                    yield [epochs[0], epochs[-1]], landscape_msg

            if create_landscape['result']:
                final_epochs = [list(self._epoch_group.values())[-1][-1]]
                if create_landscape['train']:
                    msg = f"Start to create the {len(self._epoch_group)+1}/{len(self._epoch_group)+1} landscapes," \
                          f"epochs is {final_epochs}, decomposition is Random. "
                else:
                    msg = f"Start to create the {1}/{1} landscapes, " \
                          f"epochs is {final_epochs}, decomposition is Random."
                logger.info(msg)


                kwargs['epochs'] = final_epochs
                mid_time_2 = time.time()
                landscape_data = self._create_landscape_by_random(**kwargs)
                logger.info("Create landscape end, use time: %s s." % (round(time.time() - mid_time_2, 6)))
                landscape_data.unit = data['unit']
                landscape_data.step_per_epoch = data['step_per_epoch']
                landscape_data.num_samples = data['num_samples']
                landscape_msg = landscape_data.transform_to_loss_landscape_msg(landscape_data)
                yield final_epochs, landscape_msg
        logger.info("Total use time: %s s." % (round(time.time() - start, 6)))

    @staticmethod
    def _set_context(device_id, device_target):
        """Set context."""
        context.set_context(device_id=device_id, device_target=device_target)
        context.set_context(mode=context.GRAPH_MODE)

    def _create_landscape_by_pca(self, epochs, proz, landscape_size, device_ids=None, callback_fn=None, executor=None):
        """Create landscape by PCA."""
        multi_parameters = self._get_model_params(epochs)
        param_matrixs = []
        for parameters in multi_parameters:
            parlis = []
            for param in parameters:
                if ("weight" in param.name or "bias" in param.name) and ("moment" not in param.name):
                    data = param.data.asnumpy().copy()
                    parlis = np.concatenate((parlis, data), axis=None)
                else:
                    continue
            param_matrixs.append(parlis)
        param_matrixs = np.vstack(param_matrixs)
        param_matrixs = param_matrixs[:-1] - param_matrixs[-1]
        # Only 2 are needed, as we have to reduce high dimensions into 2D.And we reserve one for loss value.
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(param_matrixs.T)
        v_ori, w_ori = np.array(principal_components[:, 0]), np.array(principal_components[:, -1])
        final_params = list(multi_parameters[-1])

        # Reshape PCA directions(include dimensions of all parameters) into original shape of Model parameters
        v_ndarray = self._reshape_vector(v_ori, final_params)
        w_ndarray = self._reshape_vector(w_ori, final_params)

        # Reshape PCA directions(include dimensions of only weights) into original shape of Model parameters
        final_params_filtered = self._filter_weight_and_bias(final_params)
        v_ndarray_filtered = self._reshape_vector(v_ori, final_params_filtered)
        w_ndarray_filtered = self._reshape_vector(w_ori, final_params_filtered)

        v_ndarray, w_ndarray = self._normalize_vector(final_params, v_ndarray, w_ndarray)
        v_ndarray_filtered, w_ndarray_filtered = self._normalize_vector(final_params_filtered, v_ndarray_filtered,
                                                                        w_ndarray_filtered)
        # Flat to a single vector and calc alpha, beta
        v_param = self._flat_ndarray(v_ndarray_filtered)
        w_param = self._flat_ndarray(w_ndarray_filtered)
        final_params_numpy = [param.data.asnumpy().copy() for param in final_params]
        final_params_filtered_numpy = [param.data.asnumpy().copy() for param in final_params_filtered]
        coefs = self._calc_coefs(multi_parameters, final_params_filtered_numpy, v_param, w_param)

        # generate coordinates of loss landscape
        coefs_x = coefs[:, 0][np.newaxis]
        coefs_y = coefs[:, 1][np.newaxis]

        boundaries_x = max(coefs_x[0]) - min(coefs_x[0])
        boundaries_y = max(coefs_y[0]) - min(coefs_y[0])

        x_axis = np.linspace(min(coefs_x[0]) - proz * boundaries_x, max(coefs_x[0]) +
                             proz * boundaries_x, landscape_size)
        y_axis = np.linspace(min(coefs_y[0]) - proz * boundaries_y, max(coefs_y[0]) +
                             proz * boundaries_y, landscape_size)
        x_points, y_points = np.meshgrid(x_axis, y_axis)

        test_final_params = dict()
        for param in final_params:
            test_final_params[param.name] = param.data.asnumpy().copy()

        if executor is not None:
            y_points_parts = []
            count_per_parts = len(y_points) // len(device_ids)
            start = 0
            logger.info("Use multi process, device_id: %s." % (device_ids))
            for i in range(len(device_ids)):
                if i != len(device_ids) - 1:
                    y_points_parts.append(y_points[start:start + count_per_parts])
                    start = start + count_per_parts
                else:
                    y_points_parts.append(y_points[start:])

            futures = []
            for i, _ in enumerate(device_ids):
                future = executor.submit(self._cont_loss_wrapper, callback_fn, test_final_params, final_params_numpy,
                                         v_ndarray, w_ndarray, x_points, y_points_parts[i])
                futures.append(future)
            wait(futures, return_when=ALL_COMPLETED)

            z_points = []
            for future in futures:
                z_points += future.result()
        else:
            z_points = self._cont_loss_wrapper(callback_fn, test_final_params, final_params_numpy, v_ndarray, w_ndarray,
                                               x_points, y_points)
        paths = []
        for epoch in epochs:
            paths.append(self._loss_map[str(epoch)])

        paths = np.array(paths)
        landscape_points = Points(x_points, y_points, np.vstack(z_points))
        path_points = Points(coefs_x[0], coefs_y[0], paths)
        zero_index = int(np.argwhere(path_points.x == 0))
        convergence_point = Points(np.array([0]), np.array([0]), np.array([path_points.z[zero_index]]))
        landscape = Landscape(intervals=epochs, decomposition='PCA', landscape_points=landscape_points,
                              path_points=path_points, convergence_point=convergence_point)
        return landscape

    def _cont_loss_wrapper(self, callback_fn, test_final_params, final_params_numpy,
                           v_ndarray, w_ndarray, x_points, y_points):
        """Compute loss wrapper."""
        model, network, valid_dataset, metrics = callback_fn()
        with open(os.path.join(self._ckpt_dir, 'train_metadata.json'), 'r') as file:
            data = json.load(file)
        self._check_json_file_data(data)
        num_samples = data['num_samples']
        batch_size = valid_dataset.get_batch_size()
        num_batches = num_samples // batch_size
        valid_dataset = valid_dataset.take(num_batches)

        final_params = []
        for (key, value) in test_final_params.items():
            parameter = Parameter(Tensor(value), name=key, requires_grad=True)
            final_params.append(parameter)

        # Start to calc loss landscape
        z_points = list()

        # Compute loss landscape
        for i, _ in enumerate(y_points):
            logger.info("Compute landscape loss value: %s/%s." % (i+1, len(y_points)))
            vals = self._cont_loss(valid_dataset, network, model, metrics, final_params,
                                   final_params_numpy, x_points[i], y_points[i][0],
                                   v_ndarray, w_ndarray)
            z_points.append(vals)

        return z_points

    def _create_landscape_by_random(self, epochs, proz, landscape_size, device_ids=None,
                                    callback_fn=None, executor=None):
        """Create landscape by Random."""
        multi_parameters = self._get_model_params(epochs)
        final_params = list(multi_parameters[-1])
        final_params_numpy = [param.data.asnumpy().copy() for param in final_params]
        total_params = sum(np.size(p) for p in final_params_numpy)
        v_rand = np.random.normal(size=total_params)
        w_rand = np.random.normal(size=total_params)

        # Reshape Random directions(include dimensions of all parameters) into original shape of Model parameters
        v_ndarray = self._reshape_random_vector(v_rand, final_params_numpy)
        w_ndarray = self._reshape_random_vector(w_rand, final_params_numpy)
        v_ndarray, w_ndarray = self._normalize_vector(final_params, v_ndarray, w_ndarray)

        boundaries_x, boundaries_y = 5, 5
        x_axis = np.linspace(-proz * boundaries_x, proz * boundaries_x, landscape_size)
        y_axis = np.linspace(-proz * boundaries_y, proz * boundaries_y, landscape_size)
        x_points, y_points = np.meshgrid(x_axis, y_axis)
        test_final_params = dict()
        for param in final_params:
            test_final_params[param.name] = param.data.asnumpy().copy()
        if executor is not None:
            logger.info("Use multi process, device_id: %s." % (device_ids))
            y_points_parts = []
            count_per_parts = len(y_points) // len(device_ids)
            start = 0
            for i in range(len(device_ids)):
                if i != len(device_ids) - 1:
                    y_points_parts.append(y_points[start:start + count_per_parts])
                    start = start + count_per_parts
                else:
                    y_points_parts.append(y_points[start:])

            futures = []
            for i in range(len(device_ids)):
                future = executor.submit(self._cont_loss_wrapper, callback_fn, test_final_params, final_params_numpy,
                                         v_ndarray, w_ndarray, x_points, y_points_parts[i])
                futures.append(future)
            wait(futures, return_when=ALL_COMPLETED)
            z_points = []
            for future in futures:
                z_points += future.result()
        else:
            z_points = self._cont_loss_wrapper(callback_fn, test_final_params, final_params_numpy, v_ndarray, w_ndarray,
                                               x_points, y_points)

        landscape_points = Points(x_points, y_points, np.vstack(z_points))
        convergence_point = Points(np.array([x_axis[len(x_axis)//2]]), np.array([y_axis[len(y_axis)//2]]),
                                   np.array([z_points[len(x_axis)//2][len(y_axis)//2]]))
        landscape = Landscape(intervals=epochs, decomposition='Random', landscape_points=landscape_points,
                              convergence_point=convergence_point)
        return landscape

    @staticmethod
    def _filter_weight_and_bias(parameters):
        """Filter the weight and bias of parameters."""

        filter_params = []
        for param in parameters:
            if ('weight' not in param.name and 'bias' not in param.name) or ('moment' in param.name):
                continue
            filter_params.append(param)
        return filter_params

    @staticmethod
    def _reshape_vector(vector, parameters):
        """Reshape vector into model shape."""
        ndarray = list()
        index = 0
        for param in parameters:
            data = param.data.asnumpy().copy()
            if ("weight" not in param.name and "bias" not in param.name) or ("moment" in param.name):
                ndarray.append(np.array(data, dtype=np.float32))
                continue

            vec_it = vector[index:(index + data.size)].reshape(data.shape)
            ndarray.append(np.array(vec_it, dtype=np.float32))
            index += data.size
        return ndarray

    @staticmethod
    def _reshape_random_vector(vector, params_numpy):
        """ Reshape random vector into model shape."""
        ndarray = list()
        index = 0
        for param in params_numpy:
            len_p = np.size(param)
            p_size = np.shape(param)
            vec_it = vector[index:(index + len_p)].reshape(p_size)
            ndarray.append(np.array(vec_it, dtype=np.float32))
            index += len_p
        return ndarray

    @staticmethod
    def _normalize_vector(parameters, get_v, get_w):
        """
        Normalizes the vectors spanning the 2D space, to make trajectories comparable between each other.
        """
        for i, param in enumerate(parameters):
            # Here as MindSpore ckpt has hyperparameters, we should skip them to make sure
            # PCA calculation is correct.
            data = param.data.asnumpy().copy()
            if ("weight" in param.name or "bias" in param.name) and ("moment" not in param.name):
                factor_v = np.linalg.norm(data) / np.linalg.norm(get_v[i])
                factor_w = np.linalg.norm(data) / np.linalg.norm(get_w[i])
                get_v[i] = get_v[i] * factor_v
                get_w[i] = get_w[i] * factor_w
            else:
                get_v[i] = get_v[i] * 0
                get_w[i] = get_w[i] * 0

        return get_v, get_w

    @staticmethod
    def _flat_ndarray(ndarray_vector):
        """Concatenates a python array of numpy arrays into a single, flat numpy array."""
        return np.concatenate([item.flatten() for item in ndarray_vector], axis=None)

    def _calc_coefs(self, parameter_group, final_param_ndarray, v_vector, w_vector):
        """
        Calculates the scale factors for plotting points
        in the 2D space spanned by the vectors v and w.
        """

        matris = [v_vector, w_vector]
        matris = np.vstack(matris)
        matris = matris.T

        pas = self._flat_ndarray(final_param_ndarray)
        coefs = list()
        for parameters in parameter_group:
            testi = list()
            for param in parameters:
                # Here as MindSpore ckpt has hyperparameters, we should skip them to make sure PCA calculation is correct
                if ('weight' not in param.name and 'bias' not in param.name) or ('moment' in param.name):
                    continue
                testi.append(param.data.asnumpy().copy())

            st_vec = self._flat_ndarray(testi)
            b_vec = st_vec - pas
            # Here using least square method to get solutions of a equation system to generate alpha and beta.
            coefs.append(np.hstack(np.linalg.lstsq(matris, b_vec, rcond=None)[0]))

        return np.array(coefs)

    def _cont_loss(self, ds_eval, network, model, metrics, parameters,
                   final_params_numpy, alph, beta, get_v, get_w):
        """
        Calculates the loss landscape based on vectors v and w (which can be principal components).
        Changes the internal state of model. Executes model.
        """
        logger.info("start to cont loss")
        vals = list()
        dataset_sink_mode = (context.get_context('device_target') == 'Ascend')

        al_item = 0
        for i, _ in enumerate(alph):
            # calculate new parameters for model

            parameters_dict = dict()
            for j, param in enumerate(parameters):
                parameters_dict[param.name] = self._change_parameter(j, param, final_params_numpy,
                                                                     alph[al_item], beta,
                                                                     get_v, get_w)

            al_item += 1
            # load parameters into model and calculate loss

            load_param_into_net(network, parameters_dict)
            del parameters_dict
            loss = self._loss_compute(model, ds_eval, metrics, dataset_sink_mode)
            logger.info("%s/%s loss: %s." % (i+1, len(alph), loss))
            vals = np.append(vals, loss['Loss'])

        return vals

    @staticmethod
    def _change_parameter(index, parameter, final_params_numpy, alpha, beta, get_v, get_w):
        """Function for changing parameter value with map and lambda."""
        data = final_params_numpy[index]
        data_target = data + alpha * get_v[index] + beta * get_w[index]
        data_target = Tensor(data_target.astype(np.float32))
        parameter.set_data(Tensor(data_target))
        return parameter

    def _loss_compute(self, model, data, metrics, dataset_sink_mode=False):
        """Compute loss."""
        self._metric_fns = get_metrics(metrics)
        for metric in self._metric_fns.values():
            metric.clear()

        network = model.train_network
        dataset_helper = DatasetHelper(data, dataset_sink_mode)
        if dataset_sink_mode:
            network = connect_network_with_dataset(network, dataset_helper)

        network.set_train(True)
        network.phase = 'train'

        for inputs in dataset_helper:
            if not dataset_sink_mode:
                inputs = transfer_tensor_to_tuple(inputs)
            outputs = network(*inputs)
            self._update_metrics(outputs)

        metrics = self._get_metrics()
        return metrics

    def _update_metrics(self, outputs):
        """Update metrics local values."""
        if isinstance(outputs, Tensor):
            outputs = (outputs,)
        if not isinstance(outputs, tuple):
            raise ValueError(f"The argument 'outputs' should be tuple, but got {type(outputs)}. "
                             f"Modify 'output' to Tensor or tuple. ")

        for metric in self._metric_fns.values():
            metric.update(outputs[0])

    def _get_metrics(self):
        """Get metrics local values."""
        metrics = dict()
        for key, value in self._metric_fns.items():
            metrics[key] = value.eval()
        return metrics

    @staticmethod
    def _check_landscape_size(landscape_size):
        """Check landscape size type and value."""
        check_value_type('landscape_size', landscape_size, int)
        # landscape size should be between 3 and 256.
        if landscape_size < 3 or landscape_size > 256:
            raise ValueError(f'Landscape size should be between 3 and 256, but got the: {landscape_size}')

    @staticmethod
    def _check_unit(unit):
        """Check unit type and value."""
        check_value_type('unit', unit, str)
        if "step" not in unit and "epoch" not in unit:
            raise ValueError(f'Unit should be step or epoch, but got the: {unit}')

    @staticmethod
    def _check_create_landscape(create_landscape):
        """Check create landscape type and value."""
        check_value_type('create_landscape', create_landscape, dict)
        for param, value in create_landscape.items():
            if param not in ["train", "result"]:
                raise ValueError(f'The key to create landscape should be in ["train", "result"], '
                                 f'but got the: {param}')
            if len(create_landscape) < 2:
                raise ValueError(f'The key to create landscape should be train and result, '
                                 f'but only got the: {param}')
            check_value_type(param, value, bool)

    @staticmethod
    def _check_intervals(intervals):
        """Check intervals type and value."""
        check_value_type('intervals', intervals, list)
        for _, interval in enumerate(intervals):
            check_value_type('each interval in intervals', interval, list)
            #Each interval have at least three epochs.
            if len(interval) < 3:
                raise ValueError(f'Each landscape interval should not be less than three, '
                                 f'but got the: {interval}.')
            for j in interval:
                if not isinstance(j, int):
                    raise TypeError(f'Landscape interval value type should be int, '
                                    f'but got the: {type(j)}.')

    @staticmethod
    def _check_device_ids(device_ids):
        """Check device_ids type and value."""
        check_value_type('device_ids', device_ids, list)
        for i in device_ids:
            if not isinstance(i, int):
                raise TypeError(f'Landscape device_ids type should be int, '
                                f'but got the: {type(i)}.')
            #device_id should be between 0 and 7.
            if i < 0 or i > 7:
                raise ValueError(f'Landscape device_ids value should be between 0 and 7,bu got {i}.')


    def _check_collect_landscape_data(self, collect_landscape):
        """Check collect landscape data type and value."""
        for param in collect_landscape.keys():
            if param not in ["landscape_size", "unit", "num_samples", "create_landscape", "intervals"]:
                raise ValueError(f'The key of collect landscape should be landscape_size, unit, num_samples'
                                 f'create_landscape or intervals, but got the: {param}. ')
        if "landscape_size" in collect_landscape:
            landscape_size = collect_landscape.get("landscape_size")
            self._check_landscape_size(landscape_size)
        if "unit" in collect_landscape:
            unit = collect_landscape.get("unit")
            self._check_unit(unit)
        if "num_samples" in collect_landscape:
            num_samples = collect_landscape.get("num_samples")
            check_value_type("num_samples", num_samples, int)
        if "create_landscape" in collect_landscape:
            create_landscape = collect_landscape.get("create_landscape")
            self._check_create_landscape(create_landscape)
        if "intervals" in collect_landscape:
            intervals = collect_landscape.get("intervals")
            self._check_intervals(intervals)

    def _check_json_file_data(self, json_file_data):
        """Check json file data."""
        file_key = ["epoch_group", "model_params_file_map", "step_per_epoch", "unit",
                    "num_samples", "landscape_size", "create_landscape", "loss_map"]
        for key in json_file_data.keys():
            if key not in file_key:
                raise ValueError(f'"train_metadata" json file should be {file_key}, but got the: {key}')
        epoch_group = json_file_data["epoch_group"]
        model_params_file_map = json_file_data["model_params_file_map"]
        step_per_epoch = json_file_data["step_per_epoch"]
        loss_map = json_file_data["loss_map"]
        unit = json_file_data["unit"]
        num_samples = json_file_data["num_samples"]
        landscape_size = json_file_data["landscape_size"]
        create_landscape = json_file_data["create_landscape"]

        for _, epochs in enumerate(epoch_group.values()):
            # Each epoch_group have at least three epochs.
            if len(epochs) < 3:
                raise ValueError(f'This group epochs length should not be less than 3'
                                 f'but got: {len(epochs)}. ')
            for epoch in epochs:
                if str(epoch) not in model_params_file_map.keys():
                    raise ValueError(f'The model_params_file_map does not exist {epoch}th epoch.')
                if str(epoch) not in loss_map.keys():
                    raise ValueError(f'The loss_map does not exist {epoch}th epoch.')

        check_value_type('step_per_epoch', step_per_epoch, int)
        self._check_landscape_size(landscape_size)
        self._check_unit(unit)
        check_value_type("num_samples", num_samples, int)
        self._check_create_landscape(create_landscape)
