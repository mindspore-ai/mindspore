# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
from __future__ import absolute_import

import os
import time
import json
import stat
import shutil
import numbers

from collections import defaultdict, namedtuple
from concurrent.futures import wait, ALL_COMPLETED, ProcessPoolExecutor

import numpy as np
from scipy import linalg, sparse

from mindspore import log as logger
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.summary_pb2 import LossLandscape
from mindspore.train.summary import SummaryRecord
from mindspore.train.summary.enums import PluginEnum
from mindspore.train.anf_ir_pb2 import DataType
from mindspore.train._utils import check_value_type, _make_directory
from mindspore.train.dataset_helper import DatasetHelper
from mindspore.train.metrics import get_metrics
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
        1. When using SummaryLandscape, you need to run the code in `if __name__ == "__main__"` .
        2. SummaryLandscape only supports Linux systems.

    Args:
        summary_dir (str): The path of summary is used to save the model weight,
            metadata and other data required to create landscape.

    Examples:
        >>> import mindspore as ms
        >>> import mindspore.nn as nn
        >>> from mindspore.nn import Loss, Accuracy
        >>> from mindspore.train import Model, SummaryCollector, SummaryLandscape
        >>>
        >>> if __name__ == '__main__':
        ...     # If the device_target is Ascend, set the device_target to "Ascend"
        ...     ms.set_context(mode=ms.GRAPH_MODE, device_target="GPU")
        ...     mnist_dataset_dir = '/path/to/mnist_dataset_directory'
        ...     # The detail of create_dataset method shown in model_zoo.official.cv.lenet.src.dataset.py
        ...     ds_train = create_dataset(mnist_dataset_dir, 32)
        ...     # The detail of LeNet5 shown in model_zoo.official.cv.lenet.src.lenet.py
        ...     network = LeNet5(10)
        ...     net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
        ...     net_opt = nn.Momentum(network.trainable_params(), 0.01, 0.9)
        ...     model = Model(network, net_loss, net_opt, metrics={"Accuracy": Accuracy()})
        ...     # Simple usage for collect landscape information:
        ...     interval_1 = [1, 2, 3, 4, 5]
        ...     summary_collector = SummaryCollector(summary_dir='./summary/lenet_interval_1',
        ...                                          collect_specified_data={'collect_landscape':{"landscape_size": 4,
        ...                                                                                        "unit": "step",
        ...                                                                          "create_landscape":{"train":True,
        ...                                                                                             "result":False},
        ...                                                                          "num_samples": 2048,
        ...                                                                          "intervals": [interval_1]}
        ...                                                                    })
        ...     model.train(1, ds_train, callbacks=[summary_collector], dataset_sink_mode=False)
        ...
        ...     # Simple usage for visualization landscape:
        ...     def callback_fn():
        ...         network = LeNet5(10)
        ...         net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
        ...         metrics = {"Loss": Loss()}
        ...         model = Model(network, net_loss, metrics=metrics)
        ...         mnist_dataset_dir = '/path/to/mnist_dataset_directory'
        ...         ds_eval = create_dataset(mnist_dataset_dir, 32)
        ...         return model, network, ds_eval, metrics
        ...
        ...     summary_landscape = SummaryLandscape('./summary/lenet_interval_1')
        ...     # parameters of collect_landscape can be modified or unchanged
        ...     summary_landscape.gen_landscapes_with_multi_process(callback_fn,
        ...                                                        collect_landscape={"landscape_size": 4,
        ...                                                                         "create_landscape":{"train":False,
        ...                                                                                            "result":False},
        ...                                                                          "num_samples": 2048,
        ...                                                                          "intervals": [interval_1]},
        ...                                                         device_ids=[1])
    """
    def __init__(self, summary_dir):
        self._summary_dir = os.path.realpath(summary_dir)
        self._ckpt_dir = os.path.join(self._summary_dir, 'ckpt_dir')
        _make_directory(self._ckpt_dir)

        # save the model params file, key is epoch, value is the ckpt file path
        self._model_params_file_map = {}
        self._epoch_group = defaultdict(list)
        self._metric_fns = None

    def _get_model_params(self, epochs):
        """Get the model params."""
        parameters = []
        for epoch in epochs:
            file_path = self._model_params_file_map.get(str(epoch))
            parameters.append(list(load_checkpoint(file_path).values()))
        return parameters

    def _create_epoch_group(self, intervals):
        for i, interval in enumerate(intervals):
            for j in interval:
                self._epoch_group[i].append(j)

    def clean_ckpt(self):
        """Clean the checkpoint."""
        shutil.rmtree(self._ckpt_dir, ignore_errors=True)

    def gen_landscapes_with_multi_process(self, callback_fn, collect_landscape=None,
                                          device_ids=None, output=None):
        """
        Use the multi process to generate landscape.

        Args:
            callback_fn (python function): A python function object. User needs to write a function,
                it has no input, and the return requirements are as follows.

                - mindspore.train.Model: User's model object.
                - mindspore.nn.Cell: User's network object.
                - mindspore.dataset: User's dataset object for create loss landscape.
                - mindspore.train.Metrics: User's metrics object.
            collect_landscape (Union[dict, None]): The meaning of the parameters
                when creating loss landscape is consistent with the fields
                with the same name in SummaryCollector. The purpose of setting here
                is to allow users to freely modify creating parameters. Default: None.

                - landscape_size (int): Specify the image resolution of the generated loss landscape.
                  For example, if it is set to 128, the resolution of the landscape is 128 * 128.
                  The calculation time increases with the increase of resolution.
                  Default: 40. Optional values: between 3 and 256.
                - create_landscape (dict): Select how to create loss landscape.
                  Training process loss landscape(train) and training result loss landscape(result).
                  Default: {"train": True, "result": True}. Optional: True/False.
                - num_samples (int): The size of the dataset used to create the loss landscape.
                  For example, in image dataset, You can set num_samples is 2048,
                  which means that 2048 images are used to create loss landscape.
                  Default: 2048.
                - intervals (List[List[int]]): Specifies the interval
                  in which the loss landscape. For example: If the user wants to
                  create loss landscape of two training processes, they are 1-5 epoch
                  and 6-10 epoch respectively. They can set [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]].
                  Note: Each interval have at least three epochs.
            device_ids (List(int)): Specifies which devices are used to create loss landscape.
                For example: [0, 1] refers to creating loss landscape with device 0 and device 1.
                Default: None.
            output (str): Specifies the path to save the loss landscape.
                Default: None. The default save path is the same as the summary file.
        """

        output_path = os.path.realpath(output) if output is not None else self._summary_dir
        summary_record = SummaryRecord(output_path)
        self._check_device_ids(device_ids)
        if collect_landscape is not None:
            try:
                self._check_collect_landscape_data(collect_landscape)
            except (ValueError, TypeError) as err:
                summary_record.close()
                raise err
            json_path = os.path.join(self._ckpt_dir, 'train_metadata.json')
            if not os.path.exists(json_path):
                summary_record.close()
                raise FileNotFoundError(f'For "{self.__class__.__name__}", '
                                        f'train_metadata.json file path of {json_path} not exists.')
            with open(json_path, 'r') as file:
                data = json.load(file)
            for key, value in collect_landscape.items():
                if key in data.keys():
                    data[key] = value

            if "intervals" in collect_landscape.keys():
                self._create_epoch_group(collect_landscape.get("intervals"))
                data["epoch_group"] = self._epoch_group
            with open(json_path, 'w') as file:
                json.dump(data, file)
            os.chmod(json_path, stat.S_IRUSR)

        for interval, landscape in self._list_landscapes(callback_fn=callback_fn, device_ids=device_ids):
            summary_record.add_value(PluginEnum.LANDSCAPE.value, f'landscape_{str(interval)}', landscape)
            summary_record.record(0)
            summary_record.flush()
        summary_record.close()

    def _list_landscapes(self, callback_fn, device_ids=None):
        """Create landscape with single device and list all landscape."""

        if not os.path.exists(os.path.join(self._ckpt_dir, 'train_metadata.json')):
            raise FileNotFoundError(f'For "{self.__class__.__name__}", train_metadata.json file does not exist '
                                    f'under the path, please use summary_collector to collect information to '
                                    f'create the json file')
        with open(os.path.join(self._ckpt_dir, 'train_metadata.json'), 'r') as file:
            data = json.load(file)
        self._check_json_file_data(data)

        self._epoch_group = data['epoch_group']
        self._model_params_file_map = data['model_params_file_map']
        kwargs = dict(proz=0.2, landscape_size=data['landscape_size'], device_ids=device_ids, callback_fn=callback_fn)

        start = time.time()
        with ProcessPoolExecutor(max_workers=len(device_ids)) as executor:
            if len(device_ids) > 1:
                futures = []
                for device_id in device_ids:
                    future = executor.submit(self._set_context, device_id)
                    futures.append(future)
                wait(futures, return_when=ALL_COMPLETED)

            kwargs['executor'] = executor if len(device_ids) > 1 else None

            if data['create_landscape']['train']:
                for i, epochs in enumerate(self._epoch_group.values()):
                    self._log_message(data['create_landscape'], index=i, interval=epochs)
                    kwargs['epochs'] = epochs
                    mid_time = time.time()
                    landscape_data = self._create_landscape_by_pca(**kwargs)
                    logger.info("Create landscape end, use time: %s s." % (round(time.time() - mid_time, 6)))
                    landscape_data.unit = data['unit']
                    landscape_data.step_per_epoch = data['step_per_epoch']
                    landscape_data.num_samples = data['num_samples']
                    yield [epochs[0], epochs[-1]], landscape_data.transform_to_loss_landscape_msg(landscape_data)

            if data['create_landscape']['result']:
                final_epochs = [list(self._epoch_group.values())[-1][-1]]
                self._log_message(data['create_landscape'], final_epochs=final_epochs)
                kwargs['epochs'] = final_epochs
                mid_time = time.time()
                landscape_data = self._create_landscape_by_random(**kwargs)
                logger.info("Create landscape end, use time: %s s." % (round(time.time() - mid_time, 6)))
                landscape_data.unit = data['unit']
                landscape_data.step_per_epoch = data['step_per_epoch']
                landscape_data.num_samples = data['num_samples']
                yield final_epochs, landscape_data.transform_to_loss_landscape_msg(landscape_data)
        logger.info("Total use time: %s s." % (round(time.time() - start, 6)))

    def _log_message(self, create_landscape, index=None, interval=None, final_epochs=None):
        """Generate drawing information using log."""
        if final_epochs is None:
            if create_landscape['result']:
                msg = f"Start to create the {index + 1}/{len(self._epoch_group) + 1} landscapes, " \
                      f"checkpoint is {interval}, decomposition is PCA."
            else:
                msg = f"Start to create the {index + 1}/{len(self._epoch_group)} landscapes, " \
                      f"checkpoint is {interval}, decomposition is PCA."
        else:
            if create_landscape['train']:
                msg = f"Start to create the {len(self._epoch_group) + 1}/{len(self._epoch_group) + 1} landscapes, " \
                      f"checkpoint is {final_epochs}, decomposition is Random. "
            else:
                msg = f"Start to create the {1}/{1} landscapes, " \
                      f"checkpoint is {final_epochs}, decomposition is Random."
        logger.info(msg)

    @staticmethod
    def _set_context(device_id):
        """Set context."""
        context.set_context(device_id=device_id)
        context.set_context(mode=context.GRAPH_MODE)

    def _create_landscape_by_pca(self, epochs, proz, landscape_size, device_ids=None, callback_fn=None, executor=None):
        """Create landscape by PCA."""
        multi_parameters = self._get_model_params(epochs)
        param_matrixs = []
        for parameters in multi_parameters:
            parlis = []
            for param in parameters:
                if ("weight" in param.name or "bias" in param.name) and ("moment" not in param.name):
                    data = param.data.asnumpy()
                    parlis = np.concatenate((parlis, data), axis=None)
                else:
                    continue
            param_matrixs.append(parlis)
        param_matrixs = np.vstack(param_matrixs)
        param_matrixs = param_matrixs[:-1] - param_matrixs[-1]
        # Only 2 are needed, as we have to reduce high dimensions into 2D.And we reserve one for loss value.
        pca = _PCA(n_comps=2)
        principal_components = pca.compute(param_matrixs.T)
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
        final_params_numpy = [param.data.asnumpy() for param in final_params]
        final_params_filtered_numpy = [param.data.asnumpy() for param in final_params_filtered]
        coefs = self._calc_coefs(multi_parameters, final_params_filtered_numpy, v_param, w_param)

        # generate coordinates of loss landscape
        coefs_x = coefs[:, 0][np.newaxis]
        coefs_y = coefs[:, 1][np.newaxis]

        x_axis = np.linspace(min(coefs_x[0]) - proz * (max(coefs_x[0]) - min(coefs_x[0])),
                             max(coefs_x[0]) + proz * (max(coefs_x[0]) - min(coefs_x[0])), landscape_size)
        y_axis = np.linspace(min(coefs_y[0]) - proz * (max(coefs_y[0]) - min(coefs_y[0])),
                             max(coefs_y[0]) + proz * (max(coefs_y[0]) - min(coefs_y[0])), landscape_size)
        x_points, y_points = np.meshgrid(x_axis, y_axis)

        test_final_params = dict()
        for param in final_params:
            test_final_params[param.name] = param.data.asnumpy()

        if executor is not None:
            coefs_parts, y_points_parts = [], []
            count_per_parts = len(coefs) // len(device_ids)
            start = 0
            for i in range(len(device_ids)):
                if i != len(device_ids) - 1:
                    coefs_parts.append(coefs[start:start + count_per_parts])
                    start = start + count_per_parts
                else:
                    coefs_parts.append(coefs[start:])
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
                                         v_ndarray, w_ndarray, x_points, y_points_parts[i], coefs=coefs_parts[i])
                futures.append(future)
            wait(futures, return_when=ALL_COMPLETED)

            z_points, paths = [], []
            for future in futures:
                paths += future.result()[0]
                z_points += future.result()[1]
        else:
            paths, z_points = self._cont_loss_wrapper(callback_fn, test_final_params, final_params_numpy,
                                                      v_ndarray, w_ndarray, x_points, y_points, coefs=coefs)

        paths = np.array(paths)
        landscape_points = Points(x_points, y_points, np.vstack(z_points))
        path_points = Points(coefs_x[0], coefs_y[0], paths.T[0])
        zero_index = int(np.argwhere(path_points.x == 0))
        convergence_point = Points(np.array([0]), np.array([0]), np.array([path_points.z[zero_index]]))
        landscape = Landscape(intervals=epochs, decomposition='PCA', landscape_points=landscape_points,
                              path_points=path_points, convergence_point=convergence_point)
        return landscape

    def _cont_loss_wrapper(self, callback_fn, test_final_params, final_params_numpy,
                           v_ndarray, w_ndarray, x_points, y_points, coefs=None):
        """Compute loss wrapper."""
        model, network, valid_dataset, metrics = callback_fn()
        with open(os.path.join(self._ckpt_dir, 'train_metadata.json'), 'r') as file:
            data = json.load(file)
        self._check_json_file_data(data)
        num_samples = data['num_samples']
        batch_size = valid_dataset.get_batch_size()
        num_batches = num_samples // batch_size
        valid_dataset = valid_dataset.take(num_batches)

        paths, final_params = [], []
        for (key, value) in test_final_params.items():
            parameter = Parameter(Tensor(value), name=key, requires_grad=True)
            final_params.append(parameter)
        if coefs is not None:
            for i, coef in enumerate(coefs):
                loss_data = self._cont_loss(valid_dataset, network, model, metrics, final_params,
                                            final_params_numpy, [coef[0]], coef[1], v_ndarray, w_ndarray, path=True)
                paths.append(loss_data)
                print("Drawing landscape path total progress is %s/%s, landscape path loss is %s."
                      % (i+1, len(coefs), loss_data[0]))
        # Start to calc loss landscape
        z_points = list()

        # Compute loss landscape
        for i, _ in enumerate(y_points):
            print("Drawing landscape total progress: %s/%s." % (i+1, len(y_points)))
            vals = self._cont_loss(valid_dataset, network, model, metrics, final_params,
                                   final_params_numpy, x_points[i], y_points[i][0],
                                   v_ndarray, w_ndarray)
            z_points.append(vals)

        return paths, z_points

    def _create_landscape_by_random(self, epochs, proz, landscape_size, device_ids=None,
                                    callback_fn=None, executor=None):
        """Create landscape by Random."""
        multi_parameters = self._get_model_params(epochs)
        final_params = list(multi_parameters[-1])
        final_params_numpy = [param.data.asnumpy() for param in final_params]
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
            test_final_params[param.name] = param.data.asnumpy()
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
                z_points += future.result()[1]
        else:
            _, z_points = self._cont_loss_wrapper(callback_fn, test_final_params, final_params_numpy,
                                                  v_ndarray, w_ndarray, x_points, y_points)

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
            data = param.data.asnumpy()
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
            data = param.data.asnumpy()
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
                # Here as MindSpore ckpt has hyperparameters,
                # we should skip them to make sure PCA calculation is correct
                if ('weight' not in param.name and 'bias' not in param.name) or ('moment' in param.name):
                    continue
                testi.append(param.data.asnumpy())

            st_vec = self._flat_ndarray(testi)
            b_vec = st_vec - pas
            # Here using least square method to get solutions of a equation system to generate alpha and beta.
            coefs.append(np.hstack(np.linalg.lstsq(matris, b_vec, rcond=None)[0]))

        return np.array(coefs)

    def _cont_loss(self, ds_eval, network, model, metrics, parameters,
                   final_params_numpy, alph, beta, get_v, get_w, path=False):
        """
        Calculates the loss landscape based on vectors v and w (which can be principal components).
        Changes the internal state of model. Executes model.
        """
        logger.info("start to cont loss")
        vals = list()

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
            loss = self._loss_compute(model, ds_eval, metrics)
            if path is False:
                print("Current local landscape progress is %s/%s, landscape loss is %s."
                      % (i+1, len(alph), loss.get('Loss')))
            vals = np.append(vals, loss.get('Loss'))

        return vals

    @staticmethod
    def _change_parameter(index, parameter, final_params_numpy, alpha, beta, get_v, get_w):
        """Function for changing parameter value with map and lambda."""
        data = final_params_numpy[index]
        data_target = data + alpha * get_v[index] + beta * get_w[index]
        data_target = Tensor(data_target.astype(np.float32))
        parameter.set_data(Tensor(data_target))
        return parameter

    def _loss_compute(self, model, data, metrics):
        """Compute loss."""
        dataset_sink_mode = False
        self._metric_fns = get_metrics(metrics)
        for metric in self._metric_fns.values():
            metric.clear()

        network = model.train_network
        dataset_helper = DatasetHelper(data, dataset_sink_mode)

        network.set_train(True)
        network.phase = 'train'

        for inputs in dataset_helper:
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

    def _check_unit(self, unit):
        """Check unit type and value."""
        check_value_type('unit', unit, str)
        if unit not in ["step", "epoch"]:
            raise ValueError(f'For "{self.__class__.__name__}", the "unit" in train_metadata.json should be '
                             f'step or epoch, but got the: {unit}')

    def _check_landscape_size(self, landscape_size):
        """Check landscape size type and value."""
        check_value_type('landscape_size', landscape_size, int)
        # landscape size should be between 3 and 256.
        if landscape_size < 3 or landscape_size > 256:
            raise ValueError(f'For "{self.__class__.__name__}", "landscape_size" in train_metadata.json should be '
                             f'between 3 and 256, but got the: {landscape_size}')

    def _check_create_landscape(self, create_landscape):
        """Check create landscape type and value."""
        check_value_type('create_landscape', create_landscape, dict)
        for param, value in create_landscape.items():
            if param not in ["train", "result"]:
                raise ValueError(f'For "{self.__class__.__name__}", the key of "create_landscape" should be in '
                                 f'["train", "result"], but got the: {param}.')
            if len(create_landscape) < 2:
                raise ValueError(f'For "{self.__class__.__name__}", the key of "create_landscape" should be train '
                                 f'and result, but only got the: {param}')
            check_value_type(param, value, bool)

    def _check_intervals(self, intervals):
        """Check intervals type and value."""
        check_value_type('intervals', intervals, list)
        for _, interval in enumerate(intervals):
            check_value_type('each interval in intervals', interval, list)
            #Each interval have at least three epochs.
            if len(interval) < 3:
                raise ValueError(f'For "{self.__class__.__name__}", the length of each list in "intervals" '
                                 f'should not be less than three, but got the: {interval}.')
            for j in interval:
                if not isinstance(j, int):
                    raise TypeError(f'For "{self.__class__.__name__}", the type of each value in "intervals" '
                                    f'should be int, but got the: {type(j)}.')

    def _check_device_ids(self, device_ids):
        """Check device_ids type and value."""
        check_value_type('device_ids', device_ids, list)
        for i in device_ids:
            if not isinstance(i, int):
                raise TypeError(f'For "{self.__class__.__name__}.gen_landscapes_with_multi_process", the parameter '
                                f'"device_ids" type should be int, but got the: {type(i)}.')
            #device_id should be between 0 and 7.
            if i < 0 or i > 7:
                raise ValueError(f'For "{self.__class__.__name__}.gen_landscapes_with_multi_process", the parameter '
                                 f'"device_ids" should be between 0 and 7,but got {i}.')

    def _check_collect_landscape_data(self, collect_landscape):
        """Check collect landscape data type and value."""
        for param in collect_landscape.keys():
            if param not in ["landscape_size", "unit", "num_samples", "create_landscape", "intervals"]:
                raise ValueError(f'For "{self.__class__.__name__}", the key of collect landscape should be '
                                 f'landscape_size, unit, num_samples create_landscape or intervals, '
                                 f'but got the: {param}. ')
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
                    "num_samples", "landscape_size", "create_landscape"]
        for key in json_file_data.keys():
            if key not in file_key:
                raise ValueError(f'"train_metadata" json file should be {file_key}, but got the: {key}')
        epoch_group = json_file_data["epoch_group"]
        model_params_file_map = json_file_data["model_params_file_map"]
        step_per_epoch = json_file_data["step_per_epoch"]
        unit = json_file_data["unit"]
        num_samples = json_file_data["num_samples"]
        landscape_size = json_file_data["landscape_size"]
        create_landscape = json_file_data["create_landscape"]

        for _, epochs in enumerate(epoch_group.values()):
            # Each epoch_group have at least three epochs.
            if len(epochs) < 3:
                raise ValueError(f'For "{self.__class__.__name__}", the "epoch_group" in train_metadata.json, '
                                 f'length of each list in "epoch_group" should not be less than 3, '
                                 f'but got: {len(epochs)}. ')
            for epoch in epochs:
                if str(epoch) not in model_params_file_map.keys():
                    raise ValueError(f'For "{self.__class__.__name__}", the "model_params_file_map" in '
                                     f'train_metadata.json does not exist {epoch}th checkpoint in intervals.')

        check_value_type('step_per_epoch', step_per_epoch, int)
        self._check_landscape_size(landscape_size)
        self._check_unit(unit)
        check_value_type("num_samples", num_samples, int)
        self._check_create_landscape(create_landscape)


class _PCA:
    r"""
    The internal class for computing PCA vectors.

    .. math::

        u, s, vt = svd(x - mean(x)),
        u_i = u_i * s_i,

    where :math:`mean` is the mean operator, :math:`svd` is the singular value decomposition operator.
    :math:`u_i` is line :math:`i` of the :math:`u`, :math:`s_i` is column :math:`i` of the :math:`s`,
    :math:`i` ranges from :math:`0` to :math:`n\_comps`.

    Args:
        n_comps (int): Number of principal components needed.
    """
    def __init__(self, n_comps):
        self._n_comps = n_comps
        self._random_status = None
        self._iterated_power = "auto"
        self._n_oversamples = 10

    @staticmethod
    def _safe_dot(a, b):
        """Dot product that handle the matrix case correctly."""
        if a.ndim > 2 or b.ndim > 2:
            if sparse.issparse(b):
                # Sparse is always 2 dimensional. Implies a is above 3 dimensional.
                # [n, ..., o, p] @ [l, m] -> [n, ..., o, m]
                a_2d = a.reshape(-1, a.shape[-1])
                ret = a_2d @ b
                ret = ret.reshape(*a.shape[:-1], b.shape[1])
            elif sparse.issparse(a):
                # Sparse is always 2 dimensional. Implies b is above 3 dimensional.
                # [l, m] @ [n, ..., o, p, q] -> [l, n, ..., o, q]
                b_ = np.rollaxis(b, -2)
                b_2d = b_.reshape((b.shape[-2], -1))
                ret = a @ b_2d
                ret = ret.reshape(a.shape[0], *b_.shape[1:])
            else:
                ret = np.dot(a, b)

        else:
            ret = a @ b

        return ret

    @staticmethod
    def _svd_turn(u, v, u_decision=True):
        """Confirm correction to ensure deterministic output from SVD."""
        if u_decision:
            # rows of v, columns of u
            max_cols = np.argmax(np.abs(u), axis=0)
            signs = np.sign(u[max_cols, list(range(u.shape[1]))])
            v *= signs[:, np.newaxis]
            u *= signs
        else:
            # rows of u, columns of v
            max_rows = np.argmax(np.abs(v), axis=1)
            signs = np.sign(v[list(range(v.shape[0])), max_rows])
            v *= signs[:, np.newaxis]
            u *= signs
        return u, v

    @staticmethod
    def _check_random_status(seed):
        """Transform seed into a np.random.RandomState instance."""
        if isinstance(seed, np.random.RandomState):
            return seed
        if seed is None or seed is np.random:
            return np.random.RandomState()
        if isinstance(seed, numbers.Integral):
            return np.random.RandomState(seed)
        raise ValueError(
            "%r cannot be used to seed a numpy.random.RandomState instance" % seed
        )

    def compute(self, x):
        """Main method for computing principal components."""
        n_components = self._n_comps
        # small dimension (the shape is less than 500), and the full amount is calculated.
        if max(x.shape) <= 500:
            u, s, _ = self._fit_few(x)
        # When dimension of x is much, truncated SVD is used for calculation.
        elif 1 <= n_components < 0.8 * min(x.shape):
            u, s, _ = self._fit_much(x, n_components)
        #  A case of n_components in (0, 1)
        else:
            u, s, _ = self._fit_few(x)

        for i, _ in enumerate(s):
            # To prevent s from being equal to 0, a small fixed noise is added.
            # Adjust 1e-19 was found a good compromise for s.
            if s[i] == 0:
                s[i] = 1e-19
        u = u[:, :self._n_comps]
        u *= s[:self._n_comps]

        return u

    def _fit_few(self, x):
        """Compute principal components with full SVD on x, when dimension of x is few."""
        mean_ = np.mean(x, axis=0)
        x -= mean_
        u, s, vt = linalg.svd(x, full_matrices=False)
        u, vt = self._svd_turn(u, vt)

        return u, s, vt

    def _fit_much(self, x, n_components):
        """Compute principal components with truncated SVD on x, when dimension of x is much."""
        random_state = self._check_random_status(self._random_status)
        mean_ = np.mean(x, axis=0)
        x -= mean_
        u, s, vt = self._random_svd(x, n_components, n_oversamples=self._n_oversamples, random_state=random_state)
        return u, s, vt

    def _random_svd(self, m, n_components, n_oversamples=10, random_state="warn"):
        """Compute a truncated randomized SVD."""
        n_random = n_components + n_oversamples
        n_samples, n_features = m.shape
        # Adjust 7 or 4 was found a good compromise for randomized SVD.
        n_iter = 7 if n_components < 0.1 * min(m.shape) else 4
        if n_samples < n_features:
            m = m.T

        q = self._random_range_finder(m, size=n_random, n_iter=n_iter, random_state=random_state)
        # Project m to the low dimensional space using the basis vectors (q vector).
        b = self._safe_dot(q.T, m)
        # Compute the svd on this matrix (b matrix)
        uhat, s, vt = linalg.svd(b, full_matrices=False)

        del b
        u = np.dot(q, uhat)

        if n_samples < n_features:
            u, vt = self._svd_turn(u, vt, u_decision=False)
        else:
            u, vt = self._svd_turn(u, vt)

        if n_samples < n_features:
            return vt[:n_components, :].T, s[:n_components], u[:, :n_components].T

        return u[:, :n_components], s[:n_components], vt[:n_components, :]

    def _random_range_finder(self, a, size, n_iter, random_state=None):
        """Computes an orthonormal matrix whose range approximates the range of A."""
        random_state = self._check_random_status(random_state)
        # Generate normal random vectors.
        q = random_state.normal(size=(a.shape[1], size))
        if a.dtype.kind == "f":
            # Ensure f32 is retained as f32
            q = q.astype(a.dtype, copy=False)
        if n_iter <= 2:
            power_iteration_normalizer = "none"
        else:
            power_iteration_normalizer = "LU"
        # use power iterations with q to further compute the top singular vectors of a in q
        for _ in range(n_iter):
            if power_iteration_normalizer == "none":
                q = self._safe_dot(a, q)
                q = self._safe_dot(a.T, q)
            elif power_iteration_normalizer == "LU":
                q, _ = linalg.lu(self._safe_dot(a, q), permute_l=True)
                q, _ = linalg.lu(self._safe_dot(a.T, q), permute_l=True)
        # The orthogonal basis is extracted by the linear projection of Q, and the range of a is sampled.
        q, _ = linalg.qr(self._safe_dot(a, q), mode="economic")
        return q
