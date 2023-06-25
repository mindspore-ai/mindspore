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
"""
High-Level API for Second Order Training or Testing.
Second-order optimizer THOR reduces the computation workload and improves the computation speed by reducing the
frequency of updating the second-order matrix. In order to optimize the overall performance, the ModelThor class
is redefined to inherit the Model class provided by MindSpore. The parameter of THOR for controlling the frequency
of updating the second-order matrix can be obtained by the ModelThor class.
"""
from __future__ import absolute_import

import math

from mindspore.train.callback import RunContext
from mindspore import context
from mindspore import nn
from mindspore.train.model import Model
from mindspore.train.dataset_helper import connect_network_with_dataset
from mindspore.parallel._utils import _need_to_full, _to_full_tensor
from mindspore.common.dtype import pytype_to_dtype
from mindspore._c_expression import init_exec_dataset
from mindspore.train.train_thor.dataset_helper import DatasetHelper


def _convert_to_ms_type(types):
    """
    Convert from numpy type to mindspore tensor type.

    Args:
        types (list): Numpy type list of element in dataset.

    Returns:
        list, list of element in dataset.
    """
    ms_types = []
    for numpy_type in types:
        ms_type = pytype_to_dtype(numpy_type)
        ms_types.append(ms_type)
    return ms_types


def _get_types_and_shapes(dataset):
    """Get dataset types and shapes."""
    dataset_types = _convert_to_ms_type(dataset.output_types())
    dataset_shapes = dataset.output_shapes()
    return dataset_types, dataset_shapes


def _exec_datagraph(exec_dataset, dataset_size, phase='dataset'):
    """Initialize and execute the dataset graph."""
    batch_size = exec_dataset.get_batch_size()
    input_indexs = exec_dataset.input_indexs

    # transform data format
    dataset_types, dataset_shapes = _get_types_and_shapes(exec_dataset)
    init_exec_dataset(exec_dataset.__transfer_dataset__.queue_name,
                      dataset_size,
                      batch_size,
                      dataset_types,
                      dataset_shapes,
                      input_indexs,
                      phase=phase,
                      need_run=False)


class ModelThor(Model):
    """
    High-Level API for Training or Testing.

    `Model` groups layers into an object with training and inference features.

    Args:
        network (Cell): A training or testing network.
        loss_fn (Cell): Objective function, if loss_fn is None, the
                             network should contain the logic of loss and grads calculation, and the logic
                             of parallel if needed. Default: ``None``.
        optimizer (Cell): Optimizer for updating the weights. Default: ``None``.
        metrics (Union[dict, set]): A Dictionary or a set of metrics to be evaluated by the model during
                        training and testing. eg: {'accuracy', 'recall'}. Default: ``None``.
        eval_network (Cell): Network for evaluation. If not defined, `network` and `loss_fn` would be wrapped as
                             `eval_network`. Default: ``None``.
        eval_indexes (list): When defining the `eval_network`, if `eval_indexes` is None, all outputs of the
                             `eval_network` would be passed to metrics, otherwise `eval_indexes` must contain three
                             elements, including the positions of loss value, predicted value and label. The loss
                             value would be passed to the `Loss` metric, the predicted value and label would be passed
                             to other metric. Default: ``None``.
        amp_level (str): Option for argument `level` in `mindspore.amp.build_train_network`, level for mixed
            precision training. Supports [O0, O2, O3]. Default: "O0".

            - O0: Do not change.
            - O2: Cast network to float16, keep batchnorm run in float32, using dynamic loss scale.
            - O3: Cast network to float16, with additional property 'keep_batchnorm_fp32=False'.

            O2 is recommended on GPU, O3 is recommended on Ascend.

        loss_scale_manager (Union[None, LossScaleManager]): If it is None, the loss would not be scaled. Otherwise,
            scale the loss by LossScaleManager. It is a key argument.
            e.g. Use `loss_scale_manager=None` to set the value.
        keep_batchnorm_fp32 (bool): Keep Batchnorm running in `float32`. If it is set to true, the level setting before
            will be overwritten. Default: ``True``.
    """

    def __init__(self, network, loss_fn=None, optimizer=None, metrics=None, eval_network=None,
                 eval_indexes=None, amp_level="O0", **kwargs):
        super(ModelThor, self).__init__(network, loss_fn, optimizer, metrics, eval_network,
                                        eval_indexes, amp_level, **kwargs)
        if isinstance(network, nn.TrainOneStepCell):
            self._frequency = network.optimizer.get_frequency()
        else:
            self._frequency = optimizer.get_frequency()
        # used to stop training for early stop, such as stopAtTIme or stopATStep
        self.switch_branch_one = True
        self.index_first_order = 0
        self.train_network_init_flag = True
        self.has_do_dataset_init = False
        self._train_network = self._build_train_network()

    def _exec_preprocess(self, network, is_train, phase, dataset, dataset_sink_mode, sink_size=-1,
                         epoch_num=1, iter_first_order=1):
        """Initializes dataset."""
        if dataset_sink_mode and not is_train:
            dataset.__loop_size__ = 1
        dataset_helper = DatasetHelper(dataset, dataset_sink_mode, sink_size, epoch_num, iter_first_order)

        if dataset_sink_mode and context.get_context("device_target") != "GPU":
            network = connect_network_with_dataset(network, dataset_helper)
        network.set_train(is_train)
        network.phase = phase

        return dataset_helper, network

    def _train_gpu_sink_step(self, cb_params, inputs, list_callback, iter_first_order, run_context):
        """train gpu sink step"""
        if self.switch_branch_one:
            cb_params.cur_step_num += 1
            if self.train_network_init_flag:
                self._train_network.add_flags_recursive(thor=True)
            self._train_network.phase = 'train0'
            self.switch_branch_one = not self.switch_branch_one
            outputs = self._train_network(*inputs)
            cb_params.net_outputs = outputs
            list_callback.on_train_step_end(run_context)
        else:
            cb_params.cur_step_num += 1
            if self.train_network_init_flag:
                self._train_network.add_flags_recursive(thor=False)
                self.train_network_init_flag = False
            self._train_network.phase = 'train1'
            outputs = self._train_network(*inputs)
            cb_params.net_outputs = outputs
            self.index_first_order += 1
            if self.index_first_order == iter_first_order:
                self.index_first_order = 0
                self.switch_branch_one = not self.switch_branch_one
                list_callback.on_train_step_end(run_context)

    def _train_ascend_sink_step(self, cb_params, train_dataset, iter_first_order, inputs, list_callback, run_context):
        """train ascend sink step"""
        if self.switch_branch_one:
            cb_params.cur_step_num += 1
            if self.train_network_init_flag:
                self._train_network.add_flags_recursive(thor=True)
            self._train_network.phase = 'train0'
        else:
            cb_params.cur_step_num += iter_first_order
            if self.train_network_init_flag:
                self._train_network.add_flags_recursive(thor=False)
                self.train_network_init_flag = False
            self._train_network.phase = 'train1'
            if not self.has_do_dataset_init:
                _exec_datagraph(train_dataset, iter_first_order, phase='train1_dataset')
                self.has_do_dataset_init = True
        self.switch_branch_one = not self.switch_branch_one
        outputs = self._train_network(*inputs)
        cb_params.net_outputs = outputs
        list_callback.on_train_step_end(run_context)

    def _train_dataset_sink_process(self, epoch, train_dataset, list_callback=None, cb_params=None,
                                    sink_size=-1, initial_epoch=0, valid_infos=None):
        """
        Training process. The data would be passed to network through dataset channel.

        Args:
            epoch (int): Total number of iterations on the data.
            train_dataset (Dataset): A training dataset iterator. If there is no
                                     loss_fn, a tuple with multiple data (data1, data2, data3, ...) should be
                                     returned and passed to the network. Otherwise, a tuple (data, label) should
                                     be returned. The data and label would be passed to the network and loss
                                     function respectively.
            list_callback (Callback): Executor of callback list. Default: ``None``.
            cb_params (_InternalCallbackParam): Callback parameters. Default: ``None``.
            sink_size (int): Control the amount of data in each sink. Default: -1.
            initial_epoch (int): Epoch at which to start train, it useful for resuming a previous training run.
                                 Default: 0.
        """
        valid_dataset, _, _ = valid_infos
        if valid_dataset:
            raise ValueError("Evaluation in training is currently not supported in the second-order scenario of thor.")
        if sink_size == -1:
            epoch_num = epoch - initial_epoch
        else:
            epoch_num = math.ceil(epoch * sink_size / train_dataset.get_dataset_size()) - initial_epoch

        iter_first_order = self._frequency - 1
        iter_second_order = 1
        train_dataset.__loop_size__ = iter_second_order
        dataset_helper, train_network = self._exec_preprocess(self._train_network,
                                                              is_train=True,
                                                              phase='train',
                                                              dataset=train_dataset,
                                                              dataset_sink_mode=True,
                                                              sink_size=sink_size,
                                                              epoch_num=epoch_num,
                                                              iter_first_order=iter_first_order)

        self._train_network = train_network
        cb_params.train_network = self._train_network
        cb_params.cur_step_num = 0

        run_context = RunContext(cb_params)
        list_callback.on_train_begin(run_context)

        for i in range(initial_epoch, epoch):
            cb_params.cur_epoch_num = i + 1
            list_callback.on_train_epoch_begin(run_context)
            # for data sink dataset_helper only iter once, other wise iter epoch_size times.
            for inputs in dataset_helper:
                if _need_to_full() and context.get_context("device_target") == "GPU":
                    inputs = _to_full_tensor(inputs, self._device_number, self._global_rank)
                list_callback.on_train_step_begin(run_context)
                if context.get_context("device_target") == "GPU":
                    self._train_gpu_sink_step(cb_params, inputs, list_callback, iter_first_order, run_context)
                else:
                    self._train_ascend_sink_step(cb_params, train_dataset, iter_first_order, inputs, list_callback,
                                                 run_context)
            list_callback.on_train_epoch_end(run_context)
            should_stop = False or run_context.get_stop_requested()
            if should_stop:
                break
        dataset_helper.stop_send()

        list_callback.on_train_end(run_context)


__all__ = ["ModelThor"]
