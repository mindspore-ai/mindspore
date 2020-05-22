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
"""Model."""
import mindspore.nn as nn
import numpy as np
from mindspore import context
from mindspore import log as logger
from mindspore._c_expression import init_exec_dataset
from mindspore._checkparam import check_input_data, check_output_data, check_int_positive, check_bool
from mindspore.common import dtype as mstype
from mindspore.common.dtype import pytype_to_dtype
from mindspore.common.tensor import Tensor
from mindspore.nn.metrics import Loss
from mindspore.nn.metrics import get_metrics
from mindspore.nn.wrap.cell_wrapper import _VirtualDatasetCell
from mindspore.parallel._utils import _get_parallel_mode, _get_device_num, _get_global_rank, \
    _get_parameter_broadcast, _device_number_check, _parameter_broadcast_check
from mindspore.train import amp
from mindspore.train.callback import _InternalCallbackParam, RunContext, _build_callbacks
from mindspore.train.parallel_utils import ParallelMode
from second_order.dataset_helper import DatasetHelper


def _convert_type(types):
    """
    Convert from numpy type to tensor type.
 
    Args:
        types (list): Numpy type list of element in dataset.
 
    Returns:
        list, list of element in dataset.
    """
    ms_types = []
    for np_type in types:
        ms_type = pytype_to_dtype(np_type)
        ms_types.append(ms_type)
    return ms_types


def _get_types_and_shapes(dataset):
    """Get dataset types and shapes."""
    dataset_types = _convert_type(dataset.output_types())
    dataset_shapes = dataset.output_shapes()
    return dataset_types, dataset_shapes


def _exec_datagraph(exec_dataset, dataset_size, phase='dataset'):
    """Initialize and execute the dataset graph."""
    batch_size = exec_dataset.get_batch_size()
    input_indexs = exec_dataset.input_indexs

    # transform data format
    dataset_types, dataset_shapes = _get_types_and_shapes(exec_dataset)
    init_exec_dataset(exec_dataset.__ME_INITED__,
                      dataset_size,
                      batch_size,
                      dataset_types,
                      dataset_shapes,
                      input_indexs,
                      phase=phase)


class Model:
    """
    High-Level API for Training or Testing.
 
    `Model` groups layers into an object with training and inference features.
 
    Args:
        network (Cell): The training or testing network.
        loss_fn (Cell): Objective function, if loss_fn is None, the
                             network should contain the logic of loss and grads calculation, and the logic
                             of parallel if needed. Default: None.
        optimizer (Cell): Optimizer for updating the weights. Default: None.
        metrics (Union[dict, set]): Dict or set of metrics to be evaluated by the model during
                        training and testing. eg: {'accuracy', 'recall'}. Default: None.
        eval_network (Cell): Network for evaluation. If not defined, `network` and `loss_fn` would be wrapped as
                             `eval_network`. Default: None.
        eval_indexes (list): In case of defining the `eval_network`, if `eval_indexes` is None, all outputs of
                             `eval_network` would be passed to metrics, otherwise `eval_indexes` must contain three
                             elements, representing the positions of loss value, predict value and label, the loss
                             value would be passed to `Loss` metric, predict value and label would be passed to other
                             metric. Default: None.
        amp_level (str): Option for argument `level` in `mindspore.amp.build_train_network`, level for mixed
            precision training. Supports [O0, O2]. Default: "O0".
 
            - O0: Do not change.
            - O2: Cast network to float16, keep batchnorm run in float32, using dynamic loss scale.
 
        loss_scale_manager (Union[None, LossScaleManager]): If None, not scale the loss, or else
            scale the loss by LossScaleManager. If it is set, overwrite the level setting. It's a eyword argument.
            e.g. Use `loss_scale_manager=None` to set the value.
        keep_batchnorm_fp32 (bool): Keep Batchnorm run in `float32`. If set, overwrite the level setting. Default: True.
 
    Examples:
        >>> class Net(nn.Cell):
        >>>     def __init__(self):
        >>>         super(Net, self).__init__()
        >>>         self.conv = nn.Conv2d(3, 64, 3, has_bias=False, weight_init='normal')
        >>>         self.bn = nn.BatchNorm2d(64)
        >>>         self.relu = nn.ReLU()
        >>>         self.flatten = nn.Flatten()
        >>>         self.fc = nn.Dense(64*224*224, 12) # padding=0
        >>>
        >>>     def construct(self, x):
        >>>         x = self.conv(x)
        >>>         x = self.bn(x)
        >>>         x = self.relu(x)
        >>>         x = self.flatten(x)
        >>>         out = self.fc(x)
        >>>         return out
        >>>
        >>> net = Net()
        >>> loss = nn.SoftmaxCrossEntropyWithLogits()
        >>> optim = Momentum(params=net.trainable_params(), learning_rate=0.1, momentum=0.9)
        >>> model = Model(net, loss_fn=loss, optimizer=optim, metrics=None)
        >>> dataset = get_dataset()
        >>> model.train(2, dataset)
    """

    def __init__(self, network, loss_fn=None, optimizer=None, metrics=None, eval_network=None,
                 eval_indexes=None, amp_level="O0", frequency=278, **kwargs):
        self._network = network
        self._loss_fn = loss_fn
        self._optimizer = optimizer
        self._loss_scale_manager = None
        self._loss_scale_manager_set = False
        self._keep_bn_fp32 = True
        self._frequency = frequency
        self._check_kwargs(kwargs)
        if 'keep_batchnorm_fp32' in kwargs:
            self._keep_bn_fp32 = kwargs['keep_batchnorm_fp32']
        if 'loss_scale_manager' in kwargs:
            self._loss_scale_manager = kwargs['loss_scale_manager']
            self._loss_scale_manager_set = True
        self._amp_level = amp_level
        self._parallel_mode = _get_parallel_mode()
        self._device_number = _get_device_num()
        self._global_rank = _get_global_rank()
        self._parameter_broadcast = _get_parameter_broadcast()

        self._train_network = self._build_train_network()
        self._build_eval_network(metrics, eval_network, eval_indexes)
        self._build_predict_network()

    def _check_kwargs(self, kwargs):
        for arg in kwargs:
            if arg not in ['loss_scale_manager', 'keep_batchnorm_fp32']:
                raise ValueError(f"Unsupport arg '{arg}'")

    def _build_train_network(self):
        """Build train network"""
        network = self._network
        if self._optimizer:
            if self._loss_scale_manager_set:
                network = amp.build_train_network(network,
                                                  self._optimizer,
                                                  self._loss_fn,
                                                  level=self._amp_level,
                                                  loss_scale_manager=self._loss_scale_manager,
                                                  keep_batchnorm_fp32=self._keep_bn_fp32)
            else:
                network = amp.build_train_network(network,
                                                  self._optimizer,
                                                  self._loss_fn,
                                                  level=self._amp_level,
                                                  keep_batchnorm_fp32=self._keep_bn_fp32)
        elif self._loss_fn:
            network = nn.WithLossCell(network, self._loss_fn)
        # If need to check if loss_fn is not None, but optimizer is None
        return network

    def _build_eval_network(self, metrics, eval_network, eval_indexes):
        """Build the network for evaluation."""
        self._metric_fns = get_metrics(metrics)
        if not self._metric_fns:
            return

        if eval_network is not None:
            if eval_indexes is not None and not (isinstance(eval_indexes, list) and len(eval_indexes) == 3):
                raise ValueError("Eval_indexes must be a list or None. If eval_indexes is a list, length of it \
                                 must be three. But got {}".format(eval_indexes))

            self._eval_network = eval_network
            self._eval_indexes = eval_indexes
        else:
            if self._loss_fn is None:
                raise ValueError("loss_fn can not be None.")
            self._eval_network = nn.WithEvalCell(self._network, self._loss_fn)
            self._eval_indexes = [0, 1, 2]

    def _build_predict_network(self):
        """Build the network for prediction."""
        self._predict_network = self._network
        if self._parallel_mode in (ParallelMode.SEMI_AUTO_PARALLEL, ParallelMode.AUTO_PARALLEL):
            self._predict_network = _VirtualDatasetCell(self._network)

    def _clear_metrics(self):
        """Clear metrics local values."""
        for metric in self._metric_fns.values():
            metric.clear()

    def _update_metrics(self, outputs):
        """Update metrics local values."""
        if not isinstance(outputs, tuple):
            raise ValueError("The `outputs` is not tuple.")

        if self._eval_indexes is not None and len(outputs) < 3:
            raise ValueError("The length of `outputs` must be greater than or equal to 3, \
                             but got {}".format(len(outputs)))

        for metric in self._metric_fns.values():
            if self._eval_indexes is None:
                metric.update(*outputs)
            else:
                if isinstance(metric, Loss):
                    metric.update(outputs[self._eval_indexes[0]])
                else:
                    metric.update(outputs[self._eval_indexes[1]], outputs[self._eval_indexes[2]])

    def _get_metrics(self):
        """Get metrics local values."""
        metrics = dict()
        for key, value in self._metric_fns.items():
            metrics[key] = value.eval()
        return metrics

    def _get_scaling_sens(self):
        """get the scaling sens"""
        scaling_sens = 1
        if self._loss_scale_manager is not None:
            scaling_sens = self._loss_scale_manager.get_loss_scale()
        if self._parallel_mode == ParallelMode.DATA_PARALLEL:
            scaling_sens /= self._device_number
        return scaling_sens

    def _train(self, epoch, train_dataset, callbacks=None, dataset_sink_mode=True):
        """
        Training.
 
        Args:
            epoch (int): Total number of iterations on the data.
            train_dataset (Dataset): A training dataset iterator. If there is no
                                     loss_fn, a tuple with multiply data (data1, data2, data3, ...) will be
                                     returned and passed to the network. Otherwise, a tuple (data, label) will
                                     be returned, and the data and label are passed to the network and loss
                                     function respectively.
            callbacks (list): List of callback object. Callbacks which should be executed while training. Default: None.
            dataset_sink_mode (bool): Determines whether to pass the data through dataset channel. Default: True.
                                      Configure pynative mode, the training process will be performed with
                                      dataset not sink.
        """
        epoch = check_int_positive(epoch)
        self._train_network.set_train()

        if self._parameter_broadcast:
            self._train_network.set_broadcast_flag()

        # build callback list
        list_callback = _build_callbacks(callbacks)
        cb_params = _InternalCallbackParam()
        cb_params.train_network = self._train_network
        cb_params.epoch_num = epoch
        cb_params.batch_num = train_dataset.get_dataset_size()
        cb_params.mode = "train"
        cb_params.loss_fn = self._loss_fn
        cb_params.optimizer = self._optimizer
        cb_params.parallel_mode = self._parallel_mode
        cb_params.device_number = self._device_number
        cb_params.train_dataset = train_dataset
        cb_params.list_callback = list_callback

        if dataset_sink_mode:
            if context.get_context("mode") == context.PYNATIVE_MODE:
                logger.warning("The pynative mode cannot support dataset sink mode currently."
                               "So the training process will be performed with dataset not sink.")
                self._train_process(epoch, train_dataset, list_callback, cb_params)
            else:
                self._train_dataset_sink_process(epoch, train_dataset, list_callback, cb_params)
        else:
            self._train_process(epoch, train_dataset, list_callback, cb_params)

    def _train_dataset_sink_process(self, epoch, train_dataset, list_callback=None, cb_params=None):
        """
        Training process. The data would be passed to network through dataset channel.
 
        Args:
            epoch (int): Total number of iterations on the data.
            train_dataset (Dataset): A training dataset iterator. If there is no
                                     loss_fn, a tuple with multiply data (data1, data2, data3, ...) should be
                                     returned and passed to the network. Otherwise, a tuple (data, label) should
                                     be returned, and the data and label are passed to the network and loss
                                     function respectively.
            list_callback (_ListCallback): Executor of callback list. Default: None.
            cb_params (_InternalCallbackParam): Callback parameters. Default: None.
        """
        # remove later to deal with loop sink
        iter_first_order = 277
        iter_second_order = 1
        train_dataset.__loop_size__ = iter_second_order
        need_wrap = False
        if not hasattr(train_dataset, '__ME_INITED__') and context.get_context("enable_loop_sink") \
                and not context.get_context("enable_ge"):
            need_wrap = True

        dataset_helper = DatasetHelper(train_dataset, iter_first_order)
        # remove later to deal with loop sink
        if need_wrap:
            self._train_network = nn.DataWrapper(self._train_network, *(dataset_helper.types_shapes()),
                                                 train_dataset.__ME_INITED__)
            cb_params.train_network = self._train_network
            self._train_network.set_train()

        cb_params.cur_step_num = 0
        loop_size = dataset_helper.loop_size()
        run_context = RunContext(cb_params)
        list_callback.begin(run_context)

        # used to stop training for early stop, such as stopAtTIme or stopATStep
        should_stop = False
        has_do_train1_dataset = False
        checkpoint_branch_one = True
        for i in range(epoch):
            cb_params.cur_epoch_num = i + 1
            list_callback.epoch_begin(run_context)

            # for data sink dataset_helper only iter once, other wise iter epoch_size times.
            for inputs in dataset_helper:
                list_callback.step_begin(run_context)
                if checkpoint_branch_one:
                    cb_params.cur_step_num += loop_size
                    self._train_network.set_second_order(True)
                    self._train_network.phase = 'train0'
                else:
                    cb_params.cur_step_num += iter_first_order
                    self._train_network.set_second_order(False)
                    self._train_network.phase = 'train1'
                    if not has_do_train1_dataset:
                        _exec_datagraph(train_dataset, iter_first_order, phase='train1_dataset')
                        has_do_train1_dataset = True
                checkpoint_branch_one = not checkpoint_branch_one
                outputs = self._train_network(*inputs)
                cb_params.net_outputs = outputs
                list_callback.step_end(run_context)

            list_callback.epoch_end(run_context)
            should_stop = should_stop or run_context.get_stop_requested()
            if should_stop:
                break

        list_callback.end(run_context)

    def _train_process(self, epoch, train_dataset, list_callback=None, cb_params=None):
        """
        Training process. The data would be passed to network directly.
 
        Args:
            epoch (int): Total number of iterations on the data.
            train_dataset (Dataset): A training dataset iterator. If there is no
                                     loss_fn, a tuple with multiply data (data1, data2, data3, ...) should be
                                     returned and passed to the network. Otherwise, a tuple (data, label) should
                                     be returned, and the data and label are passed to the network and loss
                                     function respectively.
            list_callback (_ListCallback): Executor of callback list. Default: None.
            cb_params (_InternalCallbackParam): Callback parameters. Default: None.
        """
        dataset_helper = DatasetHelper(train_dataset, dataset_sink_mode=False)
        cb_params.cur_step_num = 0
        run_context = RunContext(cb_params)
        _callback_wrapper(list_callback, run_context, "begin")
        # used to stop training for early stop, such as stopAtTIme or stopATStep
        should_stop = False

        for i in range(epoch):
            cb_params.cur_epoch_num = i + 1

            _callback_wrapper(list_callback, run_context, "epoch_begin")

            for next_element in dataset_helper:
                len_element = len(next_element)
                if self._loss_fn and len_element != 2:
                    raise ValueError("when loss_fn is not None, train_dataset should"
                                     "return two elements, but got {}".format(len_element))
                cb_params.cur_step_num += 1
                _callback_wrapper(list_callback, run_context, "step_begin")

                overflow = False
                if self._loss_scale_manager and self._loss_scale_manager.get_drop_overflow_update():
                    scaling_sens = self._get_scaling_sens()
                    next_element = tuple(next_element) + (Tensor(scaling_sens, mstype.float32),)

                outputs = self._train_network(*next_element)
                cb_params.net_outputs = outputs
                if self._loss_scale_manager and self._loss_scale_manager.get_drop_overflow_update():
                    _, overflow, _ = outputs
                    overflow = np.all(overflow.asnumpy())
                    self._loss_scale_manager.update_loss_scale(overflow)

                _callback_wrapper(list_callback, run_context, "step_end")
                should_stop = should_stop or run_context.get_stop_requested()
                if should_stop:
                    break

            train_dataset.reset()

            _callback_wrapper(list_callback, run_context, "epoch_end")
            should_stop = should_stop or run_context.get_stop_requested()
            if should_stop:
                break

        _callback_wrapper(list_callback, run_context, "end")

    def train(self, epoch, train_dataset, callbacks=None, dataset_sink_mode=True):
        """
        Training API where the iteration is controlled by python front-end.
 
        When setting pynative mode, the training process will be performed with dataset not sink.
 
        Note:
            CPU is not supported when dataset_sink_mode is true.
            If dataset_sink_mode is True, epoch of training should be equal to the count of repeat
            operation in dataset processing. Otherwise, errors could occur since the amount of data
            is not the amount training requires.
            If dataset_sink_mode is True, data will be sent to device. If device is Ascend, features
            of data will be transferred one by one. The limitation of data transmission per time is 256M.
 
        Args:
            epoch (int): Total number of iterations on the data.
            train_dataset (Dataset): A training dataset iterator. If there is no
                                     loss_fn, a tuple with multiply data (data1, data2, data3, ...) should be
                                     returned and passed to the network. Otherwise, a tuple (data, label) should
                                     be returned, and the data and label are passed to the network and loss
                                     function respectively.
            callbacks (list): List of callback object. Callbacks which should be excuted while training. Default: None.
            dataset_sink_mode (bool): Determines whether to pass the data through dataset channel. Default: True.
                                      Configure pynative mode, the training process will be performed with
                                      dataset not sink.
 
 
        Examples:
            >>> dataset = get_dataset()
            >>> net = Net()
            >>> loss = nn.SoftmaxCrossEntropyWithLogits()
            >>> loss_scale_manager = FixedLossScaleManager()
            >>> optim = Momentum(params=net.trainable_params(), learning_rate=0.1, momentum=0.9)
            >>> model = Model(net, loss_fn=loss, optimizer=optim, metrics=None, loss_scale_manager=loss_scale_manager)
            >>> model.train(2, dataset)
        """
        repeat_count = train_dataset.get_repeat_count()
        if epoch != repeat_count and dataset_sink_mode is True:
            logger.warning(f"The epoch_size {epoch} is not the same with dataset repeat_count {repeat_count}")
        check_bool(dataset_sink_mode)
        _device_number_check(self._parallel_mode, self._device_number)
        _parameter_broadcast_check(self._parallel_mode, self._parameter_broadcast)

        if context.get_context("device_target") in ["CPU", "GPU"] and context.get_context("enable_loop_sink"):
            raise ValueError("CPU and GPU can't support loop sink, please set enable_loop_sink=False.")

        self._train(epoch,
                    train_dataset,
                    callbacks=callbacks,
                    dataset_sink_mode=dataset_sink_mode)

    def _eval_dataset_sink_process(self, valid_dataset, list_callback=None, cb_params=None):
        """
        Evaluation. The data would be passed to network through dataset channel.
 
        Args:
            valid_dataset (Dataset): Dataset to evaluate the model.
            list_callback (ListCallback): Executor of callback list. Default: None.
            cb_params (_InternalCallbackParam): Callback parameters. Default: None.
 
        Returns:
            Dict, returns the loss value & metrics values for the model in test mode.
        """
        _device_number_check(self._parallel_mode, self._device_number)

        run_context = RunContext(cb_params)

        # remove later to deal with loop sink
        need_wrap = False
        if not hasattr(valid_dataset, '__ME_INITED__') and context.get_context("enable_loop_sink") \
                and not context.get_context("enable_ge"):
            need_wrap = True

        valid_dataset.__loop_size__ = 1
        dataset_helper = DatasetHelper(valid_dataset)

        # remove later to deal with loop sink
        if need_wrap:
            self._eval_network = nn.DataWrapper(self._eval_network, *(dataset_helper.types_shapes()),
                                                valid_dataset.__ME_INITED__)
            self._eval_network.set_train(mode=False)
            self._eval_network.phase = 'eval'
        list_callback.begin(run_context)

        for inputs in dataset_helper:
            cb_params.cur_step_num += 1
            list_callback.step_begin(run_context)

            outputs = self._eval_network(*inputs)

            cb_params.net_outputs = outputs
            list_callback.step_end(run_context)
            self._update_metrics(outputs)

        metrics = self._get_metrics()
        cb_params.metrics = metrics
        list_callback.end(run_context)

        return metrics

    def _eval_process(self, valid_dataset, list_callback=None, cb_params=None):
        """
        Evaluation. The data would be passed to network directly.
 
        Args:
            valid_dataset (Dataset): Dataset to evaluate the model.
            list_callback (ListCallback): Executor of callback list. Default: None.
            cb_params (_InternalCallbackParam): Callback parameters. Default: None.
 
        Returns:
            Dict, returns the loss value & metrics values for the model in test mode.
        """
        run_context = RunContext(cb_params)
        list_callback.begin(run_context)

        dataset_helper = DatasetHelper(valid_dataset, dataset_sink_mode=False)
        for next_element in dataset_helper:
            cb_params.cur_step_num += 1
            list_callback.step_begin(run_context)
            outputs = self._eval_network(*next_element)
            cb_params.net_outputs = outputs
            list_callback.step_end(run_context)
            self._update_metrics(outputs)

        metrics = self._get_metrics()
        cb_params.metrics = metrics
        list_callback.end(run_context)
        return metrics

    def eval(self, valid_dataset, callbacks=None, dataset_sink_mode=True):
        """
        Evaluation API where the iteration is controlled by python front-end.
 
        Configure to pynative mode, the evaluation will be performed with dataset non-sink mode.
 
        Note:
            CPU is not supported when dataset_sink_mode is true.
            If dataset_sink_mode is True, data will be sent to device. If device is Ascend, features
            of data will be transferred one by one. The limitation of data transmission per time is 256M.
 
        Args:
            valid_dataset (Dataset): Dataset to evaluate the model.
            callbacks (list): List of callback object. Callbacks which should be excuted
                              while training. Default: None.
            dataset_sink_mode (bool): Determines whether to pass the data through dataset channel. Default: True.
 
        Returns:
            Dict, returns the loss value & metrics values for the model in test mode.
 
        Examples:
            >>> dataset = get_dataset()
            >>> net = Net()
            >>> loss = nn.SoftmaxCrossEntropyWithLogits()
            >>> model = Model(net, loss_fn=loss, optimizer=None, metrics={'acc'})
            >>> model.eval(dataset)
        """
        check_bool(dataset_sink_mode)
        if not self._metric_fns:
            raise ValueError("metric fn can not be None or empty.")

        list_callback = _build_callbacks(callbacks)
        cb_params = _InternalCallbackParam()
        cb_params.eval_network = self._eval_network
        cb_params.valid_dataset = valid_dataset
        cb_params.batch_num = valid_dataset.get_dataset_size()
        cb_params.mode = "eval"
        cb_params.cur_step_num = 0

        self._eval_network.set_train(mode=False)
        self._eval_network.phase = 'eval'

        self._clear_metrics()

        if dataset_sink_mode:
            return self._eval_dataset_sink_process(valid_dataset, list_callback, cb_params)
        return self._eval_process(valid_dataset, list_callback, cb_params)

    def predict(self, *predict_data):
        """
        Generates output predictions for the input samples.
 
        Data could be single tensor, or list of tensor, tuple of tensor.
 
        Note:
            Batch data should be put together in one tensor.
 
        Args:
           predict_data (Tensor): Tensor of predict data. can be array, list or tuple.
 
        Returns:
            Tensor, array(s) of predictions.
 
        Examples:
            >>> input_data = Tensor(np.random.randint(0, 255, [1, 3, 224, 224]), mindspore.float32)
            >>> model = Model(Net())
            >>> model.predict(input_data)
        """
        self._predict_network.set_train(False)
        check_input_data(*predict_data, data_class=Tensor)
        result = self._predict_network(*predict_data)

        check_output_data(result)
        return result


__all__ = ["Model"]
