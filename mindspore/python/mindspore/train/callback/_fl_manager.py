# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
"""FederatedLearningManager related class and functions."""

from copy import deepcopy
import numpy as np
from mindspore import context, nn
from mindspore.common import Parameter, ParameterTuple
from mindspore.train.callback import Callback
from mindspore.ops import operations as P
from mindspore._checkparam import Validator, Rel


class _StartFLJob(nn.Cell):
    """
    StartFLJob for Federated Learning Worker.
    """

    def __init__(self, data_size):
        super(_StartFLJob, self).__init__()
        self.start_fl_job = P.StartFLJob(data_size)

    def construct(self):
        succ = self.start_fl_job()
        return succ


class _UpdateAndGetModel(nn.Cell):
    """
    Update and Get Model for Federated Learning Worker.
    """

    def __init__(self, weights, encrypt_type=""):
        super(_UpdateAndGetModel, self).__init__()
        self.update_model = P.UpdateModel(encrypt_type)
        self.get_model = P.GetModel()
        self.weights = weights

    def construct(self):
        self.update_model(self.weights)
        succ = self.get_model(self.weights)
        return succ


class _ExchangeKeys(nn.Cell):
    """
    Exchange Keys for Stable PW Encrypt.
    """

    def __init__(self):
        super(_ExchangeKeys, self).__init__()
        self.exchange_keys = P.ExchangeKeys()

    def construct(self):
        return self.exchange_keys()


class _GetKeys(nn.Cell):
    """
    Get Keys for Stable PW Encrypt.
    """

    def __init__(self):
        super(_GetKeys, self).__init__()
        self.get_keys = P.GetKeys()

    def construct(self):
        return self.get_keys()


class FederatedLearningManager(Callback):
    """
    Manage Federated Learning during training.

    Args:
        model (nn.Cell): A training model.
        sync_frequency (int): Synchronization frequency of parameters in Federated Learning.
                              Note that in dataset sink mode, the unit of the frequency is the number of epochs.
                              Otherwise, the unit of the frequency is the number of steps.
        sync_type (str): Parameter synchronization type in Federated Learning.
                         Supports ["fixed", "adaptive"]. Default: "fixed".

                         - fixed: The frequency of parameter synchronization is fixed.
                         - adaptive: The frequency of parameter synchronization changes adaptively.
        min_consistent_rate (float): Minimum consistency ratio threshold. The greater the value, the more
                                     difficult it is to improve the synchronization frequency.
                                     Value range: greater than or equal to 0.0. Default: 1.1.
        min_consistent_rate_at_round (int): The number of rounds of the minimum consistency ratio threshold.
                                            The greater the value, the more difficult it is to improve the
                                            synchronization frequency.
                                            Value range: greater than or equal to 0. Default: 0.
        ema_alpha (float): Gradient consistency smoothing coefficient. The smaller the value, the more the
                           frequency will be judged according to the gradient bifurcation of the current round
                           more. Otherwise it will be judged according to the historical gradient bifurcation
                           more.
                           Value range: (0.0, 1.0). Default: 0.5.
        observation_window_size (int): The number of rounds in the observation time window. The greater the
                                       value, the more difficult it is to reduce the synchronization frequency.
                                       Value range: greater than 0. Default: 5.
        frequency_increase_ratio (int): Frequency increase amplitude. The greater the value, the greater the
                                        frequency increase amplitude.
                                        Value range: greater than 0. Default: 2.
        unchanged_round (int): The number of rounds whose frequency does not change. The frequency is unchanged
                               before unchanged_round rounds.
                               Value range: greater than or equal to 0. Default: 0.

    Note:
        This is an experimental prototype that is subject to change.
    """

    def __init__(self, model, sync_frequency, sync_type='fixed', **kwargs):
        super(FederatedLearningManager, self).__init__()
        server_mode = context.get_fl_context("server_mode")
        if server_mode not in ("FEDERATED_LEARNING", "HYBRID_TRAINING"):
            raise ValueError("server_mode must in (\"FEDERATED_LEARNING\", \"HYBRID_TRAINING\")")
        Validator.check_isinstance('model', model, nn.Cell)
        Validator.check_positive_int(sync_frequency)
        Validator.check_string(sync_type, ["fixed", "adaptive"])
        self._model = model
        self._sync_frequency = sync_frequency
        self._next_sync_iter_id = self._sync_frequency
        self._sync_type = sync_type
        self._global_step = 0
        self._data_size = 0
        self._encrypt_type = kwargs.get("encrypt_type", "NOT_ENCRYPT")
        if self._encrypt_type != "NOT_ENCRYPT" and self._encrypt_type != "STABLE_PW_ENCRYPT":
            raise ValueError(
                "encrypt_mode must be 'NOT_ENCRYPT' or 'STABLE_PW_ENCRYPT', but got {}.".format(self._encrypt_type))

        self._as_prefix = "as_abs_grad."

        self._min_consistent_rate = kwargs.get("min_consistent_rate", 1.1)
        Validator.check_non_negative_float(self._min_consistent_rate)
        self._min_consistent_rate_at_round = kwargs.get("min_consistent_rate_at_round", 0)
        Validator.check_non_negative_int(self._min_consistent_rate_at_round)
        self._ema_alpha = kwargs.get("ema_alpha", 0.5)
        Validator.check_float_range(self._ema_alpha, 0.0, 1.0, Rel.INC_NEITHER)
        self._observation_window_size = kwargs.get("observation_window_size", 5)
        Validator.check_positive_int(self._observation_window_size)
        self._frequency_increase_ratio = kwargs.get("frequency_increase_ratio", 2)
        Validator.check_positive_int(self._frequency_increase_ratio)
        self._unchanged_round = kwargs.get("unchanged_round", 0)
        Validator.check_non_negative_int(self._unchanged_round)

        self._round_id = 0
        self._last_param = dict()
        self._model_size = 0
        self._grads_ema = dict()
        self._abs_grads_ema = dict()

        if self._is_adaptive_sync():
            self._last_param = {_.name: deepcopy(_.asnumpy()) for _ in self._model.trainable_params()
                                if self._as_prefix not in _.name}
            for param in self._model.trainable_params():
                if self._as_prefix not in param.name:
                    self._model_size += np.product(param.shape)
                    self._grads_ema[param.name] = np.zeros(param.shape)
                    self._abs_grads_ema[param.name] = np.zeros(param.shape)
            self._model_size = float(self._model_size)
            self._as_wrap_cell()

    def _is_adaptive_sync(self):
        """
        Determine whether adaptive frequency synchronization is required.
        """
        return self._sync_type == "adaptive"

    def _as_wrap_cell(self):
        """
        Wrap Cell for adaptive synchronization.
        """
        param_list = list()
        for param in self._model.trainable_params():
            new_param = param.clone()
            new_param.name = self._as_prefix + param.name
            param_list.append(new_param)
        for param in param_list:
            self._model.insert_param_to_cell(param.name, param, False)

    def _as_set_grads(self):
        """
        Set the absolute value of the gradient for adaptive synchronization.
        """
        abs_grads = dict()
        for param in self._model.trainable_params():
            if self._as_prefix not in param.name:
                try:
                    abs_grads[self._as_prefix + param.name] = np.abs(param.asnumpy() - self._last_param[param.name])
                except KeyError:
                    print("{} is not in self._last_param".format(param.name))
        for param in self._model.trainable_params():
            if self._as_prefix in param.name:
                try:
                    param.set_data(Parameter(abs_grads[param.name]))
                except KeyError:
                    print("{} is not in abs_grads".format(param.name))

    def _as_analyze_gradient(self):
        """
        Analysis of relevant statistics based on gradient for adaptive synchronization.
        """
        worker_num = context.get_fl_context("worker_num")
        ema_alpha = self._ema_alpha
        consistent_rate_sum = 0.0
        grads = dict()
        abs_grads = dict()
        for param in self._model.trainable_params():
            if self._as_prefix in param.name:
                abs_grads[param.name.replace(self._as_prefix, '')] = param.asnumpy() * worker_num
            else:
                try:
                    grads[param.name] = (param.asnumpy() - self._last_param[param.name]) * worker_num
                except KeyError:
                    print("{} is not in self._last_param".format(param.name))
        for last_p in self._last_param:
            try:
                self._grads_ema[last_p] = ema_alpha * self._grads_ema[last_p] + (1 - ema_alpha) * grads[last_p]
            except KeyError:
                print("{} is not in self._grads_ema".format(last_p))
                continue
            try:
                self._abs_grads_ema[last_p] = ema_alpha * self._abs_grads_ema[last_p] + (1 - ema_alpha) * abs_grads[
                    last_p]
            except KeyError:
                print("{} is not in self._abs_grads_ema".format(last_p))
                continue
            try:
                divide_base = np.where(self._abs_grads_ema[last_p] == 0,
                                       np.ones(self._abs_grads_ema[last_p].shape), self._abs_grads_ema[last_p])
            except KeyError:
                print("{} is not in self._abs_grads_ema".format(last_p))
                continue
            try:
                layer_consistent_rate = np.abs(self._grads_ema[last_p]) / divide_base
                consistent_rate_sum += np.sum(layer_consistent_rate)
            except KeyError:
                print("{} is not in self._grads_ema".format(last_p))

        consistent_rate = float(consistent_rate_sum / self._model_size)

        if self._min_consistent_rate > consistent_rate:
            self._min_consistent_rate = consistent_rate
            self._min_consistent_rate_at_round = self._round_id
        elif self._round_id - self._min_consistent_rate_at_round > self._observation_window_size and \
                self._sync_frequency > 1 and self._round_id > self._unchanged_round:
            self._sync_frequency = (self._sync_frequency + self._frequency_increase_ratio - 1) \
                                    // self._frequency_increase_ratio
            self._min_consistent_rate = 1.1
            self._min_consistent_rate_at_round = self._round_id
            self._observation_window_size *= self._frequency_increase_ratio

            for param in self._model.trainable_params():
                if self._as_prefix not in param.name:
                    self._grads_ema[param.name] = np.zeros(param.shape)
                    self._abs_grads_ema[param.name] = np.zeros(param.shape)

    def _as_set_last_param(self):
        """
        Set the value of last parameters for adaptive synchronization.
        """
        self._last_param = {_.name: deepcopy(_.asnumpy()) for _ in self._model.trainable_params()
                            if self._as_prefix not in _.name}

    def step_end(self, run_context):
        """
        Synchronization parameters at the end of step. If sync_type is "adaptive", the synchronous frequency is
        adaptively adjusted here.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        self._global_step += 1
        cb_params = run_context.original_args()
        self._data_size += cb_params.batch_num
        if context.get_fl_context("ms_role") == "MS_WORKER":
            if self._global_step == self._next_sync_iter_id:
                start_fl_job = _StartFLJob(self._data_size)
                start_fl_job()
                self._data_size = 0
                if self._is_adaptive_sync():
                    self._as_set_grads()
                if self._encrypt_type == "STABLE_PW_ENCRYPT":
                    exchange_keys = _ExchangeKeys()
                    exchange_keys()
                    get_keys = _GetKeys()
                    get_keys()
                    update_and_get_model = _UpdateAndGetModel(ParameterTuple(self._model.trainable_params()),
                                                              self._encrypt_type)
                else:
                    update_and_get_model = _UpdateAndGetModel(ParameterTuple(self._model.trainable_params()))
                update_and_get_model()
                self._next_sync_iter_id = self._global_step + self._sync_frequency
                if self._is_adaptive_sync():
                    self._as_analyze_gradient()
                    self._round_id += 1
                    self._as_set_last_param()

                print("sync step is: {}".format(self._global_step))
