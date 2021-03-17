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
"""Toolbox for Uncertainty Evaluation."""
from copy import deepcopy

import numpy as np
from mindspore._checkparam import Validator
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore.train import Model
from mindspore.train.callback import LossMonitor, ModelCheckpoint, CheckpointConfig
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from ...cell import Cell
from ...layer.basic import Dense, Flatten, Dropout
from ...layer.container import SequentialCell
from ...layer.conv import Conv2d
from ...loss import SoftmaxCrossEntropyWithLogits, MSELoss
from ...metrics import Accuracy, MSE
from ...optim import Adam


class UncertaintyEvaluation:
    r"""
    Toolbox for Uncertainty Evaluation.

    Args:
        model (Cell): The model for uncertainty evaluation.
        train_dataset (Dataset): A dataset iterator to train model.
        task_type (str): Option for the task types of model
            - regression: A regression model.
            - classification: A classification model.
        num_classes (int): The number of labels of classification.
                      If the task type is classification, it must be set; otherwise, it is not needed.
                      Default: None.
        epochs (int): Total number of iterations on the data. Default: 1.
        epi_uncer_model_path (str): The save or read path of the epistemic uncertainty model. Default: None.
        ale_uncer_model_path (str): The save or read path of the aleatoric uncertainty model. Default: None.
        save_model (bool): Whether to save the uncertainty model or not, if true, the epi_uncer_model_path
                        and ale_uncer_model_path must not be None. If false, the model to evaluate will be loaded from
                        the the path of the uncertainty model; if the path is not given , it will not save or load the
                        uncertainty model. Default: False.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> network = LeNet()
        >>> param_dict = load_checkpoint('checkpoint_lenet.ckpt')
        >>> load_param_into_net(network, param_dict)
        >>> ds_train = create_dataset('workspace/mnist/train')
        >>> ds_eval = create_dataset('workspace/mnist/test')
        >>> evaluation = UncertaintyEvaluation(model=network,
        ...                                    train_dataset=ds_train,
        ...                                    task_type='classification',
        ...                                    num_classes=10,
        ...                                    epochs=1,
        ...                                    epi_uncer_model_path=None,
        ...                                    ale_uncer_model_path=None,
        ...                                    save_model=False)
        >>> for eval_data in ds_eval.create_dict_iterator(output_numpy=True, num_epochs=1):
        ...    eval_data = Tensor(eval_data['image'], mstype.float32)
        ...    epistemic_uncertainty = evaluation.eval_epistemic_uncertainty(eval_data)
        ...    aleatoric_uncertainty = evaluation.eval_aleatoric_uncertainty(eval_data)
        >>> output = epistemic_uncertainty.shape
        >>> print(output)
        (32, 10)
        >>> output = aleatoric_uncertainty.shape
        >>> print(output)
        (32,)
    """

    def __init__(self, model, train_dataset, task_type, num_classes=None, epochs=1,
                 epi_uncer_model_path=None, ale_uncer_model_path=None, save_model=False):
        self.epi_model = deepcopy(model)
        self.ale_model = deepcopy(model)
        self.epi_train_dataset = train_dataset
        self.ale_train_dataset = deepcopy(train_dataset)
        self.task_type = task_type
        self.epochs = Validator.check_positive_int(epochs)
        self.epi_uncer_model_path = epi_uncer_model_path
        self.ale_uncer_model_path = ale_uncer_model_path
        self.save_model = Validator.check_bool(save_model)
        self.epi_uncer_model = None
        self.ale_uncer_model = None
        self.concat = P.Concat(axis=0)
        self.sum = P.ReduceSum()
        self.pow = P.Pow()
        if not isinstance(model, Cell):
            raise TypeError('The model should be Cell type.')
        if task_type not in ('regression', 'classification'):
            raise ValueError(
                'The task should be regression or classification.')
        if task_type == 'classification':
            self.num_classes = Validator.check_positive_int(num_classes)
        else:
            self.num_classes = num_classes
        if save_model:
            if epi_uncer_model_path is None or ale_uncer_model_path is None:
                raise ValueError("If save_model is True, the epi_uncer_model_path and "
                                 "ale_uncer_model_path should not be None.")

    def _get_epistemic_uncertainty_model(self):
        """
        Get the model which can obtain the epistemic uncertainty.
        """
        if self.epi_uncer_model is None:
            self.epi_uncer_model = EpistemicUncertaintyModel(self.epi_model)
            if self.epi_uncer_model.drop_count == 0 and self.epi_train_dataset is not None:
                if self.task_type == 'classification':
                    net_loss = SoftmaxCrossEntropyWithLogits(
                        sparse=True, reduction="mean")
                    net_opt = Adam(self.epi_uncer_model.trainable_params())
                    model = Model(self.epi_uncer_model, net_loss,
                                  net_opt, metrics={"Accuracy": Accuracy()})
                else:
                    net_loss = MSELoss()
                    net_opt = Adam(self.epi_uncer_model.trainable_params())
                    model = Model(self.epi_uncer_model, net_loss,
                                  net_opt, metrics={"MSE": MSE()})
                if self.save_model:
                    config_ck = CheckpointConfig(
                        keep_checkpoint_max=self.epochs)
                    ckpoint_cb = ModelCheckpoint(prefix='checkpoint_epi_uncer_model',
                                                 directory=self.epi_uncer_model_path,
                                                 config=config_ck)
                    model.train(self.epochs, self.epi_train_dataset, dataset_sink_mode=False,
                                callbacks=[ckpoint_cb, LossMonitor()])
                elif self.epi_uncer_model_path is None:
                    model.train(self.epochs, self.epi_train_dataset, dataset_sink_mode=False,
                                callbacks=[LossMonitor()])
                else:
                    uncer_param_dict = load_checkpoint(
                        self.epi_uncer_model_path)
                    load_param_into_net(self.epi_uncer_model, uncer_param_dict)

    def _eval_epistemic_uncertainty(self, eval_data, mc=10):
        """
        Evaluate the epistemic uncertainty of classification and regression models using MC dropout.
        """
        self._get_epistemic_uncertainty_model()
        self.epi_uncer_model.set_train(True)
        outputs = [None] * mc
        for i in range(mc):
            pred = self.epi_uncer_model(eval_data)
            outputs[i] = pred.asnumpy()
        if self.task_type == 'classification':
            outputs = np.stack(outputs, axis=2)
            epi_uncertainty = outputs.var(axis=2)
        else:
            outputs = np.stack(outputs, axis=1)
            epi_uncertainty = outputs.var(axis=1)
        epi_uncertainty = np.array(epi_uncertainty)
        return epi_uncertainty

    def _get_aleatoric_uncertainty_model(self):
        """
        Get the model which can obtain the aleatoric uncertainty.
        """
        if self.ale_train_dataset is None:
            raise ValueError(
                'The train dataset should not be None when evaluating aleatoric uncertainty.')
        if self.ale_uncer_model is None:
            self.ale_uncer_model = AleatoricUncertaintyModel(
                self.ale_model, self.num_classes, self.task_type)
            net_loss = AleatoricLoss(self.task_type)
            net_opt = Adam(self.ale_uncer_model.trainable_params())
            if self.task_type == 'classification':
                model = Model(self.ale_uncer_model, net_loss,
                              net_opt, metrics={"Accuracy": Accuracy()})
            else:
                model = Model(self.ale_uncer_model, net_loss,
                              net_opt, metrics={"MSE": MSE()})
            if self.save_model:
                config_ck = CheckpointConfig(keep_checkpoint_max=self.epochs)
                ckpoint_cb = ModelCheckpoint(prefix='checkpoint_ale_uncer_model',
                                             directory=self.ale_uncer_model_path,
                                             config=config_ck)
                model.train(self.epochs, self.ale_train_dataset, dataset_sink_mode=False,
                            callbacks=[ckpoint_cb, LossMonitor()])
            elif self.ale_uncer_model_path is None:
                model.train(self.epochs, self.ale_train_dataset, dataset_sink_mode=False,
                            callbacks=[LossMonitor()])
            else:
                uncer_param_dict = load_checkpoint(self.ale_uncer_model_path)
                load_param_into_net(self.ale_uncer_model, uncer_param_dict)

    def _eval_aleatoric_uncertainty(self, eval_data):
        """
        Evaluate the aleatoric uncertainty of classification and regression models.
        """
        self._get_aleatoric_uncertainty_model()
        _, var = self.ale_uncer_model(eval_data)
        ale_uncertainty = self.sum(self.pow(var, 2), 1)
        ale_uncertainty = ale_uncertainty.asnumpy()
        return ale_uncertainty

    def eval_epistemic_uncertainty(self, eval_data):
        """
        Evaluate the epistemic uncertainty of inference results, which also called model uncertainty.

        Args:
            eval_data (Tensor): The data samples to be evaluated, the shape must be (N,C,H,W).

        Returns:
            numpy.dtype, the epistemic uncertainty of inference results of data samples.
        """
        uncertainty = self._eval_epistemic_uncertainty(eval_data)
        return uncertainty

    def eval_aleatoric_uncertainty(self, eval_data):
        """
        Evaluate the aleatoric uncertainty of inference results, which also called data uncertainty.

        Args:
            eval_data (Tensor): The data samples to be evaluated, the shape must be (N,C,H,W).

        Returns:
            numpy.dtype, the aleatoric uncertainty of inference results of data samples.
        """
        uncertainty = self._eval_aleatoric_uncertainty(eval_data)
        return uncertainty


class EpistemicUncertaintyModel(Cell):
    """
    Using dropout during training and eval time which is approximate bayesian inference. In this way,
    we can obtain the epistemic uncertainty (also called model uncertainty).

    If the original model has Dropout layer, just use dropout when eval time, if not, add dropout layer
    after Dense layer or Conv layer, then use dropout during train and eval time.

    See more details in `Dropout as a Bayesian Approximation: Representing Model uncertainty in Deep Learning
    <https://arxiv.org/abs/1506.02142>`_.
    """

    def __init__(self, epi_model):
        super(EpistemicUncertaintyModel, self).__init__()
        self.drop_count = 0
        if not self._make_epistemic(epi_model):
            raise ValueError("The model has not Dense Layer or Convolution Layer, "
                             "it can not evaluate epistemic uncertainty so far.")
        self.epi_model = self._make_epistemic(epi_model)

    def construct(self, x):
        x = self.epi_model(x)
        return x

    def _make_epistemic(self, epi_model, keep_prob=0.5):
        """
        The dropout rate is set to 0.5 by default.
        """
        for (name, layer) in epi_model.name_cells().items():
            if isinstance(layer, (Conv2d, Dense, Dropout)):
                if isinstance(layer, Dropout):
                    self.drop_count += 1
                    return epi_model
                uncertainty_layer = layer
                uncertainty_name = name
                drop = Dropout(keep_prob=keep_prob)
                bnn_drop = SequentialCell([uncertainty_layer, drop])
                setattr(epi_model, uncertainty_name, bnn_drop)
                return epi_model
            if self._make_epistemic(layer):
                return epi_model
        return None


class AleatoricUncertaintyModel(Cell):
    """
    The aleatoric uncertainty (also called data uncertainty) is caused by input data, to obtain this
    uncertainty, the loss function must be modified in order to add variance into loss.

    See more details in `What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?
    <https://arxiv.org/abs/1703.04977>`_.
    """

    def __init__(self, ale_model, num_classes, task):
        super(AleatoricUncertaintyModel, self).__init__()
        self.task = task
        if task == 'classification':
            self.ale_model = ale_model
            self.var_layer = Dense(num_classes, num_classes)
        else:
            self.ale_model, self.var_layer, self.pred_layer = self._make_aleatoric(
                ale_model)

    def construct(self, x):
        if self.task == 'classification':
            pred = self.ale_model(x)
            var = self.var_layer(pred)
        else:
            x = self.ale_model(x)
            pred = self.pred_layer(x)
            var = self.var_layer(x)
        return pred, var

    def _make_aleatoric(self, ale_model):
        """
        In order to add variance into original loss, add var Layer after the original network.
        """
        dense_layer = dense_name = None
        for (name, layer) in ale_model.name_cells().items():
            if isinstance(layer, Dense):
                dense_layer = layer
                dense_name = name
        if dense_layer is None:
            raise ValueError("The model has not Dense Layer, "
                             "it can not evaluate aleatoric uncertainty so far.")
        setattr(ale_model, dense_name, Flatten())
        var_layer = Dense(dense_layer.in_channels, dense_layer.out_channels)
        return ale_model, var_layer, dense_layer


class AleatoricLoss(Cell):
    """
    The loss function of aleatoric model, different modification methods are adopted for
    classification and regression.
    """

    def __init__(self, task):
        super(AleatoricLoss, self).__init__()
        self.task = task
        if self.task == 'classification':
            self.sum = P.ReduceSum()
            self.exp = P.Exp()
            self.normal = C.normal
            self.to_tensor = P.ScalarToArray()
            self.entropy = SoftmaxCrossEntropyWithLogits(
                sparse=True, reduction="mean")
        else:
            self.mean = P.ReduceMean()
            self.exp = P.Exp()
            self.pow = P.Pow()

    def construct(self, data_pred, y):
        y_pred, var = data_pred
        if self.task == 'classification':
            sample_times = 10
            epsilon = self.normal((1, sample_times), self.to_tensor(
                0.0), self.to_tensor(1.0), 0)
            total_loss = 0
            for i in range(sample_times):
                y_pred_i = y_pred + epsilon[0][i] * var
                loss = self.entropy(y_pred_i, y)
                total_loss += loss
            avg_loss = total_loss / sample_times
            return avg_loss
        loss = self.mean(0.5 * self.exp(-var) *
                         self.pow(y - y_pred, 2) + 0.5 * var)
        return loss
