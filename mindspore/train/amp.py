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
"""Auto mixed precision."""
from .. import nn
from .._checkparam import Validator as validator
from .._checkparam import Rel
from ..common import dtype as mstype
from ..nn.wrap.cell_wrapper import _VirtualDatasetCell, _TrainPipelineAccuStepCell
from ..nn.wrap.loss_scale import _TrainPipelineWithLossScaleCell
from ..ops import functional as F
from ..parallel._utils import _get_parallel_mode, _get_pipeline_stages
from .loss_scale_manager import DynamicLossScaleManager, LossScaleManager
from ..context import ParallelMode
from .. import boost
from .. import context


class _OutputTo16(nn.Cell):
    "Wrap cell for amp. Cast network output back to float16"

    def __init__(self, op):
        super(_OutputTo16, self).__init__(auto_prefix=False)
        self._op = op

    def construct(self, x):
        return F.cast(self._op(x), mstype.float16)


def _do_keep_batchnorm_fp32(network):
    """Do keep batchnorm fp32."""
    cells = network.name_cells()
    change = False
    for name in cells:
        subcell = cells[name]
        if subcell == network:
            continue
        elif isinstance(subcell, (nn.BatchNorm2d, nn.BatchNorm1d)):
            network._cells[name] = _OutputTo16(subcell.to_float(mstype.float32))
            change = True
        else:
            _do_keep_batchnorm_fp32(subcell)
    if isinstance(network, nn.SequentialCell) and change:
        network.cell_list = list(network.cells())


_config_level = {
    "O0": {
        "keep_batchnorm_fp32": False,
        "cast_model_type": mstype.float32,
        "loss_scale_manager": None},
    "O2": {
        "keep_batchnorm_fp32": True,
        "cast_model_type": mstype.float16,
        "loss_scale_manager": DynamicLossScaleManager()},
    "O3": {
        "keep_batchnorm_fp32": False,
        "cast_model_type": mstype.float16,
        "loss_scale_manager": None}}


def _check_kwargs(key_words):
    """Check kwargs."""
    for arg in key_words:
        if arg not in ['cast_model_type', 'keep_batchnorm_fp32', 'loss_scale_manager']:
            raise ValueError(f"Unsupported arg '{arg}'")

    if 'cast_model_type' in key_words:
        validator.check_type_name('cast_model_type', key_words['cast_model_type'],
                                  [mstype.float16, mstype.float32], None)
    if 'keep_batchnorm_fp32' in key_words:
        validator.check_value_type('keep_batchnorm_fp32', key_words['keep_batchnorm_fp32'], bool)
    if 'loss_scale_manager' in key_words:
        loss_scale_manager = key_words['loss_scale_manager']
        if loss_scale_manager:
            validator.check_value_type('loss_scale_manager', loss_scale_manager, LossScaleManager)


def _add_loss_network(network, loss_fn, cast_model_type):
    """Add loss network."""

    class WithLossCell(nn.Cell):
        "Wrap loss for amp. Cast network output back to float32"

        def __init__(self, backbone, loss_fn):
            super(WithLossCell, self).__init__(auto_prefix=False)
            self._backbone = backbone
            self._loss_fn = loss_fn

        def construct(self, data, label):
            out = self._backbone(data)
            label = F.mixed_precision_cast(mstype.float32, label)
            return self._loss_fn(F.mixed_precision_cast(mstype.float32, out), label)

    validator.check_value_type('loss_fn', loss_fn, nn.Cell)
    if cast_model_type == mstype.float16:
        network = WithLossCell(network, loss_fn)
    else:
        network = nn.WithLossCell(network, loss_fn)
    return network


def build_train_network(network, optimizer, loss_fn=None, level='O0', boost_level='O0', **kwargs):
    """
    Build the mixed precision training cell automatically.

    Args:
        network (Cell): Definition of the network.
        loss_fn (Union[None, Cell]): Definition of the loss_fn. If None, the `network` should have the loss inside.
            Default: None.
        optimizer (Optimizer): Optimizer to update the Parameter.
        level (str): Supports ["O0", "O2", "O3", "auto"]. Default: "O0".

            - O0: Do not change.
            - O2: Cast network to float16, keep batchnorm and `loss_fn` (if set) run in float32,
              using dynamic loss scale.
            - O3: Cast network to float16, with additional property `keep_batchnorm_fp32=False` .
            - auto: Set to level to recommended level in different devices. Set level to O2 on GPU, Set
              level to O3 Ascend. The recommended level is choose by the export experience, cannot
              always general. User should specify the level for special network.

            O2 is recommended on GPU, O3 is recommended on Ascend.Property of `keep_batchnorm_fp32` , `cast_model_type`
            and `loss_scale_manager` determined by `level` setting may be overwritten by settings in `kwargs` .

        boost_level (str): Option for argument `level` in `mindspore.boost` , level for boost mode
            training. Supports ["O0", "O1", "O2"]. Default: "O0".

            - O0: Do not change.
            - O1: Enable the boost mode, the performance is improved by about 20%, and
              the accuracy is the same as the original accuracy.
            - O2: Enable the boost mode, the performance is improved by about 30%, and
              the accuracy is reduced by less than 3%.

            If O1 or O2 mode is set, the boost related library will take effect automatically.

        cast_model_type (:class:`mindspore.dtype`): Supports `mstype.float16` or `mstype.float32` . If set, the
            network will be casted to `cast_model_type` ( `mstype.float16` or `mstype.float32` ), but not to be casted
            to the type determined by `level` setting.
        keep_batchnorm_fp32 (bool): Keep Batchnorm run in `float32` when the network is set to cast to `float16` . If
            set, the `level` setting will take no effect on this property.
        loss_scale_manager (Union[None, LossScaleManager]): If None, not scale the loss, otherwise scale the loss by
            `LossScaleManager` . If set, the `level` setting will take no effect on this property.
    Raises:
        ValueError: Auto mixed precision only supported on device GPU and Ascend. If device is CPU, a `ValueError`
            exception will be raised.
        ValueError: If device is CPU, property `loss_scale_manager` only can be set as `None` or `FixedLossScaleManager`
            (with property `drop_overflow_update=False` ), or a `ValueError` exception will be raised.
    """
    validator.check_value_type('network', network, nn.Cell)
    validator.check_value_type('optimizer', optimizer, (nn.Optimizer, boost.FreezeOpt))
    if not isinstance(level, str):
        raise TypeError("The argument `level` must be a string in ['O0', 'O2', 'O3', 'auto'], \
                         but got type {}.".format(type(level)))
    validator.check('level', level, "", ['O0', 'O2', 'O3', 'auto'], Rel.IN)
    validator.check('boost_level', boost_level, "", ['O0', 'O1', 'O2'], Rel.IN)

    if level == "auto":
        device_target = context.get_context('device_target')
        if device_target == "GPU":
            level = "O2"
        elif device_target == "Ascend":
            level = "O3"
        else:
            raise ValueError("Level `auto` only support when `device_target` is GPU or Ascend.")

    _check_kwargs(kwargs)
    config = dict(_config_level[level], **kwargs)

    if config["cast_model_type"] == mstype.float16:
        network.to_float(mstype.float16)

        if config["keep_batchnorm_fp32"]:
            _do_keep_batchnorm_fp32(network)

    if loss_fn:
        network = _add_loss_network(network, loss_fn, config["cast_model_type"])

    if _get_parallel_mode() in (ParallelMode.SEMI_AUTO_PARALLEL, ParallelMode.AUTO_PARALLEL):
        network = _VirtualDatasetCell(network)

    enable_boost = False
    if boost_level in ["O1", "O2"]:
        enable_boost = True

    loss_scale = 1.0
    if config["loss_scale_manager"] is not None:
        loss_scale_manager = config["loss_scale_manager"]
        loss_scale = loss_scale_manager.get_loss_scale()
        update_cell = loss_scale_manager.get_update_cell()
        if update_cell is not None:
            # only cpu not support `TrainOneStepWithLossScaleCell` for control flow.
            if not context.get_context("enable_ge") and context.get_context("device_target") == "CPU":
                raise ValueError("Only `loss_scale_manager=None` or "
                                 "`loss_scale_manager=FixedLossScaleManager(drop_overflow_update=False)`"
                                 "are supported on device `CPU`. ")
            if _get_pipeline_stages() > 1:
                network = _TrainPipelineWithLossScaleCell(network, optimizer,
                                                          scale_sense=update_cell).set_train()
            elif enable_boost:
                network = boost.BoostTrainOneStepWithLossScaleCell(network, optimizer,
                                                                   scale_sense=update_cell).set_train()
            else:
                network = nn.TrainOneStepWithLossScaleCell(network, optimizer,
                                                           scale_sense=update_cell).set_train()
            return network
    if _get_pipeline_stages() > 1:
        network = _TrainPipelineAccuStepCell(network, optimizer).set_train()
    elif enable_boost:
        network = boost.BoostTrainOneStepCell(network, optimizer, loss_scale).set_train()
    else:
        network = nn.TrainOneStepCell(network, optimizer, loss_scale).set_train()
    return network
