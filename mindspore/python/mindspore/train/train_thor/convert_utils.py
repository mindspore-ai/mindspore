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
"""Conversion interface for second-order optimizer thor."""
from __future__ import absolute_import

import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore import context


class ConvertNetUtils:
    """
    Convert net to thor layer net, used to compute and store second-order information matrix.

    Examples:
        >>> import mindspore as ms
        >>> convert_net_utils = ms.train.ConvertNetUtils()
    """
    def __init__(self):
        self._convert_method_map = {nn.Dense: ConvertNetUtils._convert_dense,
                                    nn.Embedding: ConvertNetUtils._convert_embedding,
                                    nn.Conv2d: ConvertNetUtils._convert_conv2d,
                                    nn.EmbeddingLookup: ConvertNetUtils._convert_embeddinglookup}


    @staticmethod
    def _convert_dense(subcell):
        """
        Convert dense cell to second-order cell
        """
        weight = subcell.weight
        act_name = None
        if subcell.activation_flag:
            act_class = subcell.activation.__class__.__name__
            act_name = act_class.lower()
            if act_name == "fastgelu":
                act_name = "fast_gelu"
        if subcell.out_channels == 1001:
            new_subcell = nn.DenseThor(in_channels=subcell.in_channels,
                                       out_channels=subcell.out_channels,
                                       weight_init=weight,
                                       has_bias=subcell.has_bias,
                                       bias_init='zeros',
                                       activation=act_name)
        else:
            compute_type = mstype.float16
            if context.get_context("device_target") == "GPU":
                compute_type = mstype.float32
            new_subcell = nn.DenseThor(in_channels=subcell.in_channels,
                                       out_channels=subcell.out_channels,
                                       weight_init=weight,
                                       has_bias=subcell.has_bias,
                                       bias_init='zeros',
                                       activation=act_name).to_float(compute_type)

        if subcell.has_bias:
            new_subcell.bias = subcell.bias
        return new_subcell


    @staticmethod
    def _convert_embedding(subcell):
        """
        Convert embedding cell to second-order cell
        """
        new_subcell = nn.EmbeddingThor(vocab_size=subcell.vocab_size,
                                       embedding_size=subcell.embedding_size,
                                       use_one_hot=False)
        new_subcell.embedding_table = subcell.embedding_table
        return new_subcell


    @staticmethod
    def _convert_embeddinglookup(subcell):
        """
        convert embedding cell to second_order cell
        """
        new_subcell = nn.EmbeddingLookupThor(vocab_size=subcell.vocab_size,
                                             embedding_size=subcell.embedding_size,
                                             target=subcell.target, sparse=subcell.sparse,
                                             vocab_cache_size=subcell.vocab_cache_size)
        new_subcell.embedding_table = subcell.embedding_table
        return new_subcell


    @staticmethod
    def _convert_conv2d(subcell):
        """
        Convert conv2d cell to second-order cell
        """
        out_channel = subcell.out_channels
        in_channel = subcell.in_channels
        kernel_size = subcell.kernel_size[0]
        stride = subcell.stride
        padding = subcell.padding
        pad_mode = subcell.pad_mode
        has_bias = subcell.has_bias
        weight = subcell.weight

        new_subcell = nn.Conv2dThor(in_channel, out_channel,
                                    kernel_size=kernel_size, stride=stride, padding=padding, pad_mode=pad_mode,
                                    has_bias=has_bias, weight_init=weight)
        return new_subcell

    @staticmethod
    def _need_change(subcell, prefix):
        """for thor layers, need to change"""
        if isinstance(subcell, (nn.Dense, nn.Conv2d)) and subcell.weight.requires_grad:
            if "rpn_with_loss.rpn_convs_list." in prefix.lower() or "wide" in prefix.lower():
                return False
            return True
        if isinstance(subcell, (nn.Embedding, nn.EmbeddingLookup)) and subcell.embedding_table.requires_grad:
            return True
        return False

    def _convert_to_thor_net(self, net):
        """
        Convert net to thor net
        """
        cells = net.name_cells()
        change = False
        for name in cells:
            subcell = cells[name]
            if subcell == net:
                continue
            elif isinstance(subcell, (nn.DenseThor, nn.Conv2dThor, nn.EmbeddingThor, nn.EmbeddingLookupThor)):
                continue
            elif isinstance(subcell, (nn.Conv2dTranspose, nn.Conv1d, nn.Conv1dTranspose, nn.BatchNorm1d, nn.GroupNorm,
                                      nn.LayerNorm, nn.BatchNorm2d, nn.MaxPool2d)):
                continue
            elif isinstance(subcell, (nn.Embedding, nn.Dense, nn.Conv2d, nn.EmbeddingLookup)):
                prefix = subcell.param_prefix
                if self._need_change(subcell, prefix):
                    new_subcell = self._convert_method_map.get(type(subcell))(subcell)
                    new_subcell.update_parameters_name(prefix + '.')
                    net.insert_child_to_cell(name, new_subcell)
                    change = True
            else:
                self._convert_to_thor_net(subcell)

        if isinstance(net, nn.SequentialCell) and change:
            net.cell_list = list(net.cells())


    def convert_to_thor_net(self, net):
        """
        This interface is used to convert a network to thor layer network, in order to calculate and store the
        second-order information matrix.

        Note:
            This interface is automatically called by the second-order optimizer thor.

        Args:
            net (Cell): Network to be trained by the second-order optimizer thor.

        Supported Platforms:
            ``Ascend`` ``GPU``

        Examples:
            >>> import mindspore as ms
            >>> from mindspore import nn
            >>> from mindspore import Tensor
            >>> from mindspore.nn import thor
            >>>
            >>> # Define the network structure of LeNet5. Refer to
            >>> # https://gitee.com/mindspore/docs/blob/master/docs/mindspore/code/lenet.py
            >>> net = LeNet5()
            >>> temp = Tensor([4e-4, 1e-4, 1e-5, 1e-5], ms.float32)
            >>> opt = thor(net, learning_rate=temp, damping=temp, momentum=0.9, loss_scale=128, frequency=4)
            >>> loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
            >>> loss_scale = ms.FixedLossScaleManager(128, drop_overflow_update=False)
            >>> model = ms.Model(net, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale, metrics={'acc'},
            ...               amp_level="O2", keep_batchnorm_fp32=False)
            >>> model = ms.train.ConvertModelUtils.convert_to_thor_model(model=model, network=net, loss_fn=loss,
            ...                                           optimizer=opt,loss_scale_manager=loss_scale, metrics={'acc'},
            ...                                           amp_level="O2", keep_batchnorm_fp32=False)
        """

        net.update_cell_prefix()
        self._convert_to_thor_net(net)
        net.update_cell_type("second-order")


class ConvertModelUtils:
    """
    Convert model to thor model.
    """
    @staticmethod
    def convert_to_thor_model(model, network, loss_fn=None, optimizer=None, metrics=None, amp_level="O0",
                              loss_scale_manager=None, keep_batchnorm_fp32=False):
        """
        This interface is used to convert model to thor model.

        Args:
            model (Object): High-Level API for Training.
            network (Cell): A training network.
            loss_fn (Cell): Objective function. Default: ``None`` .
            optimizer (Cell): Optimizer used to updating the weights. Default: ``None`` .
            metrics (Union[dict, set]): A Dictionary or a set of metrics to be evaluated by the model during
                                        training. eg: {'accuracy', 'recall'}. Default: ``None`` .
            amp_level (str): Level for mixed precision training. Supports ["O0", "O2", "O3", "auto"].
                Default: ``"O0"`` .

                - O0: Do not change.
                - O2: Cast network to float16, keep batchnorm run in float32, using dynamic loss scale.
                - O3: Cast network to float16, with additional property 'keep_batchnorm_fp32=False'.
                - auto: Set level to recommended level in different devices. O2 is recommended on GPU, O3 is
                  recommended on Ascend. The recommended level is based on the expert experience, cannot
                  always generalize. User should specify the level for special network.

            loss_scale_manager (Union[None, LossScaleManager]): If it is None, the loss would not be scaled.
                Otherwise, scale the loss by LossScaleManager and optimizer can not be None. It is a key argument.
                e.g. Use `loss_scale_manager=None` to set the value. Default: ``None`` .
            keep_batchnorm_fp32 (bool): Keep Batchnorm running in `float32`. If True, the level setting before
                will be overwritten. Default: ``False`` .

        Returns:
             model (Object), High-Level API for Training.

        Supported Platforms:
            ``Ascend`` ``GPU``

        Examples:
            >>> import mindspore as ms
            >>> from mindspore import nn
            >>> from mindspore import Tensor
            >>> from mindspore.nn import thor
            >>>
            >>> # Define the network structure of LeNet5. Refer to
            >>> # https://gitee.com/mindspore/docs/blob/master/docs/mindspore/code/lenet.py
            >>> net = LeNet5()
            >>> # Create the dataset taking MNIST as an example. Refer to
            >>> # https://gitee.com/mindspore/docs/blob/master/docs/mindspore/code/mnist.py
            >>> dataset = create_dataset()
            >>> temp = Tensor([4e-4, 1e-4, 1e-5, 1e-5], ms.float32)
            >>> opt = thor(net, learning_rate=temp, damping=temp, momentum=0.9, loss_scale=128, frequency=4)
            >>> loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
            >>> loss_scale = ms.amp.FixedLossScaleManager(128, drop_overflow_update=False)
            >>> model = ms.train.Model(net, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale, metrics={'acc'},
            ...               amp_level="O2", keep_batchnorm_fp32=False)
            >>> model = ms.train.ConvertModelUtils.convert_to_thor_model(model=model, network=net, loss_fn=loss,
            ...                                          optimizer=opt, loss_scale_manager=loss_scale, metrics={'acc'},
            ...                                          amp_level="O2", keep_batchnorm_fp32=False)
        """

        optim_name = type(optimizer).__name__
        if optim_name in ("ThorAscend", "ThorGpu"):
            from mindspore.train.train_thor.model_thor import ModelThor
            if isinstance(network, nn.TrainOneStepCell):
                model = ModelThor(network=network)
            else:
                model = ModelThor(network=network, loss_fn=loss_fn, optimizer=optimizer, amp_level=amp_level,
                                  loss_scale_manager=loss_scale_manager,
                                  keep_batchnorm_fp32=keep_batchnorm_fp32, metrics=metrics)

        return model
