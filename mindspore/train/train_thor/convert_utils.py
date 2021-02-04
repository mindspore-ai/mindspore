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
convert utils for second order optimizer: thor
"""
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore import context


class ConvertNetUntils():
    """
    Convert net to thor layer net
    """
    def __init__(self):
        self._convert_method_map = {nn.Dense: self._convert_dense,
                                    nn.Embedding: self._convert_embedding,
                                    nn.Conv2d: self._convert_conv2d}


    def _convert_dense(self, subcell):
        """
        convert dense cell to second_order cell
        """

        weight = subcell.weight
        act_name = None
        if subcell.activation_flag:
            act_class = subcell.activation.__class__.__name__
            act_name = act_class.lower()
            if act_name == "fastgelu":
                act_name = "fast_gelu"
        if subcell.out_channels == 1001:
            new_subcell = nn.Dense_Thor(in_channels=subcell.in_channels,
                                        out_channels=subcell.out_channels,
                                        weight_init=weight,
                                        has_bias=subcell.has_bias,
                                        bias_init='zeros',
                                        activation=act_name)
        else:
            compute_type = mstype.float16
            if context.get_context("device_target") == "GPU":
                compute_type = mstype.float32
            new_subcell = nn.Dense_Thor(in_channels=subcell.in_channels,
                                        out_channels=subcell.out_channels,
                                        weight_init=weight,
                                        has_bias=subcell.has_bias,
                                        bias_init='zeros',
                                        activation=act_name).to_float(compute_type)

        if subcell.has_bias:
            new_subcell.bias = subcell.bias
        return new_subcell


    def _convert_embedding(self, subcell):
        """
        convert embedding cell to second_order cell
        """
        new_subcell = nn.Embedding_Thor(vocab_size=subcell.vocab_size,
                                        embedding_size=subcell.embedding_size,
                                        use_one_hot=False)
        new_subcell.embedding_table = subcell.embedding_table
        return new_subcell


    def _convert_conv2d(self, subcell):
        """
        convert conv2d cell to second_order cell
        """
        out_channel = subcell.out_channels
        in_channel = subcell.in_channels
        kernel_size = subcell.kernel_size[0]
        stride = subcell.stride
        padding = subcell.padding
        pad_mode = subcell.pad_mode
        has_bias = subcell.has_bias
        weight = subcell.weight
        new_subcell = nn.Conv2d_Thor(in_channel, out_channel,
                                     kernel_size=kernel_size, stride=stride, padding=padding, pad_mode=pad_mode,
                                     has_bias=has_bias, weight_init=weight)
        return new_subcell


    def _convert_to_thor_net(self, net):
        """
        convert net to thor net
        """
        cells = net.name_cells()
        change = False
        for name in cells:
            subcell = cells[name]
            if subcell == net:
                continue
            elif isinstance(subcell, (nn.Dense_Thor, nn.Conv2d_Thor, nn.Embedding_Thor)):
                continue
            elif isinstance(subcell, (nn.Conv2dTranspose, nn.Conv1d, nn.Conv1dTranspose, nn.BatchNorm1d, nn.GroupNorm,
                                      nn.GlobalBatchNorm, nn.LayerNorm, nn.BatchNorm2d, nn.MaxPool2d)):
                continue
            elif isinstance(subcell, (nn.Embedding, nn.Dense, nn.Conv2d)):
                prefix = subcell.param_prefix
                new_subcell = self._convert_method_map[type(subcell)](subcell)
                print("subcell name: ", name, "prefix is", prefix, flush=True)
                if isinstance(new_subcell, (nn.Dense_Thor, nn.Embedding_Thor, nn.Conv2d_Thor)):
                    print("convert to thor layer success.", flush=True)
                new_subcell.update_parameters_name(prefix + '.')
                net.insert_child_to_cell(name, new_subcell)
                change = True
            else:
                self._convert_to_thor_net(subcell)

        if isinstance(net, nn.SequentialCell) and change:
            print("is nn.SequentialCell and change")
            net.cell_list = list(net.cells())


    def convert_to_thor_net(self, net):
        """
        api for convert net to thor net
        """
        net.update_cell_prefix()
        self._convert_to_thor_net(net)
        net.update_cell_type("second_order")


class ConvertModelUtils():
    """
    convert model to thor model utils
    """

    def convert_to_thor_model(self, model, network, loss_fn=None, optimizer=None, metrics=None, amp_level="O0",
                              loss_scale_manager=None, keep_batchnorm_fp32=False, frequency=834):

        """
        api for convert model to thor model
        """
        optim_name = type(optimizer).__name__
        if optim_name in ("THOR_Ascend", "THOR_GPU"):
            from .model_thor import Model_Thor
            if isinstance(network, nn.TrainOneStepCell):
                model = Model_Thor(network=network, frequency=frequency)
            else:
                model = Model_Thor(network=network, loss_fn=loss_fn, optimizer=optimizer, amp_level=amp_level,
                                   loss_scale_manager=loss_scale_manager,
                                   keep_batchnorm_fp32=keep_batchnorm_fp32, metrics=metrics, frequency=frequency)

        return model
