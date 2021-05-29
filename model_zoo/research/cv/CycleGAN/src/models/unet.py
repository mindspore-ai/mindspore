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
"""UNet Generator."""

import mindspore.nn as nn
import mindspore.ops as ops

class UnetGenerator(nn.Cell):
    """
    Unet-based generator.

    Args:
        in_planes (int): the number of channels in input images.
        out_planes (int): the number of channels in output images.
        ngf (int): the number of filters in the last conv layer.
        n_layers (int): the number of downsamplings in UNet.
        alpha (float): LeakyRelu slope. Default: 0.2.
        norm_mode (str): Specifies norm method. The optional values are "batch", "instance".
        dropout (bool): Use dropout or not. Default: False.

    Returns:
        Tensor, output tensor.
    """
    def __init__(self, in_planes, out_planes, ngf=64, n_layers=7, alpha=0.2, norm_mode='bn', dropout=False):
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, in_planes=None, submodule=None,
                                             norm_mode=norm_mode, innermost=True)
        for _ in range(n_layers - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, in_planes=None, submodule=unet_block,
                                                 norm_mode=norm_mode, dropout=dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, in_planes=None, submodule=unet_block,
                                             norm_mode=norm_mode)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, in_planes=None, submodule=unet_block,
                                             norm_mode=norm_mode)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, in_planes=None, submodule=unet_block, norm_mode=norm_mode)
        self.model = UnetSkipConnectionBlock(out_planes, ngf, in_planes=in_planes, submodule=unet_block,
                                             outermost=True, norm_mode=norm_mode)

    def construct(self, x):
        return self.model(x)


class UnetSkipConnectionBlock(nn.Cell):
    """Unet submodule with skip connection.

    Args:
        outer_nc (int): The number of filters in the outer conv layer
        inner_nc (int): The number of filters in the inner conv layer
        in_planes (int): The number of channels in input images/features
        dropout (bool): Use dropout or not. Default: False.
        submodule (Cell): Previously defined submodules
        outermost (bool): If this module is the outermost module
        innermost (bool): If this module is the innermost module
        alpha (float): LeakyRelu slope. Default: 0.2.
        norm_mode (str): Specifies norm method. The optional values are "batch", "instance".

    Returns:
        Tensor, output tensor.
    """
    def __init__(self, outer_nc, inner_nc, in_planes=None, dropout=False,
                 submodule=None, outermost=False, innermost=False, alpha=0.2, norm_mode='batch'):
        super(UnetSkipConnectionBlock, self).__init__()
        downnorm = nn.BatchNorm2d(inner_nc)
        upnorm = nn.BatchNorm2d(outer_nc)
        use_bias = False
        if norm_mode == 'instance':
            downnorm = nn.BatchNorm2d(inner_nc, affine=False)
            upnorm = nn.BatchNorm2d(outer_nc, affine=False)
            use_bias = True
        if in_planes is None:
            in_planes = outer_nc
        downconv = nn.Conv2d(in_planes, inner_nc, kernel_size=4,
                             stride=2, padding=1, has_bias=use_bias, pad_mode='pad')
        downrelu = nn.LeakyReLU(alpha)
        uprelu = nn.ReLU()

        if outermost:
            upconv = nn.Conv2dTranspose(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, pad_mode='pad')
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.Conv2dTranspose(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, has_bias=use_bias, pad_mode='pad')
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.Conv2dTranspose(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, has_bias=use_bias, pad_mode='pad')
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            model = down + [submodule] + up
            if dropout:
                model.append(nn.Dropout(0.5))

        self.model = nn.SequentialCell(model)
        self.skip_connections = not outermost
        self.concat = ops.Concat(axis=1)

    def construct(self, x):
        out = self.model(x)
        if self.skip_connections:
            out = self.concat((out, x))
        return out
