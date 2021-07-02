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
"""HarDNet"""
import mindspore.nn as nn
from mindspore.ops import operations as P

class GlobalAvgpooling(nn.Cell):
    """
    GlobalAvgpooling function
    """
    def __init__(self):
        super(GlobalAvgpooling, self).__init__()
        self.mean = P.ReduceMean(True)
        self.shape = P.Shape()
        self.reshape = P.Reshape()

    def construct(self, x):
        x = self.mean(x, (2, 3))
        b, c, _, _ = self.shape(x)
        x = self.reshape(x, (b, c))

        return x

class _ConvLayer(nn.Cell):
    """
    convlayer
    """
    def __init__(self, in_channels, out_channels, kernel=3, stride=1, dropout=0.9, bias=False):
        super(_ConvLayer, self).__init__()
        self.ConvLayer_Conv = nn.Conv2d(in_channels, out_channels,
                                        kernel_size=kernel,
                                        stride=stride,
                                        has_bias=bias,
                                        padding=kernel // 2,
                                        pad_mode="pad")
        self.ConvLayer_BN = nn.BatchNorm2d(out_channels)
        self.ConvLayer_RE = nn.ReLU6()

    def construct(self, x):
        out = self.ConvLayer_Conv(x)
        out = self.ConvLayer_BN(out)
        out = self.ConvLayer_RE(out)

        return out

class _DWConvLayer(nn.Cell):
    """
    dwconvlayer
    """
    def __init__(self, in_channels, out_channels, stride=1, bias=False):
        super(_DWConvLayer, self).__init__()

        self.DWConvLayer_Conv = nn.Conv2d(in_channels, in_channels,
                                          kernel_size=3,
                                          stride=stride,
                                          has_bias=bias,
                                          padding=1,
                                          pad_mode="pad")
        self.DWConvLayer_BN = nn.BatchNorm2d(in_channels)

    def construct(self, x):
        out = self.DWConvLayer_Conv(x)
        out = self.DWConvLayer_BN(out)

        return out

class _CombConvLayer(nn.Cell):
    """
    combconvlayer
    """
    def __init__(self, in_channels, out_channels, kernel=1, stride=1, dropout=0.9, bias=False):
        super(_CombConvLayer, self).__init__()
        self.CombConvLayer_Conv = _ConvLayer(in_channels, out_channels, kernel=kernel)
        self.CombConvLayer_DWConv = _DWConvLayer(out_channels, out_channels, stride=stride)

    def construct(self, x):
        out = self.CombConvLayer_Conv(x)
        out = self.CombConvLayer_DWConv(out)

        return out

class _HarDBlock(nn.Cell):
    """the HarDBlock function"""
    def get_link(self, layer, bash_ch, growth_rate, grmul):
        """
        link all layers
        """
        if layer == 0:
            return bash_ch, 0, []
        out_channels = growth_rate
        link = []
        for i in range(10):
            dv = 2 ** i
            if layer % dv == 0:
                k = layer - dv
                link.append(k)
                if i > 0:
                    out_channels *= grmul
        out_channels = int(int(out_channels + 1) / 2) * 2
        in_channels = 0
        for i in link:
            ch, _, _ = self.get_link(i, bash_ch, growth_rate, grmul)
            in_channels += ch

        return out_channels, in_channels, link

    def get_out_ch(self):
        return self.out_channels

    def __init__(self, in_channels, growth_rate, grmul, n_layers, keepBase=False, residual_out=False, dwconv=False):
        super(_HarDBlock, self).__init__()
        self.keepBase = keepBase
        self.links = []
        self.layer_list = nn.CellList()
        self.out_channels = 0

        for i in range(n_layers):
            outch, inch, link = self.get_link(i + 1, in_channels, growth_rate, grmul)
            self.links.append(link)
            if dwconv:
                layer = _CombConvLayer(inch, outch)
                self.layer_list.append(layer)
            else:
                layer = _ConvLayer(inch, outch)
                self.layer_list.append(layer)

            if (i % 2 == 0) or (i == n_layers - 1):
                self.out_channels += outch

        self.concate = P.Concat(axis=1)

    def construct(self, x):
        """"
        construct all parameters
        """
        layers_ = [x]

        for layer in range(len(self.layer_list)):
            link = self.links[layer]
            tin = []
            for i in link:
                tin.append(layers_[i])
            if len(tin) > 1:
                input_ = tin[0]
                for j in range(len(tin) - 1):
                    input_ = self.concate((input_, tin[j + 1]))
            else:
                input_ = tin[0]

            out = self.layer_list[layer](input_)
            layers_.append(out)

        t = len(layers_)
        out_ = []
        for j in range(t):
            if (j == 0 and self.keepBase) or (j == t - 1) or (j % 2 == 1):
                out_.append(layers_[j])

        output = out_[0]
        for k in range(len(out_) - 1):
            output = self.concate((output, out_[k + 1]))

        return output

class _CommenHead(nn.Cell):
    """
    the transition layer
    """
    def __init__(self, num_classes, out_channels, keep_rate):
        super(_CommenHead, self).__init__()
        self.avgpool = GlobalAvgpooling()
        self.flat = nn.Flatten()
        self.drop = nn.Dropout(keep_prob=keep_rate)
        self.dense = nn.Dense(out_channels, num_classes, has_bias=True)

    def construct(self, x):
        x = self.avgpool(x)
        x = self.flat(x)
        x = self.drop(x)
        x = self.dense(x)

        return x

class HarDNet(nn.Cell):
    """
    the HarDNet layers
    """
    __constants__ = ['layers']
    def __init__(self, depth_wise=False, arch=68, pretrained=False):
        super(HarDNet, self).__init__()
        first_ch = [32, 64]
        second_kernel = 3
        max_pool = True
        grmul = 1.7
        keep_rate = 0.9

        # HarDNet68
        ch_list = [128, 256, 320, 640, 1024]
        gr = [14, 16, 20, 40, 160]
        n_layers = [8, 16, 16, 16, 4]
        downSamp = [1, 0, 1, 1, 0]

        if arch == 85:
            # HarDNet85
            first_ch = [48, 96]
            ch_list = [192, 256, 320, 480, 720, 1280]
            gr = [24, 24, 28, 36, 48, 256]
            n_layers = [8, 16, 16, 16, 16, 4]
            downSamp = [1, 0, 1, 0, 1, 0]
            keep_rate = 0.8
        elif arch == 39:
            # HarDNet39
            first_ch = [24, 48]
            ch_list = [96, 320, 640, 1024]
            grmul = 1.6
            gr = [16, 20, 64, 160]
            n_layers = [4, 16, 8, 4]
            downSamp = [1, 1, 1, 0]

        if depth_wise:
            second_kernel = 1
            max_pool = False
            keep_rate = 0.95

        blks = len(n_layers)
        self.layers = nn.CellList()
        self.layers.append(_ConvLayer(3, first_ch[0], kernel=3, stride=2, bias=False))
        self.layers.append(_ConvLayer(first_ch[0], first_ch[1], kernel=second_kernel))

        if max_pool:
            self.layers.append(nn.MaxPool2d(kernel_size=3, stride=2))
        else:
            self.layers.append(_DWConvLayer(first_ch[1], first_ch[1], stride=2))

        ch = first_ch[1]
        for i in range(blks):
            blk = _HarDBlock(ch, gr[i], grmul, n_layers[i], dwconv=depth_wise)
            ch = blk.get_out_ch()
            self.layers.append(blk)

            if i == blks - 1 and arch == 85:
                self.layers.append(nn.Dropout(keep_prob=0.9))

            self.layers.append(_ConvLayer(ch, ch_list[i], kernel=1))
            ch = ch_list[i]
            if downSamp[i] == 1:
                if max_pool:
                    self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                else:
                    self.layers.append(_DWConvLayer(ch, ch, stride=2))
        self.out_channels = ch_list[blks - 1]
        self.keeprate = keep_rate

    def construct(self, x):
        for layer in self.layers:
            x = layer(x)

        return x

    def get_out_channels(self):
        return self.out_channels

    def get_keep_rate(self):
        return self.keeprate

class HarDNet68(nn.Cell):
    """
    hardnet68
    """
    def __init__(self, num_classes):
        super(HarDNet68, self).__init__()
        self.net = HarDNet(depth_wise=False, arch=68, pretrained=False)
        out_channels = self.net.get_out_channels()
        keep_rate = self.net.get_keep_rate()

        self.head = _CommenHead(num_classes, out_channels, keep_rate)

    def construct(self, x):
        x = self.net(x)
        x = self.head(x)

        return x

class HarDNet85(nn.Cell):
    """
    hardnet85
    """
    def __init__(self, num_classes):
        super(HarDNet85, self).__init__()
        self.net = HarDNet(depth_wise=False, arch=85, pretrained=False)
        out_channels = self.net.get_out_channels()
        keep_rate = self.net.get_keep_rate()

        self.head = _CommenHead(num_classes, out_channels, keep_rate)

    def construct(self, x):
        x = self.net(x)
        x = self.head(x)
        return x
