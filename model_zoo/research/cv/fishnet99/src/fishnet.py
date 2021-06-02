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
FishNet model of MindSpore-1.2.0.
"""
import mindspore.nn as nn
import mindspore.ops.operations as P

conv_weight_init = 'HeUniform'


class adaptiveavgpool2d_ms(nn.Cell):
    """adaptiveavgpool2d_ms"""
    def __init__(self):
        super(adaptiveavgpool2d_ms, self).__init__()
        self.ada_pool = P.ReduceMean(keep_dims=True)

    def construct(self, x):
        """construct"""
        x = self.ada_pool(x, (2, 3))
        return x


def _bn_relu_conv(in_c, out_c, **conv_kwargs):
    return nn.SequentialCell([nn.BatchNorm2d(num_features=in_c, momentum=0.9),
                              nn.ReLU(),
                              nn.Conv2d(in_channels=in_c, out_channels=out_c, pad_mode='pad',
                                        weight_init=conv_weight_init, **conv_kwargs),
                              ])


class ResBlock_with_shortcut(nn.Cell):
    """
    Construct Basic Bottle-necked Residual Block module.
    Args:
        in_c : Number of channels in the input image
        out_c : Number of channels in the output image
        shortcut : Specific function for skip-connection
            Examples)
                'bn_relu_conv' for DownRefinementBlock
                'bn_relu_conv with channel reduction' for UpRefinementBlock
                'identity mapping' for regular connection
        stride : Stride of middle conv layer
        dilation : Dilation rate of middle conv layer
    Forwarding Path:
                ⎡        (shortcut)         ⎤
        input image - (BN-ReLU-Conv) * 3 - (add) -output
    """
    def __init__(self, in_c, out_c, shortcut, stride=1, dilation=1):
        super(ResBlock_with_shortcut, self).__init__()

        mid_c = out_c // 4
        self.layers = nn.SequentialCell([
            _bn_relu_conv(in_c, mid_c, kernel_size=1, has_bias=False),
            _bn_relu_conv(mid_c, mid_c, kernel_size=3, stride=stride, padding=dilation, dilation=dilation,
                          has_bias=False),
            _bn_relu_conv(mid_c, out_c, kernel_size=1, has_bias=False),
        ])
        self.shortcut = shortcut
        self.add_1 = P.Add()

    def construct(self, x):
        """construct"""
        return self.add_1(self.layers(x), self.shortcut(x))


class ResBlock_without_shortcut(nn.Cell):
    """
    Construct Basic Bottle-necked Residual Block module.
    Args:
        in_c : Number of channels in the input image
        out_c : Number of channels in the output image
        stride : Stride of middle conv layer
        dilation : Dilation rate of middle conv layer
    Forwarding Path:
                ⎡        (shortcut)         ⎤
        input image - (BN-ReLU-Conv) * 3 - (add) -output
    """
    def __init__(self, in_c, out_c, stride=1, dilation=1):
        super(ResBlock_without_shortcut, self).__init__()

        mid_c = out_c // 4
        self.layers_ = nn.SequentialCell([
            _bn_relu_conv(in_c, mid_c, kernel_size=1, has_bias=False),
            _bn_relu_conv(mid_c, mid_c, kernel_size=3, stride=stride, padding=dilation, dilation=dilation,
                          has_bias=False),
            _bn_relu_conv(mid_c, out_c, kernel_size=1, has_bias=False),
        ])
        self.add_2 = P.Add()

    def construct(self, x):
        """construct"""
        return self.add_2(self.layers_(x), x)


class TransferBlock(nn.Cell):
    """
    Construct Transfer Block module.
    Args:
        ch : Number of channels in the input and output image
        num_blk : Number of Residual Blocks
    Forwarding Path:
        input image - (ResBlock_without_shortcut) * num_blk - output
    """
    def __init__(self, ch, num_blk):
        super(TransferBlock, self).__init__()

        self.layers_TransferBlock = nn.SequentialCell([*[ResBlock_without_shortcut(ch, ch)
                                                         for _ in range(0, num_blk)]])

    def construct(self, x):
        """construct"""
        return self.layers_TransferBlock(x)


class DownStage(nn.Cell):
    """
    Construct a stage for each resolution.
    A DownStage is consisted of one DownRefinementBlock and several residual regular connection blocks.
    (Note: In fact, DownRefinementBlock is not used in FishHead according to original implementation.
           However, it seems needed to be used according to original paper.
           In this version, we followed original implementation, not original paper.)
    Args:in_c : Number of channels in the input image
        out_c : Number of channels in the output image
        num_blk : Number of Residual Blocks
        stride : Stride of shortcut conv layer
    Forwarding Path:input image - (ResBlock with Shortcut) - (ResBlock) * num_blk - (MaxPool) - output
    """
    def __init__(self, in_c, out_c, num_blk, stride=1):
        super(DownStage, self).__init__()

        shortcut = _bn_relu_conv(in_c, out_c, kernel_size=1, stride=stride, has_bias=False)
        self.layer_DownStage = nn.SequentialCell([
            ResBlock_with_shortcut(in_c, out_c, shortcut),
            *[ResBlock_without_shortcut(out_c, out_c) for _ in range(1, num_blk)],
            nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='valid')
        ])

    def construct(self, x):
        """construct"""
        return self.layer_DownStage(x)


class UpStage(nn.Cell):
    """
    Construct a stage for each resolution.
    A DownStage is consisted of one DownRefinementBlock and several residual regular connection blocks.
    Not like DownStage, this module reduces the number of channels of concatenated feature maps in the shortcut path.
    Args:in_c : Number of channels in the input image
        out_c : Number of channels in the output image
        num_blk : Number of Residual Blocks
        stride : Stride of shortcut conv layer
    Forwarding Path:input image - (ResBlock with Channel Reduction) - (ResBlock) * num_blk - (UpSample) - output
    """
    def __init__(self, in_c, out_c, num_blk, dilation=1):
        super(UpStage, self).__init__()

        self.k = in_c // out_c
        self.redece_sum = P.ReduceSum(keep_dims=False)
        self.shape_ = P.Shape()
        self.reshape_ = P.Reshape()
        self.layer_UpStage = nn.SequentialCell([
            ResBlock_with_shortcut(in_c, out_c, channel_reduction_ms(in_c // out_c), dilation=dilation),
            *[ResBlock_without_shortcut(out_c, out_c, dilation=dilation) for _ in range(1, num_blk)],
        ])

    def construct(self, x):
        """construct"""
        return self.layer_UpStage(x)


class channel_reduction_ms(nn.Cell):
    """channel_reduction_ms"""
    def __init__(self, kk):
        super(channel_reduction_ms, self).__init__()
        self.shape_ = P.Shape()
        self.kk = kk
        self.reshape_ = P.Reshape()
        self.redece_sum = P.ReduceSum(keep_dims=False)

    def construct(self, x):
        """construct"""
        n, c, h_, w_ = self.shape_(x)
        x = self.redece_sum(self.reshape_(x, (n, c // self.kk, self.kk, h_, w_)), 2)
        return x


class FishTail(nn.Cell):
    """
    Construct FishTail module.
    Each instances corresponds to each stages.
    Args:
        in_c : Number of channels in the input image
        out_c : Number of channels in the output image
        num_blk : Number of Residual Blocks
    Forwarding Path:
        input image - (DownStage) - output
    """
    def __init__(self, in_c, out_c, num_blk):
        super(FishTail, self).__init__()

        self.layer_FishTail = DownStage(in_c, out_c, num_blk)

    def construct(self, x):
        """construct"""
        x = self.layer_FishTail(x)
        return x


class Bridge(nn.Cell):
    """
    Construct Bridge module.
    This module bridges the last FishTail stage and first FishBody stage.
    Args:ch : Number of channels in the input and output image
        num_blk : Number of Residual Blocks
    Forwarding Path:
                        r                        (SEBlock)                           ㄱ
        input image - (stem) - (ResBlock with Shortcut) - (ResBlock) * num_blk - (mul & sum) - output
    """
    def __init__(self, ch, num_blk):
        super(Bridge, self).__init__()

        self.stem = nn.SequentialCell([
            nn.BatchNorm2d(num_features=ch, momentum=0.9),
            nn.ReLU(),
            nn.Conv2d(in_channels=ch, out_channels=ch // 2, kernel_size=1, pad_mode='pad', has_bias=False,
                      weight_init=conv_weight_init),
            nn.BatchNorm2d(num_features=ch // 2, momentum=0.9),
            nn.ReLU(),
            nn.Conv2d(in_channels=ch // 2, out_channels=ch * 2, kernel_size=1, pad_mode='pad', has_bias=True,
                      weight_init=conv_weight_init)
        ])
        shortcut = _bn_relu_conv(ch * 2, ch, kernel_size=1, has_bias=False)
        self.layers_Bridge = nn.SequentialCell([
            ResBlock_with_shortcut(ch * 2, ch, shortcut),
            *[ResBlock_without_shortcut(ch, ch) for _ in range(1, num_blk)],
        ])

        self.se_block = nn.SequentialCell([
            nn.BatchNorm2d(num_features=ch * 2, momentum=0.9),
            nn.ReLU(),
            adaptiveavgpool2d_ms(),
            nn.Conv2d(in_channels=ch * 2, out_channels=ch // 16, kernel_size=1, pad_mode='pad', has_bias=True
                      , weight_init=conv_weight_init),
            nn.ReLU(),
            nn.Conv2d(in_channels=ch // 16, out_channels=ch, kernel_size=1, pad_mode='pad', has_bias=True
                      , weight_init=conv_weight_init),
            nn.Sigmoid()
        ])
        self.add_3 = P.Add()
        self.mul_ = P.Mul()

    def construct(self, x):
        """construct"""
        x = self.stem(x)
        att = self.se_block(x)
        out = self.layers_Bridge(x)
        return self.add_3(self.mul_(out, att), att)


class FishBody_0(nn.Cell):
    """Construct FishBody module.
    Each instances corresponds to each stages.
    Args:in_c : Number of channels in the input image
        out_c : Number of channels in the output image
        num_blk : Number of Residual Blocks
        trans_in_c : Number of channels in the transferred image
        num_trans : Number of Transfer Blocks
        dilation : Dilation rate of Conv in UpRefinementBlock
    Forwarding Path:
        input image - (UpStage)       ㄱ
        trans image - (transfer) --(concat)-- output
    """
    def __init__(self, in_c, out_c, num_blk, trans_in_c, num_trans, dilation=1):
        super(FishBody_0, self).__init__()
        self.layer_FishBody = UpStage(in_c, out_c, num_blk, dilation=dilation)
        self.add_up = par_ms_0()
        self.transfer = TransferBlock(trans_in_c, num_trans)
        self.concat = P.Concat(1)

    def construct(self, x, trans_x):
        """construct"""
        x = self.layer_FishBody(x)
        x = self.add_up(x)
        trans_x = self.transfer(trans_x)
        return self.concat((x, trans_x))


class par_ms_0(nn.Cell):
    """par_ms_0"""
    def __init__(self):
        super(par_ms_0, self).__init__()
        self.P_up = P.ResizeNearestNeighbor((14, 14))

    def construct(self, x):
        """construct"""
        x = self.P_up(x)
        return x


class FishBody_1(nn.Cell):
    """Construct FishBody module.
    Each instances corresponds to each stages.
    Args:in_c : Number of channels in the input image
        out_c : Number of channels in the output image
        num_blk : Number of Residual Blocks
        trans_in_c : Number of channels in the transferred image
        num_trans : Number of Transfer Blocks
        dilation : Dilation rate of Conv in UpRefinementBlock
    Forwarding Path:
        input image - (UpStage)       ㄱ
        trans image - (transfer) --(concat)-- output
    """
    def __init__(self, in_c, out_c, num_blk, trans_in_c, num_trans, dilation=1):
        super(FishBody_1, self).__init__()
        self.layer_FishBody = UpStage(in_c, out_c, num_blk, dilation=dilation)
        self.add_up = par_ms_1()
        self.transfer = TransferBlock(trans_in_c, num_trans)
        self.concat = P.Concat(1)

    def construct(self, x, trans_x):
        """construct"""
        x = self.layer_FishBody(x)
        x = self.add_up(x)
        trans_x = self.transfer(trans_x)
        return self.concat((x, trans_x))


class par_ms_1(nn.Cell):
    """par_ms_1"""
    def __init__(self):
        super(par_ms_1, self).__init__()
        self.P_up = P.ResizeNearestNeighbor((28, 28))

    def construct(self, x):
        """construct"""
        x = self.P_up(x)
        return x


class FishBody_2(nn.Cell):
    """Construct FishBody module.
    Each instances corresponds to each stages.
    Args:in_c : Number of channels in the input image
        out_c : Number of channels in the output image
        num_blk : Number of Residual Blocks
        trans_in_c : Number of channels in the transferred image
        num_trans : Number of Transfer Blocks
        dilation : Dilation rate of Conv in UpRefinementBlock
    Forwarding Path:
        input image - (UpStage)       ㄱ
        trans image - (transfer) --(concat)-- output
    """
    def __init__(self, in_c, out_c, num_blk, trans_in_c, num_trans, dilation=1):
        super(FishBody_2, self).__init__()
        self.layer_FishBody = UpStage(in_c, out_c, num_blk, dilation=dilation)
        self.add_up = par_ms_2()
        self.transfer = TransferBlock(trans_in_c, num_trans)
        self.concat = P.Concat(1)

    def construct(self, x, trans_x):
        """construct"""
        x = self.layer_FishBody(x)
        x = self.add_up(x)
        trans_x = self.transfer(trans_x)
        return self.concat((x, trans_x))


class par_ms_2(nn.Cell):
    """par_ms_2"""
    def __init__(self):
        super(par_ms_2, self).__init__()
        self.P_up = P.ResizeNearestNeighbor((56, 56))

    def construct(self, x):
        """construct"""
        x = self.P_up(x)
        return x


class FishHead(nn.Cell):
    """Construct FishHead module.
    Each instances corresponds to each stages.
    Different with Official Code : we used shortcut layer in this Module. (shortcut layer is used according to the
    original paper)
    Args:in_c : Number of channels in the input image
        out_c : Number of channels in the output image
        num_blk : Number of Residual Blocks
        trans_in_c : Number of channels in the transferred image
        num_trans : Number of Transfer Blocks
    Forwarding Path:
        input image - (ResBlock) * num_blk - pool ㄱ
        trans image - (transfer)             --(concat)-- output
    """
    def __init__(self, in_c, out_c, num_blk, trans_in_c, num_trans):
        super(FishHead, self).__init__()

        self.layer_FishHead = nn.SequentialCell([
            ResBlock_without_shortcut(in_c, out_c),
            *[ResBlock_without_shortcut(out_c, out_c) for _ in range(1, num_blk)],
            nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='valid')
        ])
        self.transfer = TransferBlock(trans_in_c, num_trans)
        self.concat_ = P.Concat(1)

    def construct(self, x, trans_x):
        """construct"""
        x = self.layer_FishHead(x)
        trans_x = self.transfer(trans_x)
        return self.concat_((x, trans_x))


def _conv_bn_relu(in_ch, out_ch, stride=1):
    return nn.SequentialCell([nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=stride,
                                        pad_mode='pad', padding=1, has_bias=False, weight_init=conv_weight_init),
                              nn.BatchNorm2d(num_features=out_ch, momentum=0.9),
                              nn.ReLU()])


class Fishnet(nn.Cell):
    """
    Construct entire networks
    Args:
        start_c : Number of channels of input image.
                  Note that it is NOT the number of channels in initial input image, and it IS the number of output
                  channel of stem.
        num_cls : Number of classes
        tail_num_blk : list of the numbers of Conv blocks in each FishTail stages
        body_num_blk : list of the numbers of Conv blocks in each FishBody stages
        head_num_blk : list of the numbers of Conv blocks in each FishHead stages
            (Note : `*_num_blk` includes 1 Residual blocks in the start of each stages)
        body_num_trans : list of the numbers of Conv blocks in transfer paths in each FishTail stages
        head_num_trans : list of the numbers of Conv blocks in transfer paths in each FishHead stages
        tail_channels : list of the number of in, out channel of each stages
        body_channels : list of the number of in, out channel of each stages
        head_channels : list of the number of in, out channel of each stages
    """
    def __init__(self, start_c=64, num_cls=1000,
                 tail_num_blk=1, bridge_num_blk=2,
                 body_num_blk=1, body_num_trans=1,
                 head_num_blk=1, head_num_trans=1,
                 tail_channels=1, body_channels=1, head_channels=1):
        super(Fishnet, self).__init__()

        self.stem = nn.SequentialCell([
            _conv_bn_relu(3, start_c // 2, stride=2),
            _conv_bn_relu(start_c // 2, start_c // 2),
            _conv_bn_relu(start_c // 2, start_c),
            nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')
        ])
        print("FishNet Initialization Start")

        self.tail_layer = []
        for i, num_blk in enumerate(tail_num_blk):
            layer = FishTail(tail_channels[i], tail_channels[i + 1], num_blk)
            self.tail_layer.append(layer)
        self.tail_layer = nn.CellList(self.tail_layer)
        self.bridge = Bridge(tail_channels[-1], bridge_num_blk)

        self.body_layer = []
        for i, (num_blk, num_trans) in enumerate(zip(body_num_blk, body_num_trans)):
            if i == 0:
                layer = FishBody_0(body_channels[i][0], body_channels[i][1], num_blk,
                                   tail_channels[-i - 2], num_trans, dilation=2 ** i)
            elif i == 1:
                layer = FishBody_1(body_channels[i][0], body_channels[i][1], num_blk,
                                   tail_channels[-i - 2], num_trans, dilation=2 ** i)
            else:
                layer = FishBody_2(body_channels[i][0], body_channels[i][1], num_blk,
                                   tail_channels[-i - 2], num_trans, dilation=2 ** i)
            self.body_layer.append(layer)
        self.body_layer = nn.CellList(self.body_layer)

        self.head_layer = []
        for i, (num_blk, num_trans) in enumerate(zip(head_num_blk, head_num_trans)):
            layer = FishHead(head_channels[i][0], head_channels[i][1], num_blk,
                             body_channels[-i - 1][0], num_trans)
            self.head_layer.append(layer)
        self.head_layer = nn.CellList(self.head_layer)

        last_c = head_channels[-1][1]
        self.classifier = nn.SequentialCell([
            nn.BatchNorm2d(num_features=last_c, momentum=0.9),
            nn.ReLU(),
            nn.Conv2d(in_channels=last_c, out_channels=last_c // 2, kernel_size=1, pad_mode='pad', has_bias=False
                      , weight_init=conv_weight_init),
            nn.BatchNorm2d(num_features=last_c // 2, momentum=0.9),
            nn.ReLU(),
            adaptiveavgpool2d_ms(),
            nn.Conv2d(in_channels=last_c // 2, out_channels=num_cls, kernel_size=1, pad_mode='pad', has_bias=True
                      , weight_init=conv_weight_init)
        ])
        self.squeeze_ = P.Squeeze(2)

    def construct(self, x):
        """construct"""
        stem = self.stem(x)
        tail_features = [stem]
        for t in self.tail_layer:
            last_feature = tail_features[-1]
            tail_features.append(t(last_feature))

        bridge = self.bridge(tail_features[-1])

        body_features = [bridge]
        for b, tail in zip(self.body_layer, [tail_features[2], tail_features[1], tail_features[0]]):
            last_feature = body_features[-1]
            body_features.append(b(last_feature, tail))

        head_features = [body_features[-1]]
        for h, body in zip(self.head_layer, [body_features[2], body_features[1], body_features[0]]):
            last_feature = head_features[-1]
            head_features.append(h(last_feature, body))

        out = self.classifier(head_features[-1])
        out = self.squeeze_(out)
        out = self.squeeze_(out)
        return out


def _calc_channel(start_c, num_blk):
    """
    Calculate the number of in and out channels of each stages in FishNet.
    Example:
        fish150 : start channel=64, num_blk=3,
        tail channels : Grow double in each stages,
                        [64, 128, 256 ...] = [start channel ** (2**num_blk) ....]
        body channels : In first stage, in_channel and out_channel is the same,
                        but the other layers, the number of output channels is half of the number of input channel
                        Add the number of transfer channels to the number of output channels
                        The numbers of transfer channels are reverse of the tail channel[:-2]
                        [(512, 512), + 256
                         (768, 384), + 128
                         (512, 256)] + 64
        head channels : The number of input channels and output channels is the same.
                        Add the number of transfer channels to the number of output channels
                        The numbers of transfer channels are reverse of the tail channel[:-2]
                        [(320, 320),   + 512
                         (832, 832),   + 768
                         (1600, 1600)] + 512
    """
    # tail channels
    tail_channels = [start_c]
    for i in range(num_blk):
        tail_channels.append(tail_channels[-1] * 2)
    print("Tail Channels : ", tail_channels)

    # body channels
    in_c, transfer_c = tail_channels[-1], tail_channels[-2]
    body_channels = [(in_c, in_c), (in_c + transfer_c, (in_c + transfer_c) // 2)]
    for i in range(1, num_blk - 1):
        transfer_c = tail_channels[-i - 2]
        in_c = body_channels[-1][1] + transfer_c
        body_channels.append((in_c, in_c // 2))
    print("Body Channels : ", body_channels)

    # head channels
    in_c = body_channels[-1][1] + tail_channels[0]
    head_channels = [(in_c, in_c)]
    for i in range(num_blk):
        transfer_c = body_channels[-i - 1][0]
        in_c = head_channels[-1][1] + transfer_c
        head_channels.append((in_c, in_c))
    print("Head Channels : ", head_channels)
    return {"tail_channels": tail_channels, "body_channels": body_channels, "head_channels": head_channels}


def fish99(num_cls=1000):
    """fish99"""
    start_c = 64
    # tail
    tail_num_blk = [2, 2, 6]
    bridge_num_blk = 2
    # body
    body_num_blk = [1, 1, 1]
    body_num_trans = [1, 1, 1]
    # head
    head_num_blk = [1, 2, 2]
    head_num_trans = [1, 1, 4]

    net_channel = _calc_channel(start_c, len(tail_num_blk))

    return Fishnet(start_c, num_cls,
                   tail_num_blk, bridge_num_blk,
                   body_num_blk, body_num_trans,
                   head_num_blk, head_num_trans,
                   **net_channel)
