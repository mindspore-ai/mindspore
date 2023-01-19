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
""" test nn ops """
import numpy as np
import pytest

import mindspore
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor, Parameter
from mindspore.common.initializer import initializer
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops.operations import _grad_ops as G
from mindspore.ops import prim_attr_register, PrimitiveWithInfer
from mindspore._c_expression import security
from tests.security_utils import security_off_wrap
from ..ut_filter import non_graph_engine
from ....mindspore_test_framework.mindspore_test import mindspore_test
from ....mindspore_test_framework.pipeline.forward.compile_forward \
    import pipeline_for_compile_forward_ge_graph_for_case_by_case_config
from ....mindspore_test_framework.pipeline.forward.verify_exception \
    import pipeline_for_verify_exception_for_case_by_case_config
context.set_context(mode=context.GRAPH_MODE)


def conv3x3(in_channels, out_channels, stride=1, padding=1):
    """3x3 convolution """
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=3, stride=stride, padding=padding)


def conv1x1(in_channels, out_channels, stride=1, padding=0):
    """1x1 convolution"""
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=1, stride=stride, padding=padding)


grad = C.GradOperation()
grad_all_with_sens = C.GradOperation(get_all=True, sens_param=True)


class ResidualBlock(nn.Cell):
    """
    residual Block
    """
    expansion = 4

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 down_sample=False):
        super(ResidualBlock, self).__init__()

        out_chls = out_channels // self.expansion
        self.conv1 = conv1x1(in_channels, out_chls, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_chls)

        self.conv2 = conv3x3(out_chls, out_chls, stride=stride, padding=0)
        self.bn2 = nn.BatchNorm2d(out_chls)

        self.conv3 = conv1x1(out_chls, out_channels, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()
        self.downsample = down_sample

        self.conv_down_sample = conv1x1(in_channels, out_channels,
                                        stride=stride, padding=0)
        self.bn_down_sample = nn.BatchNorm2d(out_channels)
        self.add = P.Add()

    def construct(self, x):
        """
        :param x:
        :return:
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample:
            identity = self.conv_down_sample(identity)
            identity = self.bn_down_sample(identity)

        out = self.add(out, identity)
        out = self.relu(out)

        return out


class VirtualLossGrad(PrimitiveWithInfer):
    """ VirtualLossGrad definition """

    @prim_attr_register
    def __init__(self):
        """init VirtualLossGrad"""

    def __call__(self, x, out, dout):
        raise NotImplementedError

    def infer_shape(self, x_shape, out_shape, dout_shape):
        return x_shape

    def infer_dtype(self, x_dtype, out_dtype, dout_dtype):
        return x_dtype


class VirtualLoss(PrimitiveWithInfer):
    """ VirtualLoss definition """

    @prim_attr_register
    def __init__(self):
        """init VirtualLoss"""

    def __call__(self, x):
        raise NotImplementedError

    def get_bprop(self):
        loss_grad = VirtualLossGrad()

        def bprop(x, out, dout):
            # pylint: disable=unused-argument
            dx = loss_grad(x, out, dout)
            return (dx,)

        return bprop

    def infer_shape(self, x_shape):
        return []

    def infer_dtype(self, x_dtype):
        return x_dtype


class VirtualNetWithLoss(nn.Cell):
    """ VirtualNetWithLoss definition """

    def __init__(self, network):
        super(VirtualNetWithLoss, self).__init__()
        self.loss = VirtualLoss()
        self.network = network

    def construct(self, x):
        predict = self.network(x)
        return self.loss(predict)


class SoftMaxGrad(nn.Cell):
    """ SoftMaxGrad definition """

    def __init__(self, network):
        super(SoftMaxGrad, self).__init__()
        self.network = network

    def construct(self, x):
        return grad(self.network)(x)


class DropoutGrad(nn.Cell):
    """ DropoutGrad definition """

    def __init__(self, network):
        super(DropoutGrad, self).__init__()
        self.network = network

    def construct(self, x):
        return grad(self.network)(x)


class ScalarSummaryNet(nn.Cell):
    """ ScalarSummaryNet definition """

    def __init__(self):
        super(ScalarSummaryNet, self).__init__()
        self.summary = P.ScalarSummary()

    def construct(self, scalar):
        string_in = "bias_value"
        out = self.summary(string_in, scalar)
        return out


class L2NormalizeNet(nn.Cell):
    """ L2NormalizeNet definition """

    def __init__(self):
        super(L2NormalizeNet, self).__init__()
        self.l2_normalize = P.L2Normalize()

    def construct(self, x):
        out = self.l2_normalize(x)
        return out


class CosineSimilarityNet(nn.Cell):
    def __init__(self):
        super(CosineSimilarityNet, self).__init__()
        self.cosine_similarity = nn.layer.CosineSimilarity()

    def construct(self, x1, x2):
        return self.cosine_similarity(x1, x2)


class UpsampleNet(nn.Cell):
    def __init__(self):
        super(UpsampleNet, self).__init__()
        self.upsample = nn.Upsample(size=(5, 5))

    def construct(self, x):
        return self.upsample(x)


class PoissonNLLLossNet(nn.Cell):
    def __init__(self):
        super(PoissonNLLLossNet, self).__init__()
        self.poisson_nllloss = nn.PoissonNLLLoss()

    def construct(self, x, target):
        return self.poisson_nllloss(x, target)


class MultiLabelSoftMarginLossNet(nn.Cell):
    def __init__(self):
        super(MultiLabelSoftMarginLossNet, self).__init__()
        self.multi_label_soft_margin_loss = nn.MultiLabelSoftMarginLoss(reduction='mean')

    def construct(self, x, target):
        return self.multi_label_soft_margin_loss(x, target)


class HistogramSummaryNet(nn.Cell):
    """HistogramSummaryNet definition"""

    def __init__(self):
        super(HistogramSummaryNet, self).__init__()
        self.summary = P.HistogramSummary()

    def construct(self, tensor):
        string_in = "wight_value"
        out = self.summary(string_in, tensor)
        return out


class FusedBatchNormGrad(nn.Cell):
    """ FusedBatchNormGrad definition """

    def __init__(self, network):
        super(FusedBatchNormGrad, self).__init__()
        self.grad = C.GradOperation(get_all=True, sens_param=True)
        self.network = network

    def construct(self, inp, output_grad):
        return self.grad(self.network)(inp, output_grad)


class NetWithLoss(nn.Cell):
    """ NetWithLoss definition """

    def __init__(self, network):
        super(NetWithLoss, self).__init__()
        self.loss = P.SmoothL1Loss()
        self.network = network

    def construct(self, x, label):
        predict = self.network(x)
        return self.loss(predict, label)


class Grad(nn.Cell):
    """ GradWrap definition """

    def __init__(self, network):
        super(Grad, self).__init__()
        self.network = network
        self.network.set_train()

    def construct(self, x, label):
        return grad(self.network)(x, label)


class BatchnormNet(nn.Cell):
    """ BatchnormNet definition """

    def __init__(self):
        super(BatchnormNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 4, kernel_size=8, stride=2, pad_mode="pad", padding=3)
        self.bn1 = nn.BatchNorm2d(4)
        self.flatten = P.Flatten()
        self.weight = Parameter(Tensor(np.ones([64, 10], np.float32)), name="weight")
        self.bias = Parameter(Tensor(np.ones([10], np.float32)), name="bias")
        self.fc = P.MatMul()
        self.biasAdd = P.BiasAdd()

    def construct(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.flatten(x)
        x = self.biasAdd(self.fc(x, self.weight), self.bias)
        return x


class NetWithLossClass(nn.Cell):
    """ NetWithLossClass definition """

    def __init__(self, network):
        super(NetWithLossClass, self).__init__(auto_prefix=False)
        self.loss = nn.SoftmaxCrossEntropyWithLogits()
        self.network = network

    def construct(self, x, label):
        predict = self.network(x)
        if F.rank(predict) == 4:
            predict = predict[0, 0, :, :]
        return self.loss(predict, label)


class BlockNet(nn.Cell):
    """ BlockNet definition """

    def __init__(self):
        super(BlockNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, pad_mode="pad", padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.block_down_sample = ResidualBlock(
            64, 256, stride=1, down_sample=True
        )
        self.flatten = P.Flatten()
        self.weight = Parameter(Tensor(np.ones([1024, 10]).astype(np.float32)), name="weight")
        self.bias = Parameter(Tensor(np.ones([10]).astype((np.float32))), name="bias")
        self.fc = P.MatMul()
        self.biasAdd = P.BiasAdd()

    def construct(self, x):
        x = self.conv1(x)
        return x


class Conv2dWithBiasNet(nn.Cell):
    """ Conv2dWithBiasNet definition """

    def __init__(self):
        super(Conv2dWithBiasNet, self).__init__()
        self.conv = nn.Conv2d(3, 10, 1, bias_init='zeros')
        self.flatten = P.Flatten()

    def construct(self, input_x):
        return self.flatten(self.conv(input_x))


class Conv2dNativeNet(nn.Cell):
    """ Conv2dNativeNet definition """

    def __init__(self):
        super(Conv2dNativeNet, self).__init__()
        self.conv = P.DepthwiseConv2dNative(channel_multiplier=3, kernel_size=(3, 3))
        self.flatten = P.Flatten()
        channel_multipliers = 1
        in_channels = 3
        kernel_size = (3, 3)
        self.weight = Parameter(initializer(
            Tensor(np.ones([channel_multipliers, in_channels, *kernel_size], dtype=np.float32)),
            [channel_multipliers, in_channels, *kernel_size]), name='weight')

    def construct(self, input_x):
        return self.flatten(self.conv(input_x, self.weight))


class StateNet(nn.Cell):
    """ StateTestTensor definition """

    def __init__(self):
        super(StateNet, self).__init__()
        weight = Tensor(np.ones([2, 1, 2, 2], np.float32))
        self.s1 = Parameter(weight, name="s1")
        self.s2 = Parameter(weight, name="s2")
        self.sub = P.Sub()
        self.loss = nn.SoftmaxCrossEntropyWithLogits()
        self.assign = P.Assign()

    def construct(self, x):
        x = F.depend(x, self.assign(self.s1, x + self.s1))
        self.s1 = self.sub(self.s1, x)
        self.s2 = self.sub(self.s2, x)
        return x


def test_conv2d_same_primitive():
    class Conv2DSameNet(nn.Cell):
        def __init__(self):
            super(Conv2DSameNet, self).__init__()
            self.conv1 = nn.Conv2d(16, 64, (1, 41), (1, 4), "same", 0, 1, has_bias=True)
            self.conv2 = nn.Conv2d(16, 64, (1, 41), (1, 4), "same", 0, 1, has_bias=True)

        def construct(self, x, y):
            r1 = self.conv1(x)
            r2 = self.conv2(y)
            return (r1, r2)
    t1 = Tensor(np.ones([1, 16, 1, 1918]).astype(np.float32))
    t2 = Tensor(np.ones([1, 16, 1, 3840]).astype(np.float32))
    net = Conv2DSameNet()
    net(t1, t2)


class ComparisonNet(nn.Cell):
    def __init__(self, empty=None):
        """ ComparisonNet definition """
        super(ComparisonNet, self).__init__()
        self.empty = empty

    def construct(self, x, y):
        ret = x <= y
        return ret


def test_max_pool_with_arg_max():
    class NetMaxPoolWithArgMax(nn.Cell):
        def __init__(self):
            """ ComparisonNet definition """
            super(NetMaxPoolWithArgMax, self).__init__()
            self.max_pool_with_arg_max = P.MaxPoolWithArgmax(pad_mode="valid", kernel_size=2, strides=1)

        def construct(self, x):
            ret = self.max_pool_with_arg_max(x)
            return ret

    x = Tensor(np.ones([1, 1, 3, 3], np.float32))
    net = NetMaxPoolWithArgMax()
    context.set_context(mode=context.GRAPH_MODE)
    ret = net(x)
    print(ret)


class GradWrapUnfold(nn.Cell):
    """ GradWrapUnfold definition """

    def __init__(self, network):
        super(GradWrapUnfold, self).__init__()
        self.network = network
        self.sens = Tensor(np.ones([1, 4, 2, 2], np.float32))

    def construct(self, x):
        return grad_all_with_sens(self.network)(x, self.sens)


class UnfoldNetValid(nn.Cell):
    """ UnfoldNetValid definition """

    def __init__(self):
        super(UnfoldNetValid, self).__init__()
        self.unfold = nn.Unfold(ksizes=[1, 2, 2, 1],
                                strides=[1, 1, 1, 1],
                                rates=[1, 1, 1, 1],
                                padding='VALID')

    def construct(self, x):
        return self.unfold(x)


class UnfoldNetSame(nn.Cell):
    """ UnfoldNetSame definition """

    def __init__(self):
        super(UnfoldNetSame, self).__init__()
        self.unfold = nn.Unfold(ksizes=[1, 2, 2, 1],
                                strides=[1, 1, 1, 1],
                                rates=[1, 1, 1, 1],
                                padding='SAME')

    def construct(self, x):
        return self.unfold(x)


class FlattenNet(nn.Cell):
    """ FlattenNet definition """

    def __init__(self):
        super(FlattenNet, self).__init__()
        self.flatten = P.Flatten()

    def construct(self, x):
        return self.flatten(x)


class PReLUNet(nn.Cell):
    """ PReLUNet definition """

    def __init__(self):
        super(PReLUNet, self).__init__()
        self.prelu = P.PReLU()
        self.w = Tensor(np.ones(3, np.float32))

    def construct(self, x):
        return self.prelu(x, self.w)


class PReLUGradNet(nn.Cell):
    """ PReLUGradNet definition """

    def __init__(self):
        super(PReLUGradNet, self).__init__()
        self.prelu_grad = G.PReLUGrad()

    def construct(self, dout, x, w):
        return self.prelu_grad(dout, x, w)


class LRNNet(nn.Cell):
    """ LRNNet definition """

    def __init__(self):
        super(LRNNet, self).__init__()
        self.lrn = P.LRN()

    def construct(self, x):
        return self.lrn(x)


class LRNGradNet(nn.Cell):
    """ LRNGradNet definition """

    def __init__(self):
        super(LRNGradNet, self).__init__()
        self.lrn_grad = G.LRNGrad()

    def construct(self, dout, x, out):
        return self.lrn_grad(dout, x, out)


class ChannelShuffleFunc(nn.Cell):
    """ ChannelShuffleFunc definition """

    def __init__(self):
        super(ChannelShuffleFunc, self).__init__()
        self.channel_shuffle = mindspore.ops.function.nn_func.channel_shuffle

    def construct(self, x, group):
        return self.channel_shuffle(x, group)


test_cases = [
    ('SoftMaxGrad', {
        'block': SoftMaxGrad(VirtualNetWithLoss(P.Softmax())),
        'desc_inputs': [[128, 32, 32, 64]],
        'desc_bprop': [[128, 32, 32, 64]],
    }),
    ('DropoutGrad', {
        'block': DropoutGrad(VirtualNetWithLoss(nn.Dropout())),
        'desc_inputs': [[128, 32, 32, 64]],
        'desc_bprop': [[128, 32, 32, 64]],
    }),
    ('L2Normalize', {
        'block': L2NormalizeNet(),
        'desc_inputs': [Tensor(np.array([[1.0, 2, 3], [4.0, 5, 6], [7.0, 8, 9]]), mindspore.float32)],
    }),
    ('CosineSimilarity', {
        'block': CosineSimilarityNet(),
        'desc_inputs': [Tensor(np.array([[1, 3, 4, 7], [2, 4, 2, 5], [3, 1, 5, 8]]), mindspore.float32),
                        Tensor(np.array([[2, 4, 2, 5], [3, 1, 5, 8], [1, 3, 4, 7]]), mindspore.float32)]
    }),
    ('Upsample', {
        'block': UpsampleNet(),
        'desc_inputs': [Tensor(np.array([[[[1, 2, 3, 4], [5, 6, 7, 8]]]]), mindspore.float32)]
    }),
    ('PoissonNLLLoss', {
        'block': PoissonNLLLossNet(),
        'desc_inputs': [Tensor(np.array([[0.3, 0.7], [0.5, 0.5]]), mindspore.float32),
                        Tensor(np.array([[1, 2], [3, 4]]), mindspore.float32)]
    }),
    ('MultiLabelSoftMarginLoss', {
        'block': MultiLabelSoftMarginLossNet(),
        'desc_inputs': [Tensor(np.array([[0.3, 0.6, 0.6], [0.9, 0.4, 0.2]]), mindspore.float32),
                        Tensor(np.array([[0, 0, 1], [0, 0, 1]]), mindspore.float32)]
    }),
    ('FusedBatchNormGrad', {
        'block': FusedBatchNormGrad(nn.BatchNorm2d(num_features=512, eps=1e-5, momentum=0.1)),
        'desc_inputs': [[64, 512, 7, 7], [64, 512, 7, 7]],
        'desc_bprop': [[64, 512, 7, 7]],
    }),
    ('BatchnormGrad', {
        'block': Grad(NetWithLoss(BatchnormNet())),
        'desc_inputs': [Tensor(np.ones([1, 3, 8, 8], np.float32)), Tensor(np.zeros([1, 10], np.float32))],
    }),
    ('BlockGrad', {
        'block': Grad(NetWithLossClass(BlockNet())),
        'desc_inputs': [Tensor(np.ones([1, 3, 8, 8], np.float32)), Tensor(np.zeros([4, 4], np.float32))],
    }),
    ('Conv2dWithBiasGrad', {
        'block': Grad(NetWithLossClass(Conv2dWithBiasNet())),
        'desc_inputs': [Tensor(np.ones([1, 3, 16, 16], np.float32)), Tensor(np.zeros([1, 2560], np.float32))],
    }),
    ('Conv2dNativeGrad', {
        'block': Grad(NetWithLossClass(Conv2dNativeNet())),
        'desc_inputs': [Tensor(np.ones([1, 3, 16, 16], np.float32)), Tensor(np.zeros([1, 1764], np.float32))],
    }),
    ('StateTest', {
        'block': StateNet(),
        'desc_inputs': [Tensor(np.ones([2, 1, 2, 2]).astype(np.float32))],
    }),
    ('StateGrad', {
        'block': Grad(NetWithLossClass(StateNet())),
        'desc_inputs': [Tensor(np.ones([2, 1, 2, 2], np.float32)), Tensor(np.ones([2, 2], np.float32))],
    }),
    ('ComparisonTest', {
        'block': ComparisonNet(),
        'desc_inputs': [Tensor(np.ones([6, 9, 10], np.int32)), Tensor(np.ones([6, 9, 10], np.int32))],
    }),
    ('UnfoldValid', {
        'block': UnfoldNetValid(),
        'desc_inputs': [Tensor(np.ones([1, 1, 3, 3], np.float32))],
        'desc_bprop': [Tensor(np.ones([1, 4, 2, 2], np.float32))],
        'skip': ['backward']}),
    ('UnfoldSame', {
        'block': UnfoldNetSame(),
        'desc_inputs': [Tensor(np.ones([1, 1, 3, 3], np.float32))],
        'desc_bprop': [Tensor(np.ones([1, 4, 3, 3], np.float32))],
        'skip': ['backward']}),
    ('UnfoldGrad', {
        'block': GradWrapUnfold(UnfoldNetValid()),
        'desc_inputs': [Tensor(np.ones([1, 1, 3, 3], np.float32))],
        'desc_bprop': [Tensor(np.ones([1, 4, 2, 2], np.float32))],
        'skip': ['backward']}),
    ('LogSigmoid', {
        'block': nn.LogSigmoid(),
        'desc_inputs': [Tensor(np.array([1, 2, 3, 4]).astype(np.float32))],
        'desc_bprop': [Tensor(np.array([1, 2, 3, 4]).astype(np.float32))],
        'skip': ['backward']}),
    ('ReduceLogSumExp', {
        'block': nn.ReduceLogSumExp((0,), False),
        'desc_inputs': [Tensor(np.array([3, 4, 5, 6]).astype(np.float32))],
        'skip': ['backward']}),
    ('LGamma', {
        'block': nn.LGamma(),
        'desc_inputs': [Tensor(np.array([3, 4, 5, 6]).astype(np.float32))],
        'skip': ['backward']}),
    ('Identity', {
        'block': nn.Identity(),
        'desc_inputs': [Tensor(np.array([1, 2, 3, 4]).astype(np.int64))],
        'desc_bprop': [Tensor(np.array([1, 2, 3, 4]).astype(np.int64))]}),
    ('IGamma', {
        'block': nn.IGamma(),
        'desc_inputs': [Tensor(np.array([3, 4, 5, 6]).astype(np.float32)),
                        Tensor(np.array([3, 4, 5, 6]).astype(np.float32))],
        'skip': ['backward']}),
    ('DiGamma', {
        'block': nn.DiGamma(),
        'desc_inputs': [Tensor(np.array([3, 4, 5, 6]).astype(np.float32))],
        'skip': ['backward']}),
    ('LBeta', {
        'block': nn.LBeta(),
        'desc_inputs': [Tensor(np.array([3, 4, 5, 6]).astype(np.float32)),
                        Tensor(np.array([3, 4, 5, 6]).astype(np.float32))],
        'skip': ['backward']}),
    ('FlattenNet', {
        'block': FlattenNet(),
        'desc_inputs': [Tensor(np.ones([1, 2, 3, 4], np.float32))],
    }),
    ('PReLUNet', {
        'block': PReLUNet(),
        'desc_inputs': [Tensor(np.ones([1, 3, 4, 4], np.float32))],
    }),
    ('PReLUGradNet', {
        'block': PReLUGradNet(),
        'desc_inputs': [Tensor(np.ones([1, 3, 4, 4], np.float32)),
                        Tensor(np.ones([1, 3, 4, 4], np.float32)),
                        Tensor(np.ones(3, np.float32))],
    }),
    ('MatrixDiag', {
        'block': nn.MatrixDiag(),
        'desc_inputs': [Tensor(np.array([1, 2, 3]).astype(np.float32))],
        'skip': ['backward']
    }),
    ('MatrixDiagPart', {
        'block': nn.MatrixDiagPart(),
        'desc_inputs': [Tensor(np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float32))],
        'skip': ['backward']
    }),
    ('MatrixSetDiag', {
        'block': nn.MatrixSetDiag(),
        'desc_inputs': [Tensor(np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float32)),
                        Tensor(np.array([1, 2]).astype(np.float32))],
        'skip': ['backward']
    }),
    ('MatInverse', {
        'block': nn.MatInverse(),
        'desc_inputs': [Tensor(np.array([[4, 12, -16], [12, 37, -43], [-16, -43, 98]]).astype(np.float32))],
        'skip': ['backward']
    }),
    ('MatDet', {
        'block': nn.MatDet(),
        'desc_inputs': [Tensor(np.array([[4, 12, -16], [12, 37, -43], [-16, -43, 98]]).astype(np.float32))],
        'skip': ['backward']
    }),
    ('LRNNet', {
        'block': LRNNet(),
        'desc_inputs': [Tensor(np.ones([1, 5, 4, 4], np.float32))],
    }),
    ('LRNGradNet', {
        'block': LRNGradNet(),
        'desc_inputs': [Tensor(np.ones([1, 5, 4, 4], np.float32)),
                        Tensor(np.ones([1, 5, 4, 4], np.float32)),
                        Tensor(np.ones([1, 5, 4, 4], np.float32))],
    }),
    ('Unflatten', {
        'block': nn.Unflatten(1, (2, 5)),
        'desc_inputs': [Tensor(np.ones([2, 10, 5]))],
        'skip': ['backward']
    }),
    ('ChannelShuffleFunc', {
        'block': ChannelShuffleFunc(),
        'desc_inputs': [Tensor(np.arange(1 * 4 * 2 * 2).reshape(1, 4, 2, 2).astype(np.int16)), 2],
        'skip': ['backward']
    })
]

test_cases_for_verify_exception = [
    ('ApplyMomentum_Error', {
        'block': (P.ApplyMomentum(), {'exception': TypeError}),
        'desc_inputs': [[2], [128, 32, 32, 64], [128, 32, 32, 64], [128, 32, 32, 64], [128, 32, 32, 64]],
        'desc_bprop': [[128, 32, 32, 64]],
        'skip': ['backward']
    }),
    ('Conv2d_ValueError_1', {
        'block': (lambda _: P.Conv2D(3, 4, mode=-2.0), {'exception': TypeError}),
        'desc_inputs': [0],
    }),
    ('Conv2d_ValueError_2', {
        'block': (lambda _: P.Conv2D(3, 4, mode=-2), {'exception': ValueError}),
        'desc_inputs': [0],
    }),
    ('MaxPoolWithArgmax_ValueError_1', {
        'block': (lambda _: P.MaxPoolWithArgmax(pad_mode='sane'), {'exception': ValueError}),
        'desc_inputs': [0],
    }),
    ('MaxPoolWithArgmax_ValueError_2', {
        'block': (lambda _: P.MaxPoolWithArgmax(kernel_size='1'), {'exception': TypeError}),
        'desc_inputs': [0],
    }),
    ('MaxPoolWithArgmax_ValueError_3', {
        'block': (lambda _: P.MaxPoolWithArgmax(kernel_size=-2), {'exception': ValueError}),
        'desc_inputs': [0],
    }),
    ('MaxPoolWithArgmax_ValueError_4', {
        'block': (lambda _: P.MaxPoolWithArgmax(strides=-1), {'exception': ValueError}),
        'desc_inputs': [0],
    }),
    ('Softmax_ValueError_1', {
        'block': (lambda _: P.Softmax("1"), {'exception': TypeError}),
        'desc_inputs': [0],
    }),
    ('Softmax_ValueError_2', {
        'block': (lambda _: P.Softmax(1.1), {'exception': TypeError}),
        'desc_inputs': [0],
    }),
    ('Softmax_ValueError_3', {
        'block': (lambda _: P.Softmax(axis="1"), {'exception': TypeError}),
        'desc_inputs': [0],
    }),
    ('DropoutGenMask_ValueError_1', {
        'block': (lambda _: P.DropoutGenMask(Seed0="seed0"), {'exception': TypeError}),
        'desc_inputs': [0],
    }),
    ('DropoutGenMask_ValueError_2', {
        'block': (lambda _: P.DropoutGenMask(Seed0=1.0), {'exception': TypeError}),
        'desc_inputs': [0],
    }),
    ('DropoutGenMask_ValueError_3', {
        'block': (lambda _: P.DropoutGenMask(Seed1="seed1"), {'exception': TypeError}),
        'desc_inputs': [0],
    }),
    ('DropoutGenMask_ValueError_4', {
        'block': (lambda _: P.DropoutGenMask(Seed1=2.0), {'exception': TypeError}),
        'desc_inputs': [0],
    }),
    ('MaxPool2d_ValueError_1', {
        'block': (nn.MaxPool2d(kernel_size=120, stride=1, pad_mode="valid"), {'exception': ValueError}),
        'desc_inputs': [Tensor(np.random.randn(32, 3, 112, 112).astype(np.float32).transpose(0, 3, 1, 2))],
    }),
    ('MaxPool2d_ValueError_2', {
        'block': (
            lambda _: nn.MaxPool2d(kernel_size=120, stride=True, pad_mode="valid"),
            {'exception': TypeError},
        ),
        'desc_inputs': [Tensor(np.random.randn(32, 3, 112, 112).astype(np.float32).transpose(0, 3, 1, 2))],
    }),
    ('MaxPool2d_ValueError_3', {
        'block': (
            lambda _: nn.MaxPool2d(kernel_size=3, stride=True, pad_mode="valid"),
            {'exception': TypeError},
        ),
        'desc_inputs': [Tensor(np.random.randn(32, 3, 112, 112).astype(np.float32).transpose(0, 3, 1, 2))],
    }),
    ('Identity_TypeError_1', {
        'block': (
            lambda _: nn.Identity(),
            {'exception': TypeError},
        ),
        'desc_inputs': [np.array([3, 4, 5, 6]).astype(np.int64)],
    }),
    ('ReduceLogsumexp_TypeError_1', {
        'block': (
            lambda _: nn.ReduceLogSumExp(axis=(0,), keep_dims=2),
            {'exception': TypeError},
        ),
        'desc_inputs': [Tensor(np.array([3, 4, 5, 6]).astype(np.float32))],
    }),
    ('ReduceLogsumexp_TypeError_2', {
        'block': (
            lambda _: nn.ReduceLogSumExp(axis=1.2, keep_dims=True),
            {'exception': TypeError},
        ),
        'desc_inputs': [Tensor(np.array([3, 4, 5, 6]).astype(np.float32))],
    }),
]


@security_off_wrap
@non_graph_engine
@mindspore_test(pipeline_for_verify_exception_for_case_by_case_config)
def test_summary_nn_ops():
    if security.enable_security():
        return []
    test_cases_for_summary_ops = [
        ('ScalarSummary', {
            'block': ScalarSummaryNet(),
            'desc_inputs': [Tensor(2.2)],
        }),
        ('HistogramSummary', {
            'block': HistogramSummaryNet(),
            'desc_inputs': [[1, 2, 3]],
        }),
    ]
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    return test_cases_for_summary_ops


def test_summary_nn_ops_security_on():
    if security.enable_security():
        with pytest.raises(ValueError) as exc:
            ScalarSummaryNet()
        assert str(exc.value) == 'The Summary is not supported, please without `-s on` and recompile source.'


@non_graph_engine
@mindspore_test(pipeline_for_compile_forward_ge_graph_for_case_by_case_config)
def test_compile():
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    return test_cases


@mindspore_test(pipeline_for_verify_exception_for_case_by_case_config)
def test_check_exception():
    return test_cases_for_verify_exception
