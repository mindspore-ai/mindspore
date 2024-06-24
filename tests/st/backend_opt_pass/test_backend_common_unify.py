# Copyright 2024 Huawei Technologies Co., Ltd
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
# ==============================================================================
import numpy as np
import mindspore as ms
from mindspore import nn, Tensor, Parameter, ops
from .util import Capture, capture, WrapNet, GradNetWrtX
from tests.mark_utils import arg_mark

@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_renorm():
    """
    Feature: Validate opt pass keep node scope and id.
    Description: Test opt pass RenormSplit.
    Expectation: No exception and node id after opt pass is as expected.
    """
    ms.set_context(mode=ms.GRAPH_MODE, device_target='Ascend')

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.remorm = ops.Renorm(p=1, dim=0, maxnorm=5.)

        def construct(self, x):
            return self.remorm(x)

    cap = Capture('renorm_split', 'Renorm')
    with capture(cap):
        net = WrapNet(Net())
        x = Tensor(np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]]), ms.float32)
        net(x)

    patterns = ['Default/net-Net/BroadcastTo-op',
                'Default/net-Net/Mul-op']
    cap.check_output(patterns)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_reduce_mean():
    """
    Feature: Validate opt pass keep node scope and id.
    Description: Test opt pass ReduceAxisUpdate.
    Expectation: No exception and node id after opt pass is as expected.
    """
    ms.set_context(mode=ms.GRAPH_MODE, device_target='Ascend')

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.op = ops.ReduceMean(keep_dims=True)

        def construct(self, x):
            return self.op(x)

    cap = Capture('reduce_axis_update', 'ReduceMean')
    with capture(cap):
        net = WrapNet(Net())
        x = Tensor(np.random.randn(3, 4, 5, 6).astype(np.float32))
        net(x)

    patterns = ['Default/net-Net/ReduceMean-op']
    cap.check_output(patterns)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_histogram_fixedwidth():
    """
    Feature: Validate opt pass keep node scope and id.
    Description: Test opt pass HistogramFixedWidthFusion.
    Expectation: No exception and node id after opt pass is as expected.
    """
    ms.set_context(mode=ms.GRAPH_MODE, device_target='Ascend')

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.op = ops.HistogramFixedWidth(5)

        def construct(self, x, r):
            return self.op(x, r)

    cap = Capture('histogram_fixed_width_fusion', 'HistogramFixedWidth')
    with capture(cap):
        net = WrapNet(Net())
        x = Tensor([-1.0, 0.0, 1.5, 2.0, 5.0, 15], ms.float16)
        range_op = Tensor([0.0, 5.0], ms.float16)
        net(x, range_op)

    patterns = ['Default/net-Net/HistogramFixedWidth-op']
    cap.check_output(patterns)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_clipbynorm():
    """
    Feature: Validate opt pass keep node scope and id.
    Description: Test opt pass ClipByNormFission.
    Expectation: No exception and node id after opt pass is as expected.
    """
    ms.set_context(mode=ms.GRAPH_MODE, device_target='Ascend')

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.op = ops.operations._inner_ops.ClipByNorm()    # pylint: disable=W0212

        def construct(self, x, norm):
            return self.op(x, norm)

    cap = Capture('clip_by_norm_fission', 'ReduceSum')
    with capture(cap):
        net = WrapNet(Net())
        x = Tensor(np.random.randint(0, 10, [4, 16]), ms.float32)
        clip_norm = Tensor(np.array([100]).astype(np.float32))
        net(x, clip_norm)

    patterns = ["Default/net-Net/Mul-op",
                "Default/net-Net/Square-op",
                "Default/net-Net/ReduceSum-op",
                "Default/net-Net/ZerosLike-op",
                "Default/net-Net/Greater-op",
                "Default/net-Net/OnesLike-op",
                "Default/net-Net/Select-op",
                "Default/net-Net/Sqrt-op",
                "Default/net-Net/Select-op",
                "Default/net-Net/Maximum-op",
                "Default/net-Net/Div-op"]
    cap.check_output(patterns)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_adam_weightdecay():
    """
    Feature: Validate opt pass keep node scope and id.
    Description: Test opt pass AdamWeightDecayUnifyMindIR.
    Expectation: No exception and node id after opt pass is as expected.
    """
    ms.set_context(mode=ms.GRAPH_MODE, device_target='Ascend')

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.adam_weight_decay = ops.AdamWeightDecay()
            self.var = Parameter(Tensor(np.ones([2, 2]).astype(np.float32)), name="var")
            self.m = Parameter(Tensor(np.ones([2, 2]).astype(np.float16)), name="m")
            self.v = Parameter(Tensor(np.ones([2, 2]).astype(np.float16)), name="v")

        def construct(self, lr, beta1, beta2, epsilon, decay, grad):
            out = self.adam_weight_decay(self.var, self.m, self.v, lr, beta1, beta2, epsilon, decay, grad)
            return out

    cap = Capture('adam_weight_decay_unify_mindir', 'AdamApplyOneWithDecay')
    with capture(cap):
        net = WrapNet(Net())
        gradient = Tensor(np.ones([2, 2]).astype(np.float32))
        net(Tensor(0.001), Tensor(0.9), Tensor(0.999), Tensor(1e-8), Tensor(0.0), gradient)

    patterns = ["Default/net-Net/Sub-op",
                "Default/net-Net/Sub-op",
                "Default/net-Net/AdamApplyOneWithDecay-op",
                "Default/net-Net/TupleGetItem-op",
                "Default/net-Net/Assign-op",
                "Default/net-Net/TupleGetItem-op",
                "Default/net-Net/Assign-op",
                "Default/net-Net/MakeTuple-op"]
    cap.check_output(patterns)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_cdistfission():
    """
    Feature: Validate opt pass keep node scope and id.
    Description: Test opt pass CdistFission.
    Expectation: No exception and node id after opt pass is as expected.
    """
    ms.set_context(mode=ms.GRAPH_MODE, device_target='Ascend')

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.op = ops.Cdist(p=2.0)
            self.relu = ops.ReLU()

        def construct(self, x, y):
            return self.op(self.relu(x), self.relu(y))

    cap = Capture('cdist_fission', 'Cdist')
    with capture(cap):
        net = WrapNet(Net())
        input_x = Tensor(np.array([[[1.0, 1.0], [2.0, 2.0]]]).astype(np.float32))
        input_y = Tensor(np.array([[[3.0, 3.0], [3.0, 3.0]]]).astype(np.float32))
        net(input_x, input_y)

    patterns = ["Default/net-Net/ExpandDims-op",
                "Default/net-Net/BroadcastTo-op",
                "Default/net-Net/ExpandDims-op",
                "Default/net-Net/BroadcastTo-op",
                "Default/net-Net/Cdist-op"]
    cap.check_output(patterns)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_cdistgrad_fission():
    """
    Feature: Validate opt pass keep node scope and id.
    Description: Test opt pass CdistGradFission.
    Expectation: No exception and node id after opt pass is as expected.
    """
    ms.set_context(mode=ms.GRAPH_MODE, device_target='Ascend')

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.op = ops.Cdist(p=2.0)
            self.relu = ops.ReLU()
            self.val = Tensor(np.array([[[1.0, 1.0], [2.0, 2.0]]]).astype(np.float32))

        def construct(self, x):
            return self.relu(self.op(x, self.val))

    cap = Capture('cdist_grad_fission', 'CdistGrad')
    with capture(cap):
        net = GradNetWrtX(Net())
        input_x = Tensor(np.array([[[3.0, 3.0], [3.0, 3.0]]]).astype(np.float32))
        net(input_x)

    patterns = ["Gradients/Default/net-Net/Grad_ReLU/ExpandDims-op",
                "Gradients/Default/net-Net/Grad_ReLU/BroadcastTo-op",
                "Gradients/Default/net-Net/Grad_Cdist/CdistGrad-op"]
    cap.check_output(patterns)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_sparse_softmax_entropy():
    """
    Feature: Validate opt pass keep node scope and id.
    Description: Test opt pass SparseSoftmaxCrossEntropyWithLogitsUnifyMindIR.
    Expectation: No exception and node id after opt pass is as expected.
    """
    ms.set_context(mode=ms.GRAPH_MODE, device_target='Ascend')

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.op = ops.SparseSoftmaxCrossEntropyWithLogits()
            self.relu = ops.ReLU()

        def construct(self, logits, labels):
            return self.op(self.relu(logits), self.relu(labels))

    cap = Capture('sparse_softmax_cross_entropy_with_logits_unify_mindir', '= SoftmaxCrossEntropyWithLogits')
    with capture(cap):
        net = WrapNet(Net())
        logits = Tensor([[2, 3, 1, 4, 5], [2, 1, 2, 4, 3]], ms.float32)
        labels = Tensor([0, 1], ms.int32)
        net(logits, labels)

    patterns = ["Default/net-Net/Reshape-op",
                "Default/net-Net/Reshape-op",
                "Default/net-Net/OneHot-op",
                "Default/net-Net/SoftmaxCrossEntropyWithLogits-op",
                "Default/net-Net/TupleGetItem-op",
                "Default/net-Net/ReduceMean-op"]
    cap.check_output(patterns)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dropout():
    """
    Feature: Validate opt pass keep node scope and id.
    Description: Test opt pass DropoutUnifyMindIR1.
    Expectation: No exception and node id after opt pass is as expected.
    """
    ms.set_context(mode=ms.GRAPH_MODE, device_target='Ascend')

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.op = ops.Dropout(keep_prob=0.5)

        def construct(self, x):
            return self.op(x)

    cap = Capture('dropout_unify_mindir1', 'DropoutGenMask')
    with capture(cap):
        net = WrapNet(Net())
        x = Tensor(np.ones([1, 2, 3, 4, 5]), ms.float32)
        net(x)

    patterns = ["Default/net-Net/DropoutGenMask-op",
                "Default/net-Net/DropoutDoMask-op",
                "Default/net-Net/MakeTuple-op",
                "Default/net-Net/TupleGetItem-op"]
    cap.check_output(patterns)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_dropout_grad():
    """
    Feature: Validate opt pass keep node scope and id.
    Description: Test opt pass DropoutGradUnifyMindIR.
    Expectation: No exception and node id after opt pass is as expected.
    """
    ms.set_context(mode=ms.GRAPH_MODE, device_target='Ascend')

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.op = ops.Dropout(keep_prob=0.5)

        def construct(self, x):
            return self.op(x)

    cap = Capture('dropoutgrad_unify_mindir', 'DropoutDoMask')
    with capture(cap):
        net = GradNetWrtX(Net())
        x = Tensor(np.ones([1, 2, 3, 4, 5]), ms.float32)
        net(x)

    patterns = ["Gradients/Default/net-Net/Grad_Dropout/DropoutDoMask-op"]
    cap.check_output(patterns)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_batchnorm_grad():
    """
    Feature: Validate opt pass keep node scope and id.
    Description: Test opt pass BatchNormGradUnifyMindIR.
    Expectation: No exception and node id after opt pass is as expected.
    """
    ms.set_context(mode=ms.GRAPH_MODE, device_target='Ascend')

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.op = ops.BatchNorm()
            self.scale = Tensor(np.ones([2]), ms.float32)
            self.bias = Tensor(np.ones([2]), ms.float32)
            self.mean = Tensor(np.ones([2]), ms.float32)
            self.variance = Tensor(np.ones([2]), ms.float32)

        def construct(self, input_x):
            return self.op(input_x, self.scale, self.bias, self.mean, self.variance)

    cap = Capture('bn_grad_unify_mindir', 'BatchNormGrad')
    with capture(cap):
        net = GradNetWrtX(Net())
        input_x = Tensor(np.ones([2, 2]), ms.float32)

        net(input_x)

    patterns = ["Gradients/Default/net-Net/Grad_BatchNorm/BatchNormGrad-op"]
    cap.check_output(patterns)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_bn_grad2bninfer_grad():
    """
    Feature: Validate opt pass keep node scope and id.
    Description: Test opt pass BatchNormGrad2BNInferGrad.
    Expectation: No exception and node id after opt pass is as expected.
    """
    ms.set_context(mode=ms.GRAPH_MODE, device_target='Ascend')

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.op = ops.BatchNorm()
            self.scale = Tensor(np.ones([2]), ms.float32)
            self.bias = Tensor(np.ones([2]), ms.float32)
            self.mean = Tensor(np.ones([2]), ms.float32)
            self.variance = Tensor(np.ones([2]), ms.float32)

        def construct(self, input_x):
            return self.op(input_x, self.scale, self.bias, self.mean, self.variance)

    cap = Capture('batchnormgrad_to_bninfergrad', 'BNInferGrad')
    with capture(cap):
        net = GradNetWrtX(Net())
        input_x = Tensor(np.ones([2, 2]), ms.float32)

        net(input_x)

    patterns = ["Gradients/Default/net-Net/Grad_BatchNorm/BNInferGrad-op"]
    cap.check_output(patterns)
