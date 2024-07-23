import numpy as np

from mindspore import Tensor
from mindspore.rewrite import SymbolTree
from tests.models.official.cv.mobilenetv2.src.mobilenetV2 import MobileNetV2Backbone, MobileNetV2Head, mobilenet_v2
from tests.mark_utils import arg_mark


def define_net():
    backbone_net = MobileNetV2Backbone()
    activation = "None"
    head_net = MobileNetV2Head(input_channel=backbone_net.out_channels,
                               num_classes=2,
                               activation=activation)
    net = mobilenet_v2(backbone_net, head_net)
    return backbone_net, head_net, net


def test_mobilenet():
    """
    Feature: Test Rewrite.
    Description: Test Rewrite on Mobilenetv2.
    Expectation: Success.
    """
    _, _, net = define_net()
    predict = Tensor(np.ones([1, 3, 224, 224]).astype(np.float32))
    expect = net(predict)
    stree = SymbolTree.create(net)
    net_opt = stree.get_network()
    output = net_opt(predict)
    assert np.allclose(output.asnumpy(), expect.asnumpy())
