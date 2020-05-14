"""mobile net v2"""
from mindspore import nn
from mindspore.ops import operations as P
from mindspore.nn.layer import combined


def make_divisible(input_x, div_by=8):
    return int((input_x + div_by) // div_by)


def _conv_bn(in_channel,
             out_channel,
             ksize,
             stride=1):
    """Get a conv2d batchnorm and relu layer."""
    return nn.SequentialCell(
        [combined.Conv2d(in_channel,
                         out_channel,
                         kernel_size=ksize,
                         stride=stride,
                         batchnorm=True)])


class InvertedResidual(nn.Cell):
    def __init__(self, inp, oup, stride, expend_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expend_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup
        if expend_ratio == 1:
            self.conv = nn.SequentialCell([
                combined.Conv2d(hidden_dim,
                                hidden_dim,
                                3,
                                stride,
                                group=hidden_dim,
                                batchnorm=True,
                                activation='relu6'),
                combined.Conv2d(hidden_dim, oup, 1, 1,
                                batchnorm=True)
            ])
        else:
            self.conv = nn.SequentialCell([
                combined.Conv2d(inp, hidden_dim, 1, 1,
                                batchnorm=True,
                                activation='relu6'),
                combined.Conv2d(hidden_dim,
                                hidden_dim,
                                3,
                                stride,
                                group=hidden_dim,
                                batchnorm=True,
                                activation='relu6'),
                combined.Conv2d(hidden_dim, oup, 1, 1,
                                batchnorm=True)
            ])
        self.add = P.TensorAdd()

    def construct(self, input_x):
        out = self.conv(input_x)
        if self.use_res_connect:
            out = self.add(input_x, out)
        return out


class MobileNetV2(nn.Cell):
    def __init__(self, num_class=1000, input_size=224, width_mul=1.):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        inverted_residual_setting = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 230, 1, 1],
        ]
        if width_mul > 1.0:
            last_channel = make_divisible(last_channel * width_mul)
        self.last_channel = last_channel
        features = [_conv_bn(3, input_channel, 3, 2)]

        for t, c, n, s in inverted_residual_setting:
            out_channel = make_divisible(c * width_mul) if t > 1 else c
            for i in range(n):
                if i == 0:
                    features.append(block(input_channel, out_channel, s, t))
                else:
                    features.append(block(input_channel, out_channel, 1, t))
                input_channel = out_channel

        features.append(_conv_bn(input_channel, self.last_channel, 1))

        self.features = nn.SequentialCell(features)
        self.mean = P.ReduceMean(keep_dims=False)
        self.classifier = combined.Dense(self.last_channel, num_class)

    def construct(self, input_x):
        out = input_x
        out = self.features(out)
        out = self.mean(out, (2, 3))
        out = self.classifier(out)
        return out
