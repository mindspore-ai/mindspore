from mindspore import nn
from mindspore.ops.primitive import Primitive
from mindspore import _checkparam as Validator
from mindspore.nn.layer.activation import get_activation, LeakyReLU


class Conv2dBnAct(nn.Cell):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 pad_mode='same',
                 padding=0,
                 dilation=1,
                 group=1,
                 has_bias=False,
                 weight_init='normal',
                 bias_init='zeros',
                 has_bn=False,
                 momentum=0.997,
                 eps=1e-5,
                 activation=None,
                 alpha=0.2,
                 after_fake=True):
        """Initialize Conv2dBnAct."""
        super(Conv2dBnAct, self).__init__()

        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              pad_mode=pad_mode,
                              padding=padding,
                              dilation=dilation,
                              group=group,
                              has_bias=has_bias,
                              weight_init=weight_init,
                              bias_init=bias_init)
        self.has_bn = Validator.check_bool(has_bn, "has_bn", self.cls_name)
        self.has_act = activation is not None
        self.after_fake = Validator.check_bool(after_fake, "after_fake", self.cls_name)
        if has_bn:
            self.batchnorm = nn.BatchNorm2d(out_channels, eps, momentum)
        if activation == "leakyrelu":
            self.activation = LeakyReLU(alpha)
        else:
            self.activation = get_activation(activation) if isinstance(activation, str) else activation
            if activation is not None and not isinstance(self.activation, (nn.Cell, Primitive)):
                raise TypeError(f"For '{self.cls_name}', the 'activation' must be str or Cell or Primitive, "
                                f"but got {type(activation).__name__}.")

    def construct(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.batchnorm(x)
        if self.has_act:
            x = self.activation(x)
        return x
