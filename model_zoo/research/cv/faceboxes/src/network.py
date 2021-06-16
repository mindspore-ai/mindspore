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
"""FaceBoxes model define"""
import mindspore.ops as P
from mindspore.ops import functional as F
from mindspore.ops import composite as C
import mindspore
from mindspore import nn
from mindspore import context
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.communication.management import get_group_size


class CRelu(nn.Cell):
    """CRelu"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, num_features):
        super(CRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, pad_mode="pad",
                              dilation=(1, 1), group=1,
                              has_bias=False)
        self.batchnorm = nn.BatchNorm2d(num_features=num_features,
                                        eps=1e-5, momentum=0.9)
        self.concat = P.Concat(axis=1)
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.concat((x, -x,))
        x = self.relu(x)
        return x


class BasicConv2d(nn.Cell):
    """BasicConv2d"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, pad_mode, num_features):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, pad_mode=pad_mode,
                              dilation=(1, 1), group=1,
                              has_bias=False)
        self.batchnorm = nn.BatchNorm2d(num_features=num_features,
                                        eps=1e-5,
                                        momentum=0.9)
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x


class Inception(nn.Cell):
    """Inception"""
    def __init__(self):
        super(Inception, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels=128, out_channels=32,
                                     kernel_size=(1, 1), stride=(1, 1),
                                     padding=0, pad_mode="valid",
                                     num_features=32)
        self.pad_0 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)), mode="CONSTANT")
        self.pad_avgpool = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.avgpool = nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1))
        self.branch1x1_2 = BasicConv2d(in_channels=128, out_channels=32, kernel_size=(1, 1), stride=(1, 1),
                                       padding=0, pad_mode="valid",
                                       num_features=32)
        self.branch3x3_reduce = BasicConv2d(in_channels=128, out_channels=24, kernel_size=(1, 1), stride=(1, 1),
                                            padding=0, pad_mode="valid",
                                            num_features=24)
        self.branch3x3 = BasicConv2d(in_channels=24, out_channels=32, kernel_size=(3, 3), stride=(1, 1),
                                     padding=(1, 1, 1, 1), pad_mode="pad",
                                     num_features=32)
        self.branch3x3_reduce_2 = BasicConv2d(in_channels=128, out_channels=24, kernel_size=(1, 1), stride=(1, 1),
                                              padding=0, pad_mode="valid",
                                              num_features=24)
        self.branch3x3_2 = BasicConv2d(in_channels=24, out_channels=32, kernel_size=(3, 3), stride=(1, 1),
                                       padding=(1, 1, 1, 1), pad_mode="pad",
                                       num_features=32)
        self.branch3x3_3 = BasicConv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1),
                                       padding=(1, 1, 1, 1), pad_mode="pad",
                                       num_features=32)
        self.concat = P.Concat(axis=1)

    def construct(self, x):
        """construct"""
        branch1x1_opt = self.branch1x1(x)
        opt_pad_0 = self.pad_0(x)
        y = self.pad_avgpool(opt_pad_0)
        y = self.avgpool(y)
        branch1x1_2_opt = self.branch1x1_2(y)
        y = self.branch3x3_reduce(x)
        branch3x3_opt = self.branch3x3(y)
        y = self.branch3x3_reduce_2(x)
        y = self.branch3x3_2(y)
        branch3x3_3_opt = self.branch3x3_3(y)
        opt_concat_2 = self.concat((branch1x1_opt, branch1x1_2_opt, branch3x3_opt, branch3x3_3_opt,))
        return opt_concat_2


class FaceBoxes(nn.Cell):
    """FaceBoxes"""
    def __init__(self, phase='train'):
        super(FaceBoxes, self).__init__()
        self.num_classes = 2

        self.conv1 = CRelu(in_channels=3, out_channels=24, kernel_size=(7, 7), stride=(4, 4),
                           padding=(3, 3, 3, 3), num_features=24)
        self.pad_maxpool_0 = nn.Pad(paddings=((0, 0), (0, 0), (1, 0), (1, 0)))
        self.pad_maxpool_1 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.conv2 = CRelu(in_channels=48, out_channels=64, kernel_size=(5, 5), stride=(2, 2),
                           padding=(2, 2, 2, 2), num_features=64)
        self.inception_0 = Inception()
        self.inception_1 = Inception()
        self.inception_2 = Inception()
        self.conv3_1 = BasicConv2d(in_channels=128, out_channels=128,
                                   kernel_size=(1, 1), stride=(1, 1),
                                   padding=0, pad_mode="valid", num_features=128)
        self.conv3_2 = BasicConv2d(in_channels=128, out_channels=256,
                                   kernel_size=(3, 3), stride=(2, 2),
                                   padding=(1, 1, 1, 1), pad_mode="pad", num_features=256)
        self.conv4_1 = BasicConv2d(in_channels=256, out_channels=128,
                                   kernel_size=(1, 1), stride=(1, 1),
                                   padding=0, pad_mode="valid", num_features=128)
        self.conv4_2 = BasicConv2d(in_channels=128, out_channels=256,
                                   kernel_size=(3, 3), stride=(2, 2),
                                   padding=(1, 1, 1, 1), pad_mode="pad", num_features=256)
        self.loc_layer = nn.CellList([
            nn.Conv2d(in_channels=128, out_channels=84, kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1, 1, 1), pad_mode="pad", dilation=(1, 1), group=1, has_bias=True),
            nn.Conv2d(in_channels=256, out_channels=4, kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1, 1, 1), pad_mode="pad", dilation=(1, 1), group=1, has_bias=True),
            nn.Conv2d(in_channels=256, out_channels=4, kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1, 1, 1), pad_mode="pad", dilation=(1, 1), group=1, has_bias=True)
        ])
        self.conf_layer = nn.CellList([
            nn.Conv2d(in_channels=128, out_channels=42, kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1, 1, 1), pad_mode="pad", dilation=(1, 1), group=1, has_bias=True),
            nn.Conv2d(in_channels=256, out_channels=2, kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1, 1, 1), pad_mode="pad", dilation=(1, 1), group=1, has_bias=True),
            nn.Conv2d(in_channels=256, out_channels=2, kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1, 1, 1), pad_mode="pad", dilation=(1, 1), group=1, has_bias=True)
        ])

        self.transpose = P.Transpose()
        self.reshape = P.Reshape()
        self.get_shape = P.Shape()
        self.concat = P.Concat(axis=1)
        self.softmax = nn.Softmax(axis=2)

    def construct(self, x):
        """construct"""
        x = self.conv1(x)
        x = self.pad_maxpool_0(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.pad_maxpool_1(x)
        x = self.maxpool(x)
        x = self.inception_0(x)
        x = self.inception_1(x)
        x = self.inception_2(x)
        conv3_1_opt = self.conv3_1(x)
        conv3_2_opt = self.conv3_2(conv3_1_opt)
        conv4_1_opt = self.conv4_1(conv3_2_opt)
        conv4_2_opt = self.conv4_2(conv4_1_opt)

        detection_sources = [x, conv3_2_opt, conv4_2_opt]

        loc, conf = (), ()
        for i in range(3):
            loc_opt = self.transpose(self.loc_layer[i](detection_sources[i]), (0, 2, 3, 1))
            loc_opt = self.reshape(loc_opt, (self.get_shape(loc_opt)[0], -1))
            loc += (loc_opt,)
            conf_opt = self.transpose(self.conf_layer[i](detection_sources[i]), (0, 2, 3, 1))
            conf_opt = self.reshape(conf_opt, (self.get_shape(conf_opt)[0], -1))
            conf += (conf_opt,)

        loc = self.concat(loc)
        conf = self.concat(conf)

        loc = self.reshape(loc, (self.get_shape(loc)[0], -1, 4))
        conf = self.reshape(conf, (self.get_shape(conf)[0], -1, self.num_classes))

        if self.phase == 'train':
            output = (loc, conf)
        else:
            output = (loc, self.softmax(conf))
        return output



class FaceBoxesWithLossCell(nn.Cell):
    """FaceBoxesWithLossCell"""
    def __init__(self, network, multibox_loss, config):
        super(FaceBoxesWithLossCell, self).__init__()
        self.network = network
        self.loc_weight = config['loc_weight']
        self.class_weight = config['class_weight']
        self.multibox_loss = multibox_loss

    def construct(self, img, loc_t, conf_t):
        pred_loc, pre_conf = self.network(img)
        loss_loc, loss_conf = self.multibox_loss(pred_loc, loc_t, pre_conf, conf_t)

        return loss_loc * self.loc_weight + loss_conf * self.class_weight


class TrainingWrapper(nn.Cell):
    """TrainingWrapper"""
    def __init__(self, network, optimizer, sens=1.0):
        super(TrainingWrapper, self).__init__(auto_prefix=False)
        self.network = network
        self.weights = mindspore.ParameterTuple(network.trainable_params())
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer = None
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        class_list = [mindspore.context.ParallelMode.DATA_PARALLEL, mindspore.context.ParallelMode.HYBRID_PARALLEL]
        if self.parallel_mode in class_list:
            self.reducer_flag = True
        if self.reducer_flag:
            mean = context.get_auto_parallel_context("gradients_mean")
            if auto_parallel_context().get_device_num_is_set():
                degree = context.get_auto_parallel_context("device_num")
            else:
                degree = get_group_size()
            self.grad_reducer = nn.DistributedGradReducer(optimizer.parameters, mean, degree)

    def construct(self, *args):
        weights = self.weights
        loss = self.network(*args)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(*args, sens)
        if self.reducer_flag:
            # apply grad reducer on grads
            grads = self.grad_reducer(grads)
        return F.depend(loss, self.optimizer(grads))
