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
"""net structure"""
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore.parallel._utils import _get_device_num, _get_parallel_mode, _get_gradients_mean
from mindspore.context import ParallelMode
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer

class SiameseRPN(nn.Cell):
    """
        SiameseRPN Network.

        Args:
            groups (int): Size of one batch.
            k (int): Numbers of one point‘s anchors.
            s (int): Numbers of one anchor‘s parameters.

        Returns:
            coutputs tensor, routputs tensor.
        """
    def __init__(self, groups=1, k=5, s=4, is_train=False, is_trackinit=False, is_track=False):
        super(SiameseRPN, self).__init__()
        self.groups = groups
        self.k = k
        self.s = s
        self.is_train = is_train
        self.is_trackinit = is_trackinit
        self.is_track = is_track
        self.expand_dims = ops.ExpandDims()
        self.featureExtract = nn.SequentialCell(
            [nn.Conv2d(3, 96, kernel_size=11, stride=2, pad_mode='valid', has_bias=True),
             nn.BatchNorm2d(96, use_batch_statistics=False),
             nn.ReLU(),
             nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='valid'),
             nn.Conv2d(96, 256, kernel_size=5, pad_mode='valid', has_bias=True),
             nn.BatchNorm2d(256, use_batch_statistics=False),
             nn.ReLU(),
             nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='valid'),
             nn.Conv2d(256, 384, kernel_size=3, pad_mode='valid', has_bias=True),
             nn.BatchNorm2d(384, use_batch_statistics=False),
             nn.ReLU(),
             nn.Conv2d(384, 384, kernel_size=3, pad_mode='valid', has_bias=True),
             nn.BatchNorm2d(384),
             nn.ReLU(),
             nn.Conv2d(384, 256, kernel_size=3, pad_mode='valid', has_bias=True),
             nn.BatchNorm2d(256)])
        self.conv1 = nn.Conv2d(256, 2 * self.k * 256, kernel_size=3, pad_mode='valid', has_bias=True)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(256, 4 * self.k * 256, kernel_size=3, pad_mode='valid', has_bias=True)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, pad_mode='valid', has_bias=True)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, pad_mode='valid', has_bias=True)
        self.relu4 = nn.ReLU()

        self.op_split_input = ops.Split(axis=1, output_num=self.groups)
        self.op_split_krenal = ops.Split(axis=0, output_num=self.groups)
        self.op_concat = ops.Concat(axis=1)
        self.conv2d_cout = ops.Conv2D(out_channel=10, kernel_size=4)
        self.conv2d_rout = ops.Conv2D(out_channel=20, kernel_size=4)
        self.regress_adjust = nn.Conv2d(4 * self.k, 4 * self.k, 1, pad_mode='valid', has_bias=True)
        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()
        self.softmax = ops.Softmax(axis=2)
        self.print = ops.Print()

    def construct(self, template=None, detection=None, ckernal=None, rkernal=None):
        """ forward function """
        if self.is_train is True and template is not None and detection is not None:
            template_feature = self.featureExtract(template)
            detection_feature = self.featureExtract(detection)

            ckernal = self.conv1(template_feature)
            ckernal = self.reshape(ckernal.view(self.groups, 2 * self.k, 256, 4, 4), (-1, 256, 4, 4))
            cinput = self.reshape(self.conv3(detection_feature), (1, -1, 20, 20))

            rkernal = self.conv2(template_feature)
            rkernal = self.reshape(rkernal.view(self.groups, 4 * self.k, 256, 4, 4), (-1, 256, 4, 4))
            rinput = self.reshape(self.conv4(detection_feature), (1, -1, 20, 20))
            c_features = self.op_split_input(cinput)
            c_weights = self.op_split_krenal(ckernal)
            r_features = self.op_split_input(rinput)
            r_weights = self.op_split_krenal(rkernal)
            coutputs = ()
            routputs = ()
            for i in range(self.groups):
                coutputs = coutputs + (self.conv2d_cout(c_features[i], c_weights[i]),)
                routputs = routputs + (self.conv2d_rout(r_features[i], r_weights[i]),)
            coutputs = self.op_concat(coutputs)
            routputs = self.op_concat(routputs)
            coutputs = self.reshape(coutputs, (self.groups, 10, 17, 17))
            routputs = self.reshape(routputs, (self.groups, 20, 17, 17))
            routputs = self.regress_adjust(routputs)
            out1, out2 = coutputs, routputs

        elif self.is_trackinit is True and template is not None:

            template = self.transpose(template, (2, 0, 1))
            template = self.expand_dims(template, 0)
            template_feature = self.featureExtract(template)

            ckernal = self.conv1(template_feature)
            ckernal = self.reshape(ckernal.view(self.groups, 2 * self.k, 256, 4, 4), (-1, 256, 4, 4))

            rkernal = self.conv2(template_feature)
            rkernal = self.reshape(rkernal.view(self.groups, 4 * self.k, 256, 4, 4), (-1, 256, 4, 4))
            out1, out2 = ckernal, rkernal
        elif self.is_track is True and detection is not None:
            detection = self.transpose(detection, (2, 0, 1))
            detection = self.expand_dims(detection, 0)
            detection_feature = self.featureExtract(detection)
            cinput = self.reshape(self.conv3(detection_feature), (1, -1, 20, 20))
            rinput = self.reshape(self.conv4(detection_feature), (1, -1, 20, 20))

            c_features = self.op_split_input(cinput)
            c_weights = self.op_split_krenal(ckernal)
            r_features = self.op_split_input(rinput)
            r_weights = self.op_split_krenal(rkernal)
            coutputs = ()
            routputs = ()
            for i in range(self.groups):
                coutputs = coutputs + (self.conv2d_cout(c_features[i], c_weights[i]),)
                routputs = routputs + (self.conv2d_rout(r_features[i], r_weights[i]),)
            coutputs = self.op_concat(coutputs)
            routputs = self.op_concat(routputs)
            coutputs = self.reshape(coutputs, (self.groups, 10, 17, 17))
            routputs = self.reshape(routputs, (self.groups, 20, 17, 17))
            routputs = self.regress_adjust(routputs)
            pred_score = self.transpose(
                self.reshape(coutputs, (-1, 2, 1445)), (0, 2, 1))
            pred_regression = self.transpose(
                self.reshape(routputs, (-1, 4, 1445)), (0, 2, 1))
            pred_score = self.softmax(pred_score)[0, :, 1]
            out1, out2 = pred_score, pred_regression
        else:
            out1, out2 = template, detection
        return out1, out2



GRADIENT_CLIP_TYPE = 1
GRADIENT_CLIP_VALUE = 5.0
clip_grad = C.MultitypeFuncGraph("clip_grad")


@clip_grad.register("Number", "Number", "Tensor")
def _clip_grad(clip_type, clip_value, grad):
    """
    Clip gradients.

    Inputs:
        clip_type (int): The way to clip, 0 for 'value', 1 for 'norm'.
        clip_value (float): Specifies how much to clip.
        grad (tuple[Tensor]): Gradients.

    Outputs:
        tuple[Tensor], clipped gradients.
    """
    if clip_type not in (0, 1):
        return grad
    dt = F.dtype(grad)
    if clip_type == 0:
        new_grad = C.clip_by_value(grad, F.cast(F.tuple_to_array((-clip_value,)), dt),
                                   F.cast(F.tuple_to_array((clip_value,)), dt))
    else:
        new_grad = nn.ClipByNorm()(grad, F.cast(F.tuple_to_array((clip_value,)), dt))
    return new_grad
class MyTrainOneStepCell(nn.Cell):
    """MyTrainOneStepCell"""
    def __init__(self, network, optimizer, sens=1.0):
        super(MyTrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_train()

        self.network.set_grad()

        self.network.add_flags(defer_inline=True)
        self.weights = optimizer.parameters
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)

        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer = F.identity
        self.parallel_mode = _get_parallel_mode()
        self.hyper_map = C.HyperMap()
        self.cast = ops.Cast()
        if self.parallel_mode in (ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL):
            self.reducer_flag = True
        if self.reducer_flag:
            mean = _get_gradients_mean()
            degree = _get_device_num()
            self.grad_reducer = DistributedGradReducer(self.weights, mean, degree)

    def construct(self, template, detection, target):
        weights = self.weights

        loss = self.network(template, detection, target)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(template, detection, target, sens)
        grads = self.hyper_map(F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)
        if self.reducer_flag:
            grads = self.grad_reducer(grads)
        return F.depend(loss, self.optimizer(grads))

class BuildTrainNet(nn.Cell):
    def __init__(self, network, criterion):
        super(BuildTrainNet, self).__init__()
        self.network = network
        self.criterion = criterion
    def construct(self, template, detection, target):
        cout, rout = self.network(template=template, detection=detection, ckernal=detection, rkernal=detection)
        total_loss = self.criterion(cout, rout, target)
        return total_loss
