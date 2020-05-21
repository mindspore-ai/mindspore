from mindspore.nn.loss.loss import _Loss
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore import Tensor
from mindspore.common import dtype as mstype
import mindspore.nn as nn

class CrossEntropy(_Loss):
    def __init__(self, smooth_factor=0., num_classes=1000):
        super(CrossEntropy, self).__init__()
        self.onehot = P.OneHot()
        self.on_value = Tensor(1.0 - smooth_factor, mstype.float32)
        self.off_value = Tensor(1.0 * smooth_factor / (num_classes -1), mstype.float32)
        #self.cast = P.Cast()
        self.ce = nn.SoftmaxCrossEntropyWithLogits()
        self.mean = P.ReduceMean(False)
    def construct(self, logit, label):
        #one_hot_label = self.onehot(self.cast(label, mstype.int32),
        #                F.shape(logit)[1], self.on_value, self.off_value)、
        one_hot_label = self.onehot(label, F.shape(logit)[1], self.on_value, self.off_value)
        loss = self.ce(logit, one_hot_label)
        loss = self.mean(loss, 0)
        return loss
