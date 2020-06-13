import numpy as np
import mindspore.context as context
from mindspore import Tensor, Parameter
from mindspore.nn import Cell, Composite
from mindspore.ops import operations as P
import mindspore.ops.composite as C
import logging
from mindspore._checkparam import ParamValidator as validator
from mindspore.ops import Primitive
from mindspore._checkparam import Rel
from mindspore.common.initializer import initializer
from mindspore.nn.composite_ops.composite_ops import InplaceAssign

log = logging.getLogger("ME")
log.setLevel(level=logging.DEBUG)
context.set_context(mode=context.GRAPH_MODE, save_graphs=True, device_target="Ascend")

class DtypeTest(Composite):
    def __init__(self, fix_precision = "float16"):
        super(DtypeTest, self).__init__()
        self.sum = P.ReduceSum()
        self.pow = P.Pow()
        self.sum.add_prim_attr("fix_precision", fix_precision)
        self.pow.add_prim_attr("fix_precision", fix_precision)

    def construct(self, x):
        res = self.sum(x, (0,))
        res = self.pow(res, 2.0)
        return res


class Net(Cell):
    def __init__(self, fix_precision = "float16"):
        super(Net, self).__init__()
        self.net = DtypeTest(fix_precision)

    def construct(self, x):
        return P.Neg()(self.net(x))

class FusedBatchNorm(Composite):
    def __init__(self,
                 mode=0,
                 epsilon=1e-5,
                 momentum=0.1,
                 fix_precision = "float16"):
        super(FusedBatchNorm, self).__init__()
        self.mode = validator.check_integer('mode', mode, [0, 1], Rel.IN)
        self.epsilon = validator.check_number_range('epsilon', epsilon, 0, 1, Rel.INC_RIGHT)
        self.momentum = validator.check_number_range('momentum', momentum, 0, 1, Rel.INC_BOTH)
        self.reduce1 = P.ReduceSum()
        self.reduce2 = P.ReduceSum()
        self.reshape1 = P.Reshape()
        self.reshape2 = P.Reshape()
        self.reshape3 = P.Reshape()
        self.reshape4 = P.Reshape()
        self.pow1 = P.Pow()
        self.pow2 = P.Pow()
        self.mul1 = P.Mul()
        self.mul2 = P.Mul()
        self.mul3 = P.Mul()
        self.mul4 = P.Mul()
        self.mul5 = P.Mul()
        self.mul6 = P.Mul()
        self.neg = P.Neg()
        self.sub1 = P.Sub()
        self.sub2 = P.Sub()
        self.rsqrt = P.Rsqrt()
        self.add1 = P.TensorAdd()
        self.add2 = P.TensorAdd()
        self.add3 = P.TensorAdd()
        self.add4 = P.TensorAdd()
        self.inplaceAssign1 = InplaceAssign()
        self.inplaceAssign2 = InplaceAssign()
        for _, value in vars(self).items():
            if isinstance(value, Primitive):
                value.add_prim_attr("fix_precision", fix_precision)

    def construct(self, x, scale, b, moving_mean, moving_variance):
        axes = (3, 2, 0) # NCHW
        # axes = (2, 1, 0) # NHWC

        shape = P.Shape()(x)
        value_num = 1
        for axis in axes:
            value_num *= shape[axis]
        # value_num = 4.0 * 4.0 * 16.0 # NCHW

        avg_num = 1.0 / P.Fill()(P.DType()(x), (1, ), value_num)

        data_square = self.pow1(x, 2.0)

        # cal mean
        data_sum =self.reduce1(x, axes)
        data_square_sum = self.reduce2(data_square, axes)

        data_mean = self.mul1(data_sum, avg_num)
        #data_mean = data_sum * avg_num
        data_square_mean = self.mul2(data_square_sum, avg_num)
        data_mean_square = self.pow2(data_mean, 2.0)

        # cal variance
        data_variance = self.sub1(data_square_mean, data_mean_square)

        def update_by_moving_average(hat_z, z, momentum):
            run = self.mul5(hat_z, momentum)
            now = self.mul6(z, 1.0 - momentum)
            return self.add4(run, now)

        _moving_mean = update_by_moving_average(moving_mean, data_mean, self.momentum)
        _moving_variance = update_by_moving_average(moving_variance,
                                                   data_variance, self.momentum)

        # var + eps
        veps_no_bc = self.add1(data_variance, self.epsilon)

        # rsqrt(var + eps)
        rsveps_no_bc = self.rsqrt(veps_no_bc)

        # -mean
        mean2_no_bc = self.neg(data_mean)

        mid_shape = (1, shape[1], 1, 1)

        # scale * (x + mean) / sqrt(var + eps) + b
        dmean = self.add2(x, self.reshape1(mean2_no_bc, mid_shape)) # broadcast result error
        dmsve = self.mul3(dmean, self.reshape2(rsveps_no_bc, mid_shape))

        dmsveg = self.mul4(dmsve, self.reshape3(scale, mid_shape))
        outs = self.add3(dmsveg, self.reshape4(b, mid_shape))

        #outs = self.inplaceAssign1(moving_mean, _moving_mean, outs)
        #outs = self.inplaceAssign2(moving_variance, _moving_variance, outs)
        outs = InplaceAssign()(moving_mean, _moving_mean, outs)
        outs = InplaceAssign()(moving_variance, _moving_variance, outs)
        return outs


class Net_BN(Cell):
    def __init__(self, fix_precision = "float16"):
        super(Net_BN, self).__init__()
        self.bn = FusedBatchNorm(fix_precision=fix_precision)
        self.gamma = Parameter(initializer('ones', [4]), name='gamma')
        self.beta = Parameter(initializer('zeros', [4]), name='beta')
        self.mean = Parameter(initializer('ones', [4]), name='mean')
        self.variance = Parameter(initializer('zeros', [4]), name='variance')

    def construct(self, x):
        return self.bn(x, self.gamma, self.beta, self.mean, self.variance)


def test_dtype_test_float16():
    x = np.array([1.0, 2.0, 3.0]).astype(np.float16)
    net = Net()
    result = net(Tensor(x))
    print("=======================================")
    print("x: {}".format(x))
    print("result: {}".format(result))
    print("=======================================")

def test_dtype_test_float32():
    x = np.array([1.0, 2.0, 3.0]).astype(np.float16)
    net = Net(fix_precision = "float32")
    result = net(Tensor(x))
    print("=======================================")
    print("x: {}".format(x))
    print("result: {}".format(result))
    print("=======================================")

def test_composite_bn():
    x = np.random.normal(0, 1, [16, 4, 4, 4]).astype(np.float32)
    net = Net_BN(fix_precision = "float16")
    #net1 = Net1()

    output = net(Tensor(x))
    # output1 = net1(Tensor(x))
    print("=======================================")
    print("x:\n{}".format(x))
    print("output:\n{}".format(output))
    print("=======================================")
    # print("x:\n{}".format(x))
    # print("output1:\n{}".format(output1))
    # print("=======================================")


#test_dtype_test_float16()
#test_dtype_test_float32()
test_composite_bn()
