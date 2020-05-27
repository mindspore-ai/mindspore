# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0(the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http:  // www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Operators for quantization."""

from ..._checkparam import Validator as validator
from ..._checkparam import Rel
from ..primitive import PrimitiveWithInfer, prim_attr_register
from ...common import dtype as mstype

__all__ = ["FakeQuantWithMinMax",
           "FakeQuantWithMinMaxGrad",
           "FakeQuantWithMinMaxPerChannel",
           "FakeQuantWithMinMaxPerChannelGrad",
           "BatchNormFold",
           "BatchNormFoldGrad",
           "CorrectionMul",
           "CorrectionMulGrad",
           "BatchNormFold2",
           "BatchNormFold2Grad",
           "BatchNormFoldD",
           "BNTrainingReduce",
           "BatchNormFold2_D",
           "FakeQuantWithMinMaxUpdate",
           ]


class FakeQuantWithMinMax(PrimitiveWithInfer):
    r"""
    Simulate the quantize and dequantize operations in training time.

    Args:
        num_bits (int) : Number bits for aware quantilization. Default: 8.
        ema (bool): Use EMA algorithm update value min and max. Default: False.
        ema_decay (int) : EMA algorithm decay parameter. Default: 0.999.
        quant_delay (int): Quantilization delay parameter. Before delay step in training time not update
            simulate aware quantize funcion. After delay step in training time begin simulate the aware
            quantize funcion. Default: 0.
        symmetric (bool): Quantization algorithm use symmetric or not. Default: False.
        narrow_range (bool): Quantization algorithm use narrow range or not. Default: False.
        training (bool): Training the network or not. Default: True.

    Inputs:
        - **x** (Tensor) : float32 Tensor representing the shape of the output tensor.
        - **min** (Tensor) : Value of the min range of the input data x.
        - **max** (Tensor) : Value of the max range of the input data x.

    Outputs:
        - Tensor: Simulate quantize tensor of x.

    Examples:
        >>> input_tensor = Tensor(np.random.rand(3, 16, 5, 5), mstype.float32)
        >>> min_tensor = Tensor(np.array([-6]), mstype.float32)
        >>> max_tensor = Tensor(np.array([6]), mstype.float32)
        >>> output_tensor = P.FakeQuantWithMinMax(num_bits=8)(input_tensor, min_tensor, max_tensor)
    """
    support_quant_bit = [4, 7, 8]

    @prim_attr_register
    def __init__(self, num_bits=8, ema=False, ema_decay=0.999, quant_delay=0, symmetric=False, narrow_range=False,
                 training=True):
        """init FakeQuantWithMinMax OP"""
        if num_bits not in self.support_quant_bit:
            raise ValueError(f"For '{self.name}' attr \'num_bits\' is not support.")
        if ema and not ema_decay:
            raise ValueError(f"For '{self.name}' attr \'ema\' and \'ema_decay\' should set together.")

        self.ema = validator.check_value_type('ema', ema, (bool,), self.name)
        self.symmetric = validator.check_value_type('symmetric', symmetric, (bool,), self.name)
        self.narrow_range = validator.check_value_type('narrow_range', narrow_range, (bool,), self.name)
        self.training = validator.check_value_type('training', training, (bool,), self.name)
        self.ema_decay = validator.check_number_range('ema_decay', ema_decay, 0, 1, Rel.INC_BOTH, self.name)
        self.num_bits = validator.check_integer('num_bits', num_bits, 0, Rel.GT, self.name)
        self.quant_delay = validator.check_value_type('quant_delay', quant_delay, (int,), self.name)
        self.init_prim_io_names(inputs=['x', 'min', 'max'],
                                outputs=['out'])

    def infer_shape(self, x_shape, min_shape, max_shape):
        validator.check_integer("x rank", len(x_shape), 1, Rel.GT, self.name)
        validator.check("min shape", min_shape, "max shape", max_shape, Rel.EQ, self.name)
        validator.check_integer("min rank", len(min_shape), 1, Rel.EQ, self.name)
        return x_shape

    def infer_dtype(self, x_type, min_type, max_type):
        valid_types = (mstype.float16, mstype.float32)
        validator.check_tensor_type_same({"x": x_type}, valid_types, self.name)
        validator.check_tensor_type_same({"min": min_type}, valid_types, self.name)
        validator.check_tensor_type_same({"max": max_type}, valid_types, self.name)
        return x_type


class FakeQuantWithMinMaxGrad(PrimitiveWithInfer):
    r"""
    Performs grad of FakeQuantWithMinMax operation.

    Examples:
        >>> fake_min_max_grad = P.FakeQuantWithMinMaxGrad()
        >>> dout = Tensor(np.array([[-2.3, 1.2], [5.7, 0.2]]), mindspore.float32)
        >>> input_x = Tensor(np.array([[18, -23], [0.2, 6]]), mindspore.float32)
        >>> _min = Tensor(np.array([-4]), mindspore.float32)
        >>> _max = Tensor(np.array([2]), mindspore.float32)
        >>> result = fake_min_max_grad(dout, input_x, _min, _max)
    """
    support_quant_bit = [4, 8]

    @prim_attr_register
    def __init__(self, num_bits=8, quant_delay=0):
        if num_bits not in self.support_quant_bit:
            raise ValueError(f"For '{self.name}' attr \'num_bits\' is not support.")

        self.quant_delay = validator.check_value_type('quant_delay', quant_delay, (int,), self.name)
        self.num_bits = validator.check_integer('num_bits', num_bits, 0, Rel.GT, self.name)
        self.init_prim_io_names(inputs=['dout', 'x', 'min', 'max'], outputs=['dx'])

    def infer_shape(self, dout_shape, x_shape, min_shape, max_shape):
        validator.check("dout shape", dout_shape, "x shape", x_shape, Rel.EQ, self.name)
        validator.check("min shape", min_shape, "max shape", max_shape, Rel.EQ, self.name)
        validator.check_integer("min rank", len(min_shape), 1, Rel.EQ, self.name)
        return dout_shape

    def infer_dtype(self, dout_type, x_type, min_type, max_type):
        valid_types = (mstype.float16, mstype.float32)
        validator.check_tensor_type_same({"dout": dout_type}, valid_types, self.name)
        validator.check_tensor_type_same({"x": x_type}, valid_types, self.name)
        validator.check_tensor_type_same({"min": min_type}, valid_types, self.name)
        validator.check_tensor_type_same({"max": max_type}, valid_types, self.name)
        return dout_type


class FakeQuantWithMinMaxPerChannel(PrimitiveWithInfer):
    r"""
    Simulate the quantize and dequantize operations in training time base on per channel.

    Args:
        num_bits (int) : Number bits to quantilization. Default: 8.
        ema (bool): Use EMA algorithm update tensor min and tensor max. Default: False.
        ema_decay (int) : EMA algorithm decay parameter. Default: 0.999.
        quant_delay (int): Quantilization delay  parameter. Before delay step in training time not
            update the weight data to simulate quantize operation. After delay step in training time
            begin simulate the quantize operation. Default: 0.
        symmetric (bool): Quantization algorithm use symmetric or not. Default: False.
        narrow_range (bool): Quantization algorithm use narrow range or not. Default: False.
        training (bool): Training the network or not. Default: True.

    Inputs:
        - **x** (Tensor) : 4-D float32 Tensor representing the shape of the output tensor.
        - **min** (int, float) : Value of the min range of the input data.
        - **max** (int, float) : Value of the max range of the input data.

    Outputs:
        - Tensor, has the same type as input.

    Examples:
        >>> fake_quant = P.FakeQuantWithMinMaxPerChannel()
        >>> input_x = Tensor(np.array([3, 4, 5, -2, -3, -1]).reshape(3, 2), mindspore.float32)
        >>> _min = Tensor(np.linspace(-2, 2, 12).reshape(3, 2, 2), mindspore.float32)
        >>> _max = Tensor(np.linspace(8, 12, 12).reshape(3, 2, 2), mindspore.float32)
        >>> result = fake_quant(input_x, _min, _max)
    """
    support_quant_bit = [4, 8]
    channel_axis = 0

    @prim_attr_register
    def __init__(self, num_bits=8, ema=False, ema_decay=0.999, quant_delay=0, symmetric=False, narrow_range=False,
                 training=True):
        """init FakeQuantWithMinMaxPerChannel OP"""
        if num_bits not in self.support_quant_bit:
            raise ValueError(f"For '{self.name}' Attr \'num_bits\' is not support.")
        if ema and not ema_decay:
            raise ValueError(f"For '{self.name}' attr \'ema\' and \'ema_decay\' should set together.")

        self.ema = validator.check_value_type('ema', ema, (bool,), self.name)
        self.symmetric = validator.check_value_type('symmetric', symmetric, (bool,), self.name)
        self.narrow_range = validator.check_value_type('narrow_range', narrow_range, (bool,), self.name)
        self.training = validator.check_value_type('training', training, (bool,), self.name)
        self.ema_decay = validator.check_number_range('ema_decay', ema_decay, 0, 1, Rel.INC_BOTH, self.name)
        self.num_bits = validator.check_integer('num_bits', num_bits, 0, Rel.GT, self.name)
        self.quant_delay = validator.check_value_type('quant_delay', quant_delay, (int,), self.name)
        self.init_prim_io_names(inputs=['x', 'min', 'max'], outputs=['out'])

    def infer_shape(self, x_shape, min_shape, max_shape):
        validator.check_integer("x rank", len(x_shape), 1, Rel.GT, self.name)
        validator.check_integer("min shape[0]", min_shape[0], x_shape[self.channel_axis], Rel.EQ, self.name)
        validator.check_integer("max shape[0]", max_shape[0], x_shape[self.channel_axis], Rel.EQ, self.name)
        return x_shape

    def infer_dtype(self, x_type, min_type, max_type):
        valid_types = (mstype.float16, mstype.float32)
        validator.check_tensor_type_same({"x": x_type}, valid_types, self.name)
        validator.check_tensor_type_same({"min": min_type}, valid_types, self.name)
        validator.check_tensor_type_same({"max": max_type}, valid_types, self.name)
        return x_type


class FakeQuantWithMinMaxPerChannelGrad(PrimitiveWithInfer):
    r"""
    Performs grad of FakeQuantWithMinMaxPerChannel operation.

    Examples:
        >>> fqmmpc_grad = P.FakeQuantWithMinMaxPerChannelGrad()
        >>> input_x = Tensor(np.random.randint(-4, 4, (2, 3, 4)), mindspore.float32)
        >>> dout = Tensor(np.random.randint(-2, 2, (2, 3, 4)), mindspore.float32)
        >>> _min = Tensor(np.random.randint(-8, 2, (2, 3, 4)), mindspore.float32)
        >>> _max = Tensor(np.random.randint(-2, 8, (2, 3, 4)), mindspore.float32)
        >>> result = fqmmpc_grad(dout, input_x, _min, _max)
    """
    support_quant_bit = [4, 8]

    @prim_attr_register
    def __init__(self, num_bits=8, quant_delay=0):
        """init FakeQuantWithMinMaxPerChannel Fill"""
        if num_bits not in self.support_quant_bit:
            raise ValueError(f"For '{self.name}' attr \'num_bits\' is not support.")

        self.quant_delay = validator.check_value_type('quant_delay', quant_delay, (int,), self.name)
        self.num_bits = validator.check_integer('num_bits', num_bits, 0, Rel.GT, self.name)
        self.init_prim_io_names(inputs=['dout', 'x', 'min', 'max'], outputs=['dx'])

    def infer_shape(self, dout_shape, x_shape, min_shape, max_shape):
        validator.check("dout shape", dout_shape, "x shape", x_shape)
        validator.check("min shape", min_shape, "max shape", max_shape)
        return dout_shape

    def infer_dtype(self, dout_type, x_type, min_type, max_type):
        valid_types = (mstype.float16, mstype.float32)
        validator.check_tensor_type_same({"dout": dout_type}, valid_types, self.name)
        validator.check_tensor_type_same({"x": x_type}, valid_types, self.name)
        validator.check_tensor_type_same({"min": min_type}, valid_types, self.name)
        validator.check_tensor_type_same({"max": max_type}, valid_types, self.name)
        return dout_type


class BatchNormFold(PrimitiveWithInfer):
    """
    Batch normalization folded.

    Args:
        momentum (float): Momentum value should be [0, 1]. Default: 0.1.
        epsilon (float): A small float number to avoid dividing by 0. 1e-5 if dtype in
            float32 else 1e-3. Default: 1e-5.
        is_training (bool): In training mode set True, else set False. Default: True.
        freeze_bn (int): Delay in steps at which computation switches from regular batch
            norm to frozen mean and std. Default: 0.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C)`.
        - **mean** (Tensor) - Tensor of shape :math:`(C,)`.
        - **variance** (Tensor) - Tensor of shape :math:`(C,)`.
        - **global_step** (Tensor) - Tensor to record current global step.

    Outputs:
        Tuple of 4 Tensor, the normalized input and the updated parameters.

        - **batch_mean** (Tensor) - Tensor of shape :math:`(C,)`.
        - **batch_std** (Tensor) - Tensor of shape :math:`(C,)`.
        - **running_mean** (Tensor) - Tensor of shape :math:`(C,)`.
        - **running_std** (Tensor) - Tensor of shape :math:`(C,)`.

    Examples:
        >>> batch_norm_fold = P.BatchNormFold()
        >>> input_x = Tensor(np.array([1, 2, -1, -2, -2, 1]).reshape(2, 3), mindspore.float32)
        >>> mean = Tensor(np.array([0.5, -1, 1,]), mindspore.float32)
        >>> variance = Tensor(np.array([0.36, 0.4, 0.49]), mindspore.float32)
        >>> global_step = Tensor(np.arange(6), mindspore.int32)
        >>> batch_mean, batch_std, running_mean, running_std = batch_norm_fold(input_x, mean, variance, global_step)
    """
    channel_axis = 1

    @prim_attr_register
    def __init__(self, momentum=0.1, epsilon=1e-5, is_training=True, freeze_bn=0):
        """init batch norm fold layer"""
        self.momentum = validator.check_number_range('momentum', momentum, 0, 1, Rel.INC_BOTH, self.name)
        self.epsilon = validator.check_float_positive('epsilon', epsilon, self.name)
        self.is_training = validator.check_value_type('is_training', is_training, (bool,), self.name)
        self.freeze_bn = validator.check_value_type('freeze_bn', freeze_bn, (int,), self.name)

        self.init_prim_io_names(inputs=['x', 'mean', 'variance', 'global_step'],
                                outputs=['batch_mean', 'batch_std', 'running_mean', 'running_std'])

    def infer_shape(self, x_shape, mean_shape, variance_shape, global_step_shape):
        validator.check("mean shape", mean_shape, "gamma_shape", variance_shape, Rel.EQ, self.name)
        validator.check("mean_shape[0]", mean_shape[0], "input channel", x_shape[self.channel_axis], Rel.EQ, self.name)
        validator.check_integer("global_step rank", len(global_step_shape), 1, Rel.EQ, self.name)
        return mean_shape, mean_shape, mean_shape, mean_shape

    def infer_dtype(self, x_type, mean_type, variance_type, global_step_type):
        validator.check("input type", x_type, "mean type", mean_type)
        validator.check("input type", x_type, "variance type", variance_type)
        args = {"x": x_type, "mean": mean_type, "variance": variance_type}
        validator.check_tensor_type_same(args, (mstype.float16, mstype.float32), self.name)
        validator.check_tensor_type_same({"global_step": global_step_type}, (mstype.int32,), self.name)
        return x_type, x_type, x_type, x_type


class BatchNormFoldGrad(PrimitiveWithInfer):
    r"""
    Performs grad of BatchNormFold operation.

    Examples:
        >>> batch_norm_fold_grad = P.BatchNormFoldGrad()
        >>> d_batch_mean = Tensor(np.random.randint(-2., 2., (1, 2, 2, 3)), mindspore.float32)
        >>> d_batch_std = Tensor(np.random.randn(1, 2, 2, 3), mindspore.float32)
        >>> input_x = Tensor(np.random.randint(0, 256, (4, 1, 4, 6)), mindspore.float32)
        >>> batch_mean = Tensor(np.random.randint(-8., 8., (1, 2, 2, 3)), mindspore.float32)
        >>> batch_std = Tensor(np.random.randint(0, 12, (1, 2, 2, 3)), mindspore.float32)
        >>> global_step = Tensor([2], mindspore.int32)
        >>> result = batch_norm_fold_grad(d_batch_mean, d_batch_std, input_x, batch_mean, batch_std, global_step)
    """
    channel_axis = 1

    @prim_attr_register
    def __init__(self, epsilon=1e-5, is_training=True, freeze_bn=0):
        """init BatchNormGrad layer"""
        self.is_training = validator.check_value_type('is_training', is_training, (bool,), self.name)
        self.freeze_bn = validator.check_value_type('freeze_bn', freeze_bn, (int,), self.name)
        self.epsilon = validator.check_float_positive('epsilon', epsilon, self.name)
        self.init_prim_io_names(inputs=['d_batch_mean', 'd_batch_std', 'x', 'batch_mean', 'batch_std', 'global_step'],
                                outputs=['dx'])

    def infer_shape(self, d_batch_mean_shape, d_batch_std_shape, x_shape, batch_mean_shape, batch_std_shape,
                    global_step_shape):
        validator.check("d_batch_mean shape", d_batch_mean_shape,
                        "d_batch_std shape", d_batch_std_shape, Rel.EQ, self.name)
        validator.check("d_batch_mean shape", d_batch_mean_shape,
                        "batch_mean shape", batch_mean_shape, Rel.EQ, self.name)
        validator.check("d_batch_mean shape", d_batch_mean_shape,
                        "batch_std shape", batch_std_shape, Rel.EQ, self.name)
        validator.check("d_batch_mean_shape[0]", d_batch_mean_shape[0],
                        "input channel", x_shape[self.channel_axis], Rel.EQ, self.name)
        validator.check_integer("global_step rank", len(global_step_shape), 1, Rel.EQ, self.name)
        return x_shape

    def infer_dtype(self, d_batch_mean_type, d_batch_std_type, x_type, batch_mean_type, batch_std_type,
                    global_step_type):
        args = {"input": x_type, "d_batch_mean": d_batch_mean_type, "d_batch_std": d_batch_std_type,
                "batch_mean": batch_mean_type, "batch_std": batch_std_type}
        validator.check_tensor_type_same(args, (mstype.float16, mstype.float32), self.name)
        validator.check_tensor_type_same({"global_step": global_step_type}, (mstype.int32,), self.name)
        return x_type


class CorrectionMul(PrimitiveWithInfer):
    """
    Scale the weights with a correction factor to the long term statistics
    prior to quantization. This ensures that there is no jitter in the quantized weights
    due to batch to batch variation.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C)`.
        - **batch_std** (Tensor) - Tensor of shape :math:`(C,)`.
        - **running_std** (Tensor) - Tensor of shape :math:`(C,)`.

    Outputs:
        - **out** (Tensor) - Tensor has the same shape as x.

    Examples:
        >>> correction_mul = P.CorrectionMul()
        >>> input_x = Tensor(np.random.randint(-8, 12, (3, 4)), mindspore.float32)
        >>> batch_std = Tensor(np.array([1.5, 3, 2]), mindspore.float32)
        >>> running_std = Tensor(np.array([2, 1.2, 0.5]), mindspore.float32)
        >>> out = correction_mul(input_x, batch_std, running_std)
    """

    @prim_attr_register
    def __init__(self, channel_axis=0):
        """init correction mul layer"""
        self.channel_axis = channel_axis
        self.init_prim_io_names(inputs=['x', 'batch_std', 'running_std'],
                                outputs=['out'])

    def infer_shape(self, x_shape, batch_std_shape, running_std_shape):
        validator.check("batch_std shape", batch_std_shape, "running_std shape", running_std_shape, Rel.EQ, self.name)
        validator.check("batch_std_shape[0]", batch_std_shape[0], "x_shape channel size", x_shape[self.channel_axis],
                        Rel.EQ, self.name)
        return x_shape

    def infer_dtype(self, x_type, batch_std_type, running_std_type):
        args = {"x": x_type, "batch_std": batch_std_type, "running_std": running_std_type}
        validator.check_tensor_type_same(args, (mstype.float16, mstype.float32), self.name)
        return x_type


class CorrectionMulGrad(PrimitiveWithInfer):
    r"""
    Performs grad of CorrectionMul operation.

    Examples:
        >>> correction_mul_grad = P.CorrectionMulGrad()
        >>> dout = Tensor(np.array([1.5, -2.2, 0.7, -3, 1.6, 2.8]).reshape(2, 1, 1, 3), mindspore.float32)
        >>> input_x = Tensor(np.random.randint(0, 256, (2, 1, 1, 3)), mindspore.float32)
        >>> gamma = Tensor(np.array([0.2, -0.2, 2.5, -1.]).reshape(2, 1, 2), mindspore.float32)
        >>> running_std = Tensor(np.array([1.2, 0.1, 0.7, 2.3]).reshape(2, 1, 2), mindspore.float32)
        >>> result = correction_mul_grad(dout, input_x, gamma, running_std)
    """

    @prim_attr_register
    def __init__(self, channel_axis=0):
        """init correction mul layer"""
        self.channel_axis = channel_axis
        self.init_prim_io_names(inputs=['dout', 'x', 'gamma', 'running_std'],
                                outputs=['dx', 'd_gamma'])

    def infer_shape(self, dout_shape, x_shape, gamma_shape, running_std_shape):
        validator.check("dout shape", dout_shape, "x_shape x", x_shape, Rel.EQ, self.name)
        validator.check("gamma_shape[0]", gamma_shape[0], "dout channel size", dout_shape[self.channel_axis],
                        Rel.EQ, self.name)
        validator.check("running_std_shape[0]", running_std_shape[0],
                        "dout channel size", dout_shape[self.channel_axis], Rel.EQ, self.name)
        return x_shape, gamma_shape

    def infer_dtype(self, dout_type, x_type, gamma_type, running_std_type):
        args = {"dout": dout_type, "x": x_type, "gamma": gamma_type, "running_std": running_std_type}
        validator.check_tensor_type_same(args, (mstype.float16, mstype.float32), self.name)
        return x_type, x_type


class BatchNormFold2(PrimitiveWithInfer):
    """
    Scale the bias with a correction factor to the long term statistics
    prior to quantization. This ensures that there is no jitter in the quantized bias
    due to batch to batch variation.

    Inputs:
        - **x** (Tensor)  - Tensor of shape :math:`(N, C)`.
        - **beta** (Tensor) - Tensor of shape :math:`(C,)`.
        - **gamma** (Tensor) - Tensor of shape :math:`(C,)`.
        - **batch_std** (Tensor) - Tensor of shape :math:`(C,)`.
        - **batch_mean** (Tensor) - Tensor of shape :math:`(C,)`.
        - **running_std** (Tensor) - Tensor of shape :math:`(C,)`.
        - **running_mean** (Tensor) - Tensor of shape :math:`(C,)`.
        - **global_step** (Tensor) - Tensor to record current global step.

    Outputs:
        - **y** (Tensor) - Tensor has the same shape as x.

    Examples:
        >>> batch_norm_fold2 = P.BatchNormFold2()
        >>> input_x = Tensor(np.random.randint(-6, 6, (4, 3)), mindspore.float32)
        >>> beta = Tensor(np.array([0.2, -0.1, 0.25]), mindspore.float32)
        >>> gamma = Tensor(np.array([-0.1, -0.25, 0.1]), mindspore.float32)
        >>> batch_std = Tensor(np.array([0.1, 0.2, 0.1]), mindspore.float32)
        >>> batch_mean = Tensor(np.array([0, 0.05, 0.2]), mindspore.float32)
        >>> running_std = Tensor(np.array([0.1, 0.1, 0.3]), mindspore.float32)
        >>> running_mean = Tensor(np.array([-0.1, 0, -0.1]), mindspore.float32)
        >>> global_step = Tensor(np.random.randint(1, 8, (8, )), mindspore.int32)
        >>> result = batch_norm_fold2(input_x, beta, gamma, batch_std, batch_mean,
        >>>                           running_std, running_mean, global_step)
    """
    channel_axis = 1

    @prim_attr_register
    def __init__(self, freeze_bn=0):
        """init conv2d fold layer"""
        self.freeze_bn = validator.check_value_type('freeze_bn', freeze_bn, (int,), self.name)
        self.init_prim_io_names(inputs=['x', 'beta', 'gamma', 'batch_std', 'batch_mean',
                                        'running_std', 'running_mean', 'global_step'],
                                outputs=['y'])

    def infer_shape(self, x_shape, beta_shape, gamma_shape, batch_std_shape, running_std_shape, batch_mean_shape,
                    running_mean_shape, global_step_shape):
        validator.check("batch_std shape", batch_std_shape, "running_std shape", running_std_shape, Rel.EQ, self.name)
        validator.check("batch_std shape", batch_std_shape, "batch_mean shape", batch_mean_shape, Rel.EQ, self.name)
        validator.check("batch_std shape", batch_std_shape, "beta shape", beta_shape, Rel.EQ, self.name)
        validator.check("batch_std shape", batch_std_shape, "running_mean shape", running_mean_shape, Rel.EQ, self.name)
        validator.check("batch_std shape", batch_std_shape, "batch_mean shape", gamma_shape, Rel.EQ, self.name)
        validator.check("batch_std_shape[0]", batch_std_shape[0], "x_shape channel size", x_shape[self.channel_axis],
                        Rel.EQ, self.name)
        validator.check_integer("global_step rank", len(global_step_shape), 1, Rel.EQ, self.name)
        return x_shape

    def infer_dtype(self, x_type, beta_type, gamma_type, batch_std_type, running_std_type, batch_mean_type,
                    running_mean_type, global_step_type):
        args = {"batch_std": batch_std_type, "running_std": running_std_type, "batch_mean": batch_mean_type,
                "beta": beta_type, "running_mean": running_mean_type, "gamma": gamma_type, "x": x_type}
        validator.check_tensor_type_same(args, (mstype.float16, mstype.float32), self.name)
        validator.check_tensor_type_same({"global_step": global_step_type}, (mstype.int32,), self.name)
        return x_type


class BatchNormFold2Grad(PrimitiveWithInfer):
    r"""
    Performs grad of CorrectionAddGrad operation.

    Examples:
        >>> bnf2_grad = P.BatchNormFold2Grad()
        >>> input_x = Tensor(np.arange(3*3*12*12).reshape(6, 3, 6, 12), mindspore.float32)
        >>> dout = Tensor(np.random.randint(-32, 32, (6, 3, 6, 12)), mindspore.float32)
        >>> gamma = Tensor(np.random.randint(-4, 4, (3, 1, 1, 2)), mindspore.float32)
        >>> batch_std = Tensor(np.random.randint(0, 8, (3, 1, 1, 2)), mindspore.float32)
        >>> batch_mean = Tensor(np.random.randint(-6, 6, (3, 1, 1, 2)), mindspore.float32)
        >>> running_std = Tensor(np.linspace(0, 2, 6).reshape(3, 1, 1, 2), mindspore.float32)
        >>> running_mean = Tensor(np.random.randint(-3, 3, (3, 1, 1, 2)), mindspore.float32)
        >>> global_step = Tensor(np.array([-2]), mindspore.int32)
        >>> result = bnf2_grad(dout, input_x, gamma, batch_std, batch_mean, running_std, running_mean, global_step)
    """
    channel_axis = 1

    @prim_attr_register
    def __init__(self, freeze_bn=0):
        """init MulFold layer"""
        self.freeze_bn = freeze_bn
        self.init_prim_io_names(inputs=['dout', 'x', 'gamma',
                                        'batch_std', 'batch_mean',
                                        'running_std', 'running_mean', 'global_step'],
                                outputs=['d_batch_std', 'd_batch_mean', 'd_beta', 'd_gamma', 'dx'])

    def infer_shape(self, dout_shape, x_shape, gamma_shape,
                    batch_std_shape, batch_mean_shape,
                    running_std_shape, running_mean_shape, global_step_shape):
        validator.check("batch_std shape", batch_std_shape, "batch_mean shape", batch_mean_shape, Rel.EQ, self.name)
        validator.check("batch_std shape", batch_std_shape, "running_std shape", running_std_shape, Rel.EQ, self.name)
        validator.check("batch_std shape", batch_std_shape, "running_mean shape", running_mean_shape, Rel.EQ, self.name)
        validator.check("batch_std shape", batch_std_shape, "gamma shape", gamma_shape, Rel.EQ, self.name)
        validator.check("batch_std size", batch_std_shape[0], "dout channel size", dout_shape[self.channel_axis],
                        Rel.EQ, self.name)
        validator.check_integer("global_step rank", len(global_step_shape), 1, Rel.EQ, self.name)
        return gamma_shape, gamma_shape, gamma_shape, gamma_shape, x_shape

    def infer_dtype(self, dout_type, x_type, gamma_type,
                    batch_std_type, batch_mean_type,
                    running_std_type, running_mean_type, global_step_type):
        validator.check("batch_std type", batch_std_type,
                        "batch_mean type", batch_mean_type)
        validator.check("batch_std type", batch_std_type,
                        "gamma type", gamma_type)
        validator.check("batch_std type", batch_std_type,
                        "running_std type", running_std_type)
        validator.check("batch_std type", batch_std_type,
                        "running_mean type", running_mean_type)
        validator.check("batch_std_type", batch_std_type,
                        "dout type", dout_type)
        args = {"batch_std": batch_std_type, "batch_mean": batch_mean_type, "gamma": gamma_type,
                "running_std": running_std_type, "running_mean": running_mean_type, "dout": dout_type}
        validator.check_tensor_type_same(args, (mstype.float16, mstype.float32), self.name)
        validator.check_tensor_type_same({"global_step": global_step_type}, (mstype.int32,), self.name)
        return gamma_type, gamma_type, gamma_type, gamma_type, gamma_type


class BatchNormFoldD(PrimitiveWithInfer):
    """Performs grad of _BatchNormFold operation."""

    @prim_attr_register
    def __init__(self, momentum=0.9, epsilon=1e-5, is_training=True, freeze_bn=0):
        """init _BatchNormFold layer"""
        from mindspore.ops._op_impl._custom_op import batchnorm_fold
        self.momentum = validator.check_number_range('momentum', momentum, 0, 1, Rel.INC_BOTH, self.name)
        self.epsilon = validator.check_float_positive('epsilon', epsilon, self.name)
        self.is_training = validator.check_value_type('is_training', is_training, (bool,), self.name)
        self.freeze_bn = validator.check_value_type('freeze_bn', freeze_bn, (int,), self.name)
        self.data_format = "NCHW"
        self.init_prim_io_names(inputs=['x', 'x_sum', 'x_square_sum', 'mean', 'variance'],
                                outputs=['batch_mean', 'batch_std', 'running_mean', 'running_std',
                                         'mean_updated', 'variance_updated'])

    def infer_shape(self, x_shape, x_sum_shape, x_square_sum_shape, mean_shape, variance_shape):
        validator.check("mean shape", mean_shape, "gamma_shape", variance_shape, Rel.EQ, self.name)
        validator.check("mean_shape[0]", mean_shape[0], "input channel", x_shape[1], Rel.EQ, self.name)
        return x_shape, mean_shape, mean_shape, mean_shape, mean_shape, mean_shape, mean_shape

    def infer_dtype(self, x_type, x_sum_type, x_square_sum_type, mean_type, variance_type):
        validator.check("input type", x_type, "mean type", mean_type)
        validator.check("input type", x_type, "variance type", variance_type)
        args = {"x": x_type, "mean": mean_type, "variance": variance_type}
        validator.check_tensor_type_same(args, (mstype.float16, mstype.float32), self.name)
        return x_type, x_type, x_type, x_type, x_type, x_type, x_type


class BatchNormFoldGradD(PrimitiveWithInfer):
    """Performs grad of _BatchNormFoldGrad operation."""

    @prim_attr_register
    def __init__(self, epsilon=1e-5, is_training=True, freeze_bn=0):
        """init _BatchNormFoldGrad layer"""
        from mindspore.ops._op_impl._custom_op import batchnorm_fold_grad
        self.epsilon = validator.check_float_positive('epsilon', epsilon, self.name)
        self.is_training = validator.check_value_type('is_training', is_training, (bool,), self.name)
        self.freeze_bn = validator.check_value_type('freeze_bn', freeze_bn, (int,), self.name)
        self.init_prim_io_names(inputs=['d_batch_mean', 'd_batch_std', 'x', 'batch_mean', 'batch_std'],
                                outputs=['dx'])

    def infer_shape(self, d_batch_mean_shape, d_batch_std_shape, x_shape, batch_mean_shape, batch_std_shape):
        validator.check("d_batch_mean shape", d_batch_mean_shape, "d_batch_std shape", d_batch_std_shape)
        validator.check("d_batch_mean shape", d_batch_mean_shape, "batch_mean shape", batch_mean_shape)
        validator.check("d_batch_mean shape", d_batch_mean_shape, "batch_std shape", batch_std_shape)
        validator.check("x_shape shape", d_batch_mean_shape[0], "input channel", x_shape[1])
        return x_shape

    def infer_dtype(self, d_batch_mean_type, d_batch_std_type, x_type, batch_mean_type, batch_std_type):
        validator.check("input type", x_type, "d_batch_mean type", d_batch_mean_type)
        validator.check("input type", x_type, "d_batch_std type", d_batch_std_type)
        validator.check("input type", x_type, "batch_mean type", batch_mean_type)
        validator.check("input type", x_type, "batch_std type", batch_std_type)
        args = {"input type": x_type}
        validator.check_tensor_type_same(args, (mstype.float16, mstype.float32), self.name)
        return x_type


class BNTrainingReduce(PrimitiveWithInfer):
    """
    reduce sum at axis [0, 2, 3].

    Inputs:
        - **x** (Tensor)  - Tensor of shape :math:`(N, C)`.

    Outputs:
        - **x_sum** (Tensor) - Tensor has the same shape as x.
        - **x_square_sum** (Tensor) - Tensor has the same shape as x.

    """

    @prim_attr_register
    def __init__(self):
        """init _BNTrainingReduce layer"""
        self.init_prim_io_names(inputs=['x'],
                                outputs=['x_sum', 'x_square_sum'])

    def infer_shape(self, x_shape):
        return [x_shape[1]], [x_shape[1]]

    def infer_dtype(self, x_type):
        return x_type, x_type


class BatchNormFold2_D(PrimitiveWithInfer):
    """
    Scale the bias with a correction factor to the long term statistics
    prior to quantization. This ensures that there is no jitter in the quantized bias
    due to batch to batch variation.

    Inputs:
        - **x** (Tensor)  - Tensor of shape :math:`(N, C)`.
        - **beta** (Tensor) - Tensor of shape :math:`(C,)`.
        - **gamma** (Tensor) - Tensor of shape :math:`(C,)`.
        - **batch_std** (Tensor) - Tensor of shape :math:`(C,)`.
        - **batch_mean** (Tensor) - Tensor of shape :math:`(C,)`.
        - **running_std** (Tensor) - Tensor of shape :math:`(C,)`.
        - **running_mean** (Tensor) - Tensor of shape :math:`(C,)`.
        - **global_step** (Tensor) - Tensor to record current global step.

    Outputs:
        - **y** (Tensor) - Tensor has the same shape as x.

    """
    channel_axis = 1

    @prim_attr_register
    def __init__(self, freeze_bn=0):
        """init conv2d fold layer"""
        from mindspore.ops._op_impl._custom_op import batchnorm_fold2
        self.init_prim_io_names(inputs=['x', 'beta', 'gamma', 'batch_std', 'batch_mean', 'running_std'],
                                outputs=['y'])

    def infer_shape(self, x_shape, beta_shape, gamma_shape, batch_std_shape, running_std_shape, batch_mean_shape):
        validator.check("batch_std shape", batch_std_shape, "running_std shape", running_std_shape, Rel.EQ, self.name)
        validator.check("batch_std shape", batch_std_shape, "batch_mean shape", batch_mean_shape, Rel.EQ, self.name)
        validator.check("batch_std shape", batch_std_shape, "beta shape", beta_shape, Rel.EQ, self.name)
        validator.check("batch_std shape", batch_std_shape, "batch_mean shape", gamma_shape, Rel.EQ, self.name)
        validator.check("batch_std_shape[0]", batch_std_shape[0], "x_shape channel size", x_shape[self.channel_axis],
                        Rel.EQ, self.name)
        return x_shape

    def infer_dtype(self, x_type, beta_type, gamma_type, batch_std_type, running_std_type, batch_mean_type):
        args = {"batch_std": batch_std_type, "running_std": running_std_type, "batch_mean": batch_mean_type,
                "beta": beta_type, "gamma": gamma_type, "x": x_type}
        validator.check_tensor_type_same(args, (mstype.float16, mstype.float32), self.name)
        return x_type


class BatchNormFold2GradD(PrimitiveWithInfer):
    """Performs grad of CorrectionAddGrad operation."""
    channel_axis = 1

    @prim_attr_register
    def __init__(self, freeze_bn=False):
        """init MulFold layer"""
        from mindspore.ops._op_impl._custom_op import batchnorm_fold2_grad
        self.freeze_bn = freeze_bn
        self.init_prim_io_names(
            inputs=['dout', 'dout_reduce', 'dout_x_reduce', 'gamma', 'batch_std', 'batch_mean', 'running_std'],
            outputs=['d_batch_std', 'd_batch_mean', 'd_gamma', 'dx'])

    def infer_shape(self, dout_shape, dout_reduce_shape, dout_x_reduce_shape, gamma_shape, batch_std_shape,
                    batch_mean_shape, running_std_shape):
        validator.check("batch_std shape", batch_std_shape, "batch_mean shape", batch_mean_shape, Rel.EQ, self.name)
        validator.check("batch_std shape", batch_std_shape, "running_std shape", running_std_shape, Rel.EQ, self.name)
        validator.check("batch_std shape", batch_std_shape, "gamma shape", gamma_shape, Rel.EQ, self.name)
        validator.check("batch_std size", batch_std_shape[0], "dout channel size", dout_shape[self.channel_axis],
                        Rel.EQ, self.name)
        return gamma_shape, gamma_shape, gamma_shape, dout_shape

    def infer_dtype(self, dout_type, dout_reduce_type, dout_x_reduce_type, gamma_type, batch_std_type,
                    batch_mean_type, running_std_type):
        validator.check("batch_std type", batch_std_type,
                        "batch_mean type", batch_mean_type)
        validator.check("batch_std type", batch_std_type,
                        "gamma type", gamma_type)
        validator.check("batch_std type", batch_std_type,
                        "running_std type", running_std_type)
        validator.check("batch_std_type", batch_std_type,
                        "dout type", dout_type)
        args = {"batch_std": batch_std_type, "batch_mean": batch_mean_type, "gamma": gamma_type,
                "running_std": running_std_type, "dout": dout_type}
        validator.check_tensor_type_same(args, (mstype.float16, mstype.float32), self.name)
        return gamma_type, gamma_type, gamma_type, gamma_type


class BatchNormFold2GradReduce(PrimitiveWithInfer):
    """Performs grad of CorrectionAddGrad operation."""
    channel_axis = 1

    @prim_attr_register
    def __init__(self, freeze_bn=False):
        """init MulFold layer"""
        from mindspore.ops._op_impl._custom_op import batchnorm_fold2_grad_reduce
        self.freeze_bn = freeze_bn
        self.init_prim_io_names(inputs=['dout', 'x'],
                                outputs=['dout_reduce', 'dout_x_reduce'])

    def infer_shape(self, dout_shape, x_shape):
        validator.check("dout shape", dout_shape, "x shape", x_shape, Rel.EQ, self.name)
        return (dout_shape[self.channel_axis],), (dout_shape[self.channel_axis],)

    def infer_dtype(self, dout_type, x_type):
        validator.check("dout type", dout_type, "x type", x_type)
        return dout_type, dout_type


class FakeQuantWithMinMaxUpdate(PrimitiveWithInfer):
    r"""
    Simulate the quantize and dequantize operations in training time.

    Args:
        num_bits (int) : Number bits for aware quantilization. Default: 8.
        ema (bool): Use EMA algorithm update value min and max. Default: False.
        ema_decay (int) : EMA algorithm decay parameter. Default: 0.999.
        quant_delay (int): Quantilization delay parameter. Before delay step in training time not update
            simulate aware quantize funcion. After delay step in training time begin simulate the aware
            quantize funcion. Default: 0.
        symmetric (bool): Quantization algorithm use symmetric or not. Default: False.
        narrow_range (bool): Quantization algorithm use narrow range or not. Default: False.
        training (bool): Training the network or not. Default: True.

    Inputs:
        - **x** (Tensor) : float32 Tensor representing the shape of the output tensor.
        - **min** (Tensor) : Value of the min range of the input data x.
        - **max** (Tensor) : Value of the max range of the input data x.

    Outputs:
        - Tensor: Simulate quantize tensor of x.

    Examples:
        >>> input_tensor = Tensor(np.random.rand(3, 16, 5, 5), mstype.float32)
        >>> min_tensor = Tensor(np.array([-6]), mstype.float32)
        >>> max_tensor = Tensor(np.array([6]), mstype.float32)
        >>> output_tensor = P.FakeQuantWithMinMax(num_bits=8)(input_tensor, min_tensor, max_tensor)
    """
    support_quant_bit = [4, 7, 8]

    @prim_attr_register
    def __init__(self, num_bits=8, ema=False, ema_decay=0.999, quant_delay=0, symmetric=False, narrow_range=False,
                 training=True):
        """init FakeQuantWithMinMax OP"""
        from mindspore.ops._op_impl._custom_op import correction_mul, correction_mul_grad
        from mindspore.ops._op_impl._custom_op import fake_quant_with_min_max, fake_quant_with_min_max_grad
        from mindspore.ops._op_impl._custom_op import fake_quant_with_min_max_update
        if num_bits not in self.support_quant_bit:
            raise ValueError(f"For '{self.name}' attr \'num_bits\' is not support.")
        if ema and not ema_decay:
            raise ValueError(f"For '{self.name}' attr \'ema\' and \'ema_decay\' should set together.")

        self.ema = validator.check_value_type('ema', ema, (bool,), self.name)
        self.symmetric = validator.check_value_type('symmetric', symmetric, (bool,), self.name)
        self.narrow_range = validator.check_value_type('narrow_range', narrow_range, (bool,), self.name)
        self.training = validator.check_value_type('training', training, (bool,), self.name)
        self.ema_decay = validator.check_number_range('ema_decay', ema_decay, 0, 1, Rel.INC_BOTH, self.name)
        self.num_bits = validator.check_integer('num_bits', num_bits, 0, Rel.GT, self.name)
        self.quant_delay = validator.check_value_type('quant_delay', quant_delay, (int,), self.name)
        self.init_prim_io_names(inputs=['x', 'min', 'max'],
                                outputs=['min_up', 'max_up'])

    def infer_shape(self, x_shape, min_shape, max_shape):
        validator.check_integer("x rank", len(x_shape), 1, Rel.GT, self.name)
        validator.check("min shape", min_shape, "max shape", max_shape, Rel.EQ, self.name)
        validator.check_integer("min rank", len(min_shape), 1, Rel.EQ, self.name)
        return min_shape, max_shape

    def infer_dtype(self, x_type, min_type, max_type):
        valid_types = (mstype.float16, mstype.float32)
        validator.check_tensor_type_same({"x": x_type}, valid_types, self.name)
        validator.check_tensor_type_same({"min": min_type}, valid_types, self.name)
        validator.check_tensor_type_same({"max": max_type}, valid_types, self.name)
        return min_type, max_type
