# Copyright 2022 Huawei Technologies Co., Ltd
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

import math
import numpy as np
import pytest

from mindspore import ops, nn, context, set_seed
from mindspore.train import DatasetHelper, connect_network_with_dataset
from mindspore.common.tensor import Tensor
from mindspore.ops import functional as F
import mindspore.dataset as ds
import mindspore.common.dtype as mstype

context.set_context(mode=context.GRAPH_MODE)
context.set_context(device_target="Ascend")
set_seed(2)


def _dynamic_exec_preprocess(net, is_training, datasets, data_sink_mode, epoch_num, sink_size):
    if data_sink_mode and not is_training:
        datasets.__loop_size__ = 1

    dataset_helper = DatasetHelper(datasets, data_sink_mode, sink_size, epoch_num)

    if data_sink_mode:
        net = connect_network_with_dataset(net, dataset_helper)

    return dataset_helper, net


def dynamic_sink_process(net, dataset, is_training=True):
    data_sink_mode = True
    sink_size = 1
    epoch_num = 1
    data_helper, net = _dynamic_exec_preprocess(net, is_training, dataset, data_sink_mode, epoch_num, sink_size)
    net.set_train(is_training)
    for inputs in data_helper:
        outputs = net(*inputs)
        return outputs


def static_process(net, datasets, is_training=True):
    net.set_train(is_training)
    for inputs in datasets.create_tuple_iterator():
        outputs = net(*inputs)
        return outputs


def compare_acc(outputs, expects):
    if isinstance(outputs, (tuple, list)):
        assert isinstance(expects, (tuple, list))
        for outputs_, expects_ in zip(outputs, expects):
            if not compare_acc(outputs_, expects_):
                return False
    else:
        if not np.allclose(outputs.asnumpy(), expects.asnumpy(), rtol=1.0e-4, atol=1.0e-4):
            return False
    return True


class GradNetWrtX(nn.Cell):
    def __init__(self, network):
        super(GradNetWrtX, self).__init__()
        self.grad = ops.GradOperation(get_all=True, sens_param=True)
        self.network = network

    def construct(self, input_, output_grad):
        return self.grad(self.network)(input_, output_grad)



class GradNetWrtX2inputs(nn.Cell):
    def __init__(self, network):
        super(GradNetWrtX2inputs, self).__init__()
        self.grad = ops.GradOperation(get_all=True, sens_param=True)
        self.network = network

    def construct(self, input1, input2, output_grad):
        return self.grad(self.network)(input1, input2, output_grad)


def comm_func(dyn_range, input_shp, data_type, op_net, num=None):
    list_data = []
    for i in dyn_range:
        tmp_data = []
        for data_shp in input_shp:
            if num is None:
                cur_shp = [dim if dim is not None else i for dim in data_shp]
            else:
                cur_shp = []
                k = 0
                for dim in data_shp:
                    if dim is not None:
                        cur_shp.append(dim)
                    elif k == 1:
                        cur_shp.append(num)
                    else:
                        cur_shp.append(i)
                    k = k + 1
            tmp_data.append(np.random.random(cur_shp).astype(data_type))
        list_data.append(tuple(tmp_data))

    data_map = {}
    dyn_tensors = []

    def np_type_to_ms(data_type):
        if data_type == np.float32:
            return mstype.float32
        if data_type == np.float64:
            return mstype.float64
        if data_type == np.int32:
            return mstype.int32
        if data_type == np.int64:
            return mstype.int64
        raise ValueError("Unsupportted datatype: {}".format(data_type))

    for i, val in enumerate(input_shp):
        data_map["data" + str(i + 1)] = val
        dyn_tensors.append(Tensor(dtype=np_type_to_ms(data_type), shape=val))

    dataset = ds.GeneratorDataset(list_data, list(data_map.keys()))
    op_net.set_inputs(*dyn_tensors)
    gradient = dynamic_sink_process(op_net, dataset)
    gradient_cmp = static_process(op_net, dataset)
    assert compare_acc(gradient, gradient_cmp)


class CustomDense(nn.Dense):
    def __init__(self,
                 in_channels,
                 out_channels,
                 weight_init='normal',
                 bias_init='zeros',
                 has_bias=True,
                 activation=None):
        """Initialize CustomDense."""
        super(CustomDense, self).__init__(in_channels,
                                          out_channels,
                                          weight_init,
                                          bias_init,
                                          has_bias,
                                          activation)
        self.scatterupdate = ops.TensorScatterUpdate()
        self.dyn_shape = ops.TensorShape()
        self.mul = ops.Mul()
        self.cast = ops.Cast()
        self.indices_0 = Tensor(np.array([[0]]), mstype.int32)
        self.indices_1 = Tensor(np.array([[-1]]), mstype.int32)
        self.indices_2 = Tensor(np.array([[2]]), mstype.int32)

    def construct(self, x):
        if F.is_sequence_value_unknown(x.shape):
            x_dyn_shape = self.dyn_shape(x)
            x_dyn_shape = self.cast(x_dyn_shape, mstype.float16)
            if len(x_dyn_shape) != 2:
                new_shape = x_dyn_shape[1:]
                updates = self.mul(x_dyn_shape[0:1], x_dyn_shape[1:2])
                new_shape = self.scatterupdate(
                    new_shape, self.indices_0, updates)
                new_shape = self.cast(new_shape, mstype.int64)
                x = self.reshape(x, new_shape)
            x = self.matmul(x, self.weight)
            if self.has_bias:
                if self.bias.dtype != mstype.float16:
                    ori_dtype = x.dtype
                    x = self.bias_add(self.cast(x, mstype.float16), self.cast(self.bias, mstype.float16))
                    x = self.cast(x, ori_dtype)
                else:
                    x = self.bias_add(x, self.bias)
            if self.activation_flag:
                x = self.activation(x)
            if len(x_dyn_shape) != 2:
                out_shape = self.dyn_shape(x)
                out_shape = self.cast(out_shape, mstype.float16)
                updates = out_shape[1:2]
                x_dyn_shape = self.scatterupdate(
                    x_dyn_shape, self.indices_2, updates)
                x_dyn_shape = self.cast(x_dyn_shape, mstype.int64)
                x = self.reshape(x, x_dyn_shape)
        else:
            x_shape = self.shape_op(x)
            if len(x_shape) != 2:
                x = self.reshape(x, (-1, x_shape[-1]))
            x = self.matmul(x, self.weight)
            if self.has_bias:
                x = self.bias_add(x, self.bias)
            if self.activation_flag:
                x = self.activation(x)
            if len(x_shape) != 2:
                out_shape = x_shape[:-1] + (-1,)
                x = self.reshape(x, out_shape)
        return x


class BatchNorm1d(nn.Cell):
    def __init__(self, channels):
        super(BatchNorm1d, self).__init__()
        self.expand_dims = ops.ExpandDims()
        self.shape = ops.TensorShape()
        self.reshape = ops.Reshape()
        self.batchnorm = nn.BatchNorm2d(channels)

    def construct(self, x):
        x_shape = self.shape(x)
        x = self.expand_dims(x, 2)
        x = self.expand_dims(x, 3)
        out = self.batchnorm(x)
        out = self.reshape(out, x_shape)
        return out


class BatchNorm1dSingleOp(nn.Cell):
    def __init__(self, channels):
        super(BatchNorm1dSingleOp, self).__init__()
        self.batchnorm = nn.BatchNorm1d(channels)

    def construct(self, x):
        out = self.batchnorm(x)
        return out


class Positional(nn.Cell):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.xscale = Tensor([math.sqrt(self.d_model)], dtype=mstype.float32)
        self.max_len = max_len

        self.pe = np.zeros((self.max_len, self.d_model))
        position = np.expand_dims(np.arange(0, self.max_len, dtype=np.float32), 1)
        div_term = np.exp(
            np.arange(0, self.d_model, 2, dtype=np.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        self.pe[:, 0::2] = np.sin(position * div_term)
        self.pe[:, 1::2] = np.cos(position * div_term)
        self.pe = Tensor(np.expand_dims(self.pe, 0), mstype.float32)
        self.dyn_shape = ops.TensorShape()
        self.stridedslice = ops.StridedSlice()
        self.indices_1 = Tensor(([[1]]), mstype.int32)
        self.scatterupdate = ops.TensorScatterUpdate()
        self.end = Tensor((self.pe.shape[0], 0, self.pe.shape[2]), mstype.float32)

    def construct(self, x: Tensor, offset: int = 0):
        if not F.is_sequence_value_unknown(x.shape):
            pos_emb = self.pe[:, offset: offset + x.shape[1]]
        else:
            x_dyn_shape = self.dyn_shape(x)
            x_dyn_shape = self.cast(x_dyn_shape, mstype.float32)
            begin = (0, offset, 0)
            end = self.end
            end = self.scatterupdate(end, self.indices_1, offset + x_dyn_shape[1:2])
            end = self.cast(end, mstype.int64)
            pos_emb = self.stridedslice(self.pe, begin, end, (1, 1, 1))
        x = x * self.xscale + pos_emb
        return x, pos_emb


class Sort(nn.Cell):
    def __init__(self):
        super(Sort, self).__init__()
        self.sort = ops.Sort(axis=0)

    def construct(self, x):
        out = self.sort(x)
        return out


class NMSWithMask(nn.Cell):
    def __init__(self, iou_threshold):
        super(NMSWithMask, self).__init__()
        self.nmswithmask = ops.NMSWithMask(iou_threshold)

    def construct(self, x):
        box, _, mask = self.nmswithmask(x)
        return box, mask


class Concat(nn.Cell):
    def __init__(self):
        super(Concat, self).__init__()
        self.concat = ops.Concat(axis=0)

    def construct(self, x):
        x = 65 * [x]
        out = self.concat(x)
        return out


class Stack(nn.Cell):
    def __init__(self):
        super(Stack, self).__init__()
        self.stack = ops.Stack(axis=0)

    def construct(self, x):
        x = 65 * [x]
        out = self.stack(x)
        return out


class MaxPool(nn.Cell):
    def __init__(self):
        super(MaxPool, self).__init__()
        self.maxpool = ops.MaxPool(pad_mode="VALID", kernel_size=2, strides=1)

    def construct(self, x):
        out = self.maxpool(x)
        return out


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_custom_dense():
    """
    Feature: Test Dynamic Dense and its backward. The input shape is dynamic.
    Description: The input shape is dynamic.
    Expectation: Assert that results are consistent with fixed shape.
    """
    batch_size = 16
    dynamic_range = range(2, 64)
    data_type = np.float32
    input_shape = [(batch_size, None, 64), (batch_size, None, 64)]
    net = GradNetWrtX(CustomDense(64, 64))
    comm_func(dynamic_range, input_shape, data_type, net)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_batchnorm1d():
    """
    Feature: Test Dynamic batchnorm1d and its backward. The input shape is dynamic.
    Description: The input shape is dynamic.
    Expectation: Assert that results are consistent with fixed shape.
    """
    dynamic_range = range(2, 64)
    data_type = np.float32
    input_shape = [(None, 64)]
    net = BatchNorm1d(64)
    comm_func(dynamic_range, input_shape, data_type, net)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_batchnorm1d_single_op():
    """
    Feature: Test Dynamic batchnorm1d and its backward. The input shape is dynamic.
    Description: The input shape is dynamic.
    Expectation: Assert that results are consistent with fixed shape.
    """
    dynamic_range = range(2, 64)
    data_type = np.float32
    input_shape = [(None, 64)]
    net = BatchNorm1dSingleOp(64)
    comm_func(dynamic_range, input_shape, data_type, net)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_batchnorm1d_single_op_2_unknown_shape():
    """
    Feature: Test Dynamic batchnorm1d and its backward. The input shape is dynamic.
    Description: The input shape is dynamic.
    Expectation: Assert that results are consistent with fixed shape.
    """
    dynamic_range = range(2, 64)
    data_type = np.float32
    input_shape = [(None, None)]
    net = BatchNorm1dSingleOp(64)
    comm_func(dynamic_range, input_shape, data_type, net, 64)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_positional():
    """
    Feature: Test Dynamic Positional and its backward. The input shape is dynamic.
    Description: The input shape is dynamic.
    Expectation: Assert that results are consistent with fixed shape.
    """
    dynamic_range = range(2, 64)
    data_type = np.float32
    input_shape = [(32, None, 256)]
    net = Positional(256)
    comm_func(dynamic_range, input_shape, data_type, net)


def test_dynamic_sort():
    """
    Feature: Test Dynamic sort and its backward. The input shape is dynamic.
    Description: The input shape is dynamic.
    Expectation: Assert that results are consistent with fixed shape.
    """
    dynamic_range = range(2, 64)
    data_type = np.float32
    input_shape = [(None, 64)]
    net = Sort()
    comm_func(dynamic_range, input_shape, data_type, net)


def test_dynamic_sort2():
    """
    Feature: Test Dynamic sort and its backward. The input shape is dynamic.
    Description: The input shape is dynamic.
    Expectation: Assert that results are consistent with fixed shape.
    """
    dynamic_range = range(2, 64)
    data_type = np.float32
    input_shape = [(None, None)]
    net = Sort()
    comm_func(dynamic_range, input_shape, data_type, net)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_nms_with_mask():
    """
    Feature: Test Dynamic NMSWithMask and its backward. The input shape is dynamic.
    Description: The input shape is dynamic.
    Expectation: Assert that results are consistent with fixed shape.
    """
    dynamic_range = range(2, 64)
    data_type = np.float32
    input_shape = [(None, 5)]
    net = NMSWithMask(0.3)
    comm_func(dynamic_range, input_shape, data_type, net)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_concat():
    """
    Feature: Test Dynamic Concat and its backward. The input shape is dynamic.
    Description: The input shape is dynamic.
    Expectation: Assert that results are consistent with fixed shape.
    """
    dynamic_range = range(2, 64)
    data_type = np.float32
    input_shape = [(None, 5)]
    net = Concat()
    comm_func(dynamic_range, input_shape, data_type, net)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_stack():
    """
    Feature: Test Dynamic Stack and its backward. The input shape is dynamic.
    Description: The input shape is dynamic.
    Expectation: Assert that results are consistent with fixed shape.
    """
    dynamic_range = range(2, 64)
    data_type = np.float32
    input_shape = [(None, 5)]
    net = Stack()
    comm_func(dynamic_range, input_shape, data_type, net)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_maxpool1():
    """
    Feature: Test Dynamic maxpool and its backward. The input shape is dynamic.
    Description: The input shape is dynamic.
    Expectation: Assert that results are consistent with fixed shape.
    """
    dynamic_range = range(2, 64)
    data_type = np.float32
    input_shape = [(32, 16, 32, None)]
    net = MaxPool()
    comm_func(dynamic_range, input_shape, data_type, net)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_maxpool2():
    """
    Feature: Test Dynamic maxpool and its backward. The input shape is dynamic.
    Description: The input shape is dynamic.
    Expectation: Assert that results are consistent with fixed shape.
    """
    dynamic_range = range(2, 64)
    data_type = np.float32
    input_shape = [(32, 16, None, 8)]
    net = MaxPool()
    comm_func(dynamic_range, input_shape, data_type, net)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_maxpool3():
    """
    Feature: Test Dynamic maxpool and its backward. The input shape is dynamic.
    Description: The input shape is dynamic.
    Expectation: Assert that results are consistent with fixed shape.
    """
    dynamic_range = range(2, 64)
    data_type = np.float32
    input_shape = [(32, None, 32, 8)]
    net = MaxPool()
    comm_func(dynamic_range, input_shape, data_type, net)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_maxpool4():
    """
    Feature: Test Dynamic maxpool and its backward. The input shape is dynamic.
    Description: The input shape is dynamic.
    Expectation: Assert that results are consistent with fixed shape.
    """
    dynamic_range = range(2, 64)
    data_type = np.float32
    input_shape = [(None, 16, 32, 8)]
    net = MaxPool()
    comm_func(dynamic_range, input_shape, data_type, net)
