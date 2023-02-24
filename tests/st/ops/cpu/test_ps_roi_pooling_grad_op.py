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
"""
Test PSROIPoolingGrad.
"""
import numpy as np
import pytest

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as P
from mindspore.ops.operations import _grad_ops as G

DEVICE_TARGET = "CPU"
CTX_MODE = ms.context.GRAPH_MODE
ALL_CLOSE_CRITERION = 1e-4


class NetPSROIPoolingGrad(nn.Cell):
    """Simple net for PSROIPoolingGrad."""

    def __init__(self, input_size, spatial_scale, group_size, output_dim, dynamic_shape=False):
        super(NetPSROIPoolingGrad, self).__init__()
        self.dynamic_shape = dynamic_shape
        self.unique_op = P.Unique()
        self.reshape_op = P.Reshape()
        self.ps_roi_pooling_grad = G.PSROIPoolingGrad(input_size, spatial_scale,
                                                      group_size, output_dim)

    def construct(self, dy, rois):
        if self.dynamic_shape:
            rois = self.reshape_op(rois, (-1,))
            rois, _ = self.unique_op(rois)
            rois = self.reshape_op(rois, (2, 5, -1))
        return self.ps_roi_pooling_grad(dy, rois)


def _ps_roi_pooling_grad_case(data_type, mode, y_size_adjust=None, dynamic_shape=False):
    """Run op calculation."""
    size_scale = 10
    rois_np = np.array(
        [[[0.0000], [150.3563 / size_scale],
          [200.1320 / size_scale], [579.3563 / size_scale],
          [602.3452 / size_scale]],
         [[1.0000], [65.1263 / size_scale],
          [30.8564 / size_scale], [762.4214 / size_scale],
          [567.9854 / size_scale]]]).astype(data_type)
    batch_size = rois_np.shape[0]
    rois_number = rois_np.shape[2]
    rois_ms = ms.Tensor(rois_np)

    x_height = 5
    x_width = 4
    group_size = 2
    output_dim = 2

    y_size = [batch_size * rois_number, output_dim, group_size, group_size]
    if y_size_adjust is not None:
        y_size[y_size_adjust] += 1
    dy_np = np.ones(y_size).astype(data_type)
    dy_ms = ms.Tensor(dy_np)

    input_size = (x_height, x_width)
    spatial_scale = 1.0 / 16

    ms.context.set_context(mode=mode,
                           device_target=DEVICE_TARGET)
    ps_roi_pooling_grad = NetPSROIPoolingGrad(
        input_size, spatial_scale, group_size, output_dim,
        dynamic_shape=dynamic_shape)

    runtime_error_occurred = False
    try:
        output = ps_roi_pooling_grad(dy_ms, rois_ms)
    except RuntimeError:
        runtime_error_occurred = True
        if y_size_adjust is None:
            raise
    if y_size_adjust is not None:
        assert runtime_error_occurred, "Expected RunttimeError to occur, but it does not occur."
        return
    output_ms = output.asnumpy()

    output_gt = np.array(
        [[[[0., 0., 0., 0.], [0., 0.25, 0.25, 0.],
           [0., 0.25, 0.25, 0.], [0., 0., 0., 0.],
           [0., 0., 0., 0.]],

          [[0., 0., 0., 0.], [0., 0., 0.25, 0.25],
           [0., 0., 0.25, 0.25], [0., 0., 0., 0.],
           [0., 0., 0., 0.]],

          [[0., 0., 0., 0.], [0., 0., 0., 0.],
           [0., 0.25, 0.25, 0.], [0., 0.25, 0.25, 0.],
           [0., 0., 0., 0.]],

          [[0., 0., 0., 0.], [0., 0., 0., 0.],
           [0., 0., 0.25, 0.25], [0., 0., 0.25, 0.25],
           [0., 0., 0., 0.]],

          [[0., 0., 0., 0.], [0., 0.25, 0.25, 0.],
           [0., 0.25, 0.25, 0.], [0., 0., 0., 0.],
           [0., 0., 0., 0.]],

          [[0., 0., 0., 0.], [0., 0., 0.25, 0.25],
           [0., 0., 0.25, 0.25], [0., 0., 0., 0.],
           [0., 0., 0., 0.]],

          [[0., 0., 0., 0.], [0., 0., 0., 0.],
           [0., 0.25, 0.25, 0.], [0., 0.25, 0.25, 0.],
           [0., 0., 0., 0.]],

          [[0., 0., 0., 0.], [0., 0., 0., 0.],
           [0., 0., 0.25, 0.25], [0., 0., 0.25, 0.25],
           [0., 0., 0., 0.]]],

         [[[0.16666667, 0.16666667, 0.16666667, 0.],
           [0.16666667, 0.16666667, 0.16666667, 0.],
           [0., 0., 0., 0.], [0., 0., 0., 0.],
           [0., 0., 0., 0.]],

          [[0., 0., 0.25, 0.25], [0., 0., 0.25, 0.25],
           [0., 0., 0., 0.], [0., 0., 0., 0.],
           [0., 0., 0., 0.]],

          [[0., 0., 0., 0.], [0., 0., 0., 0.],
           [0.16666667, 0.16666667, 0.16666667, 0.],
           [0.16666667, 0.16666667, 0.16666667, 0.],
           [0., 0., 0., 0.]],

          [[0., 0., 0., 0.], [0., 0., 0., 0.],
           [0., 0., 0.25, 0.25], [0., 0., 0.25, 0.25],
           [0., 0., 0., 0.]],

          [[0.16666667, 0.16666667, 0.16666667, 0.],
           [0.16666667, 0.16666667, 0.16666667, 0.],
           [0., 0., 0., 0.], [0., 0., 0., 0.],
           [0., 0., 0., 0.]],

          [[0., 0., 0.25, 0.25], [0., 0., 0.25, 0.25],
           [0., 0., 0., 0.], [0., 0., 0., 0.],
           [0., 0., 0., 0.]],

          [[0., 0., 0., 0.], [0., 0., 0., 0.],
           [0.16666667, 0.16666667, 0.16666667, 0.],
           [0.16666667, 0.16666667, 0.16666667, 0.],
           [0., 0., 0., 0.]],

          [[0., 0., 0., 0.], [0., 0., 0., 0.],
           [0., 0., 0.25, 0.25], [0., 0., 0.25, 0.25],
           [0., 0., 0., 0.]]]], dtype=data_type)
    assert np.allclose(
        output_ms, output_gt,
        atol=ALL_CLOSE_CRITERION, rtol=ALL_CLOSE_CRITERION)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_ps_roi_pooling_grad_mind_ir():
    """
    Feature: PSROIPoolingGrad op.
    Description: Test the MindIR behavior of PSROIPooingGrad op.
    Expectation: success.
    """
    ms.context.set_context(mode=CTX_MODE, device_target=DEVICE_TARGET)
    data_type = np.float32
    size_scale = 10
    rois_np = np.array(
        [[[0.0000], [150.3563 / size_scale],
          [200.1320 / size_scale], [579.3563 / size_scale],
          [602.3452 / size_scale]],
         [[1.0000], [65.1263 / size_scale],
          [30.8564 / size_scale], [762.4214 / size_scale],
          [567.9854 / size_scale]]]).astype(data_type)
    batch_size = rois_np.shape[0]
    rois_number = rois_np.shape[2]
    rois_ms = ms.Tensor(rois_np)

    x_height = 5
    x_width = 4
    group_size = 2
    output_dim = 2

    y_size = [batch_size * rois_number, output_dim, group_size, group_size]
    dy_np = np.ones(y_size).astype(data_type)
    dy_ms = ms.Tensor(dy_np)

    input_size = (x_height, x_width)
    spatial_scale = 1.0 / 16
    net = NetPSROIPoolingGrad(input_size, spatial_scale, group_size, output_dim)
    old_out = net(dy_ms, rois_ms)
    ms.export(
        net, dy_ms, rois_ms,
        file_name="ps_roi_pooling_grad",
        file_format='MINDIR')

    graph = ms.load("ps_roi_pooling_grad.mindir")
    new_net = nn.GraphCell(graph)
    new_out = new_net(dy_ms, rois_ms)
    assert np.allclose(
        old_out.asnumpy(), new_out.asnumpy(),
        atol=ALL_CLOSE_CRITERION, rtol=ALL_CLOSE_CRITERION)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_ps_roi_pooling_grad_graph_mode():
    """
    Feature: PSROIPoolingGrad op.
    Description: Test the normal behavior of PSROIPooingGrad op.
    Expectation: success.
    """
    _ps_roi_pooling_grad_case(
        data_type=np.float32,
        mode=CTX_MODE)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_ps_roi_pooling_grad_y_0_shape_wrong():
    """
    Feature: PSROIPoolingGrad op.
    Description: Test the validation of shape.
    Expectation: success.
    """
    _ps_roi_pooling_grad_case(
        data_type=np.float32,
        mode=CTX_MODE,
        y_size_adjust=0
    )


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_ps_roi_pooling_grad_y_1_shape_wrong():
    """
    Feature: PSROIPoolingGrad op.
    Description: Test the validation of shape.
    Expectation: success.
    """
    _ps_roi_pooling_grad_case(
        data_type=np.float32,
        mode=CTX_MODE,
        y_size_adjust=0
    )


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_ps_roi_pooling_grad_y_2_shape_wrong():
    """
    Feature: PSROIPoolingGrad op.
    Description: Test the validation of shape.
    Expectation: success.
    """
    _ps_roi_pooling_grad_case(
        data_type=np.float32,
        mode=CTX_MODE,
        y_size_adjust=0
    )


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_ps_roi_pooling_grad_y_3_shape_wrong():
    """
    Feature: PSROIPoolingGrad op.
    Description: Test the validation of shape.
    Expectation: success.
    """
    _ps_roi_pooling_grad_case(
        data_type=np.float32,
        mode=CTX_MODE,
        y_size_adjust=0
    )


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_ps_roi_pooling_grad_pynative_mode():
    """
    Feature: PSROIPoolingGrad op.
    Description: Test the normal behavior of PSROIPooingGrad op.
    Expectation: success.
    """
    _ps_roi_pooling_grad_case(
        data_type=np.float32,
        mode=ms.context.PYNATIVE_MODE)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_ps_roi_pooling_grad_input_size_attr_type_wrong():
    """
    Feature: PSROIPoolingGrad op.
    Description: Test the validation of attrs.
    Expectation: success.
    """

    arg_name = "input_size"
    arg_value = 1.1

    _check_attr_validation(arg_name, arg_value)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_ps_roi_pooling_grad_input_size_attr_type_wrong2():
    """
    Feature: PSROIPoolingGrad op.
    Description: Test the validation of attrs.
    Expectation: success.
    """

    arg_name = "input_size"
    arg_value = (1, 1.1)

    _check_attr_validation(arg_name, arg_value)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_ps_roi_pooling_grad_input_size_attr_type_wrong3():
    """
    Feature: PSROIPoolingGrad op.
    Description: Test the validation of attrs.
    Expectation: success.
    """

    arg_name = "input_size"
    arg_value = [2, 2]

    _check_attr_validation(arg_name, arg_value)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_ps_roi_pooling_grad_input_size_attr_range_wrong():
    """
    Feature: PSROIPoolingGrad op.
    Description: Test the validation of attrs.
    Expectation: success.
    """
    _check_attr_validation(arg_name="input_size", arg_value=-1)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_ps_roi_pooling_grad_input_size_attr_range_wrong2():
    """
    Feature: PSROIPoolingGrad op.
    Description: Test the validation of attrs.
    Expectation: success.
    """
    _check_attr_validation(arg_name="input_size", arg_value=0)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_ps_roi_pooling_grad_input_size_attr_range_wrong3():
    """
    Feature: PSROIPoolingGrad op.
    Description: Test the validation of attrs.
    Expectation: success.
    """
    _check_attr_validation(arg_name="input_size", arg_value=(100, -1))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_ps_roi_pooling_grad_input_size_attr_range_wrong4():
    """
    Feature: PSROIPoolingGrad op.
    Description: Test the validation of attrs.
    Expectation: success.
    """
    _check_attr_validation(arg_name="input_size", arg_value=(0, 100))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_ps_roi_pooling_grad_spatial_scale_attr_type_wrong():
    """
    Feature: PSROIPoolingGrad op.
    Description: Test the validation of attrs.
    Expectation: success.
    """
    _check_attr_validation(arg_name="spatial_scale", arg_value=object())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_ps_roi_pooling_grad_spatial_scale_attr_range_wrong():
    """
    Feature: PSROIPoolingGrad op.
    Description: Test the validation of attrs.
    Expectation: success.
    """
    _check_attr_validation(arg_name="spatial_scale", arg_value=-0.15)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_ps_roi_pooling_grad_group_size_attr_type_wrong():
    """
    Feature: PSROIPoolingGrad op.
    Description: Test the validation of attrs.
    Expectation: success.
    """
    _check_attr_validation(arg_name="group_size", arg_value=7.1)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_ps_roi_pooling_grad_group_size_attr_range_wrong():
    """
    Feature: PSROIPoolingGrad op.
    Description: Test the validation of attrs.
    Expectation: success.
    """
    _check_attr_validation(arg_name="group_size", arg_value=-1)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_ps_roi_pooling_output_dim_attr_type_wrong():
    """
    Feature: PSROIPoolingGrad op.
    Description: Test the validation of attrs.
    Expectation: success.
    """
    _check_attr_validation(arg_name="output_dim", arg_value=7.1)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_ps_roi_pooling_output_dim_attr_range_wrong():
    """
    Feature: PSROIPoolingGrad op.
    Description: Test the validation of attrs.
    Expectation: success.
    """
    _check_attr_validation(arg_name="output_dim", arg_value=-1)


def _check_attr_validation(arg_name, arg_value):
    ms.context.set_context(mode=CTX_MODE,
                           device_target=DEVICE_TARGET)
    op_kwargs = {
        "input_size": 100,
        "spatial_scale": 1 / 16,
        "group_size": 7,
        "output_dim": 3
    }
    op_kwargs[arg_name] = arg_value
    try:
        G.PSROIPoolingGrad(**op_kwargs)
    except TypeError:
        assert True
    except ValueError:
        assert True
    else:
        assert False


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_ps_roi_pooling_grad_input_args_num():
    """
    Feature: PSROIPoolingGrad op.
    Description: Test the validation of input args num.
    Expectation: success.
    """
    ms.context.set_context(mode=CTX_MODE,
                           device_target=DEVICE_TARGET)
    op = G.PSROIPoolingGrad(
        input_size=100,
        spatial_scale=1/16,
        group_size=7,
        output_dim=3
    )
    try:
        op(
            ms.Tensor([1.0], dtype=ms.float32),
            ms.Tensor([1.0], dtype=ms.float32),
            ms.Tensor([1.0], dtype=ms.float32)
        )
    except ValueError:
        return
    else:
        assert False, "Expected ValueError to occur, but it does not occur."


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_ps_roi_pooling_grad_input_type_unsupported():
    """
    Feature: PSROIPoolingGrad op.
    Description: Test the validation of input type.
    Expectation: success.
    """
    ms.context.set_context(mode=CTX_MODE,
                           device_target=DEVICE_TARGET)
    op = G.PSROIPoolingGrad(
        input_size=100,
        spatial_scale=1/16,
        group_size=7,
        output_dim=3
    )

    try:
        op(
            ms.Tensor([1.0], dtype=ms.float64),
            ms.Tensor([1.0], dtype=ms.float64)
        )
    except TypeError:
        return
    else:
        assert False, "Expected TypeError to occur, but it does not occur."


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_ps_roi_pooling_grad_input_type_unsupported2():
    """
    Feature: PSROIPoolingGrad op.
    Description: Test the validation of input type.
    Expectation: success.
    """
    ms.context.set_context(mode=CTX_MODE,
                           device_target=DEVICE_TARGET)
    op = G.PSROIPoolingGrad(
        input_size=100,
        spatial_scale=1/16,
        group_size=7,
        output_dim=3
    )

    try:
        op(
            1.0,
            ms.Tensor([1.0], dtype=ms.float32)
        )
    except TypeError:
        return
    else:
        assert False, "Expected TypeError to occur, but it does not occur."


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_ps_roi_pooling_grad_input_type_unsupported3():
    """
    Feature: PSROIPoolingGrad op.
    Description: Test the validation of input type.
    Expectation: success.
    """
    ms.context.set_context(mode=CTX_MODE,
                           device_target=DEVICE_TARGET)
    op = G.PSROIPoolingGrad(
        input_size=100,
        spatial_scale=1/16,
        group_size=7,
        output_dim=3
    )

    try:
        op(
            ms.Tensor([1.0], dtype=ms.float64),
            1.0
        )
    except TypeError:
        return
    else:
        assert False, "Expected TypeError to occur, but it does not occur."
