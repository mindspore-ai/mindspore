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
Test PSROIPooling.
"""
import numpy as np
import pytest

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops.operations import nn_ops as G
from mindspore.common.api import _pynative_executor


DEVICE_TARGET = "GPU"
CTX_MODE = ms.context.GRAPH_MODE
ALL_CLOSE_CRITERION = 1e-4


class NetPSROIPooling(nn.Cell):
    """Simple net for PSROIPooling."""

    def __init__(self, spatial_scale, group_size, output_dim):
        super(NetPSROIPooling, self).__init__()
        self.ps_roi_pooling = G.PSROIPooling(spatial_scale, group_size, output_dim)

    def construct(self, dy, rois):
        return self.ps_roi_pooling(dy, rois)


def _ps_roi_pooling_case(data_type, mode, x_size_adjust=None):
    """Run op calculation."""
    np_rois = np.array([[[0.0000],
                         [150.3563],
                         [200.1320],
                         [579.3563],
                         [602.3452]],
                        [[1.0000],
                         [657.1263],
                         [302.8564],
                         [762.4214],
                         [567.9854]]]).astype(data_type)
    rois = Tensor.from_numpy(np_rois)

    batch_size = rois.shape[0]
    x_height = 80
    x_width = 48
    group_size = 2
    output_dim = 4
    spatial_scale = 1.0 / 16

    x_size = [batch_size, output_dim * group_size * group_size, x_height, x_width]
    if x_size_adjust is not None:
        x_size[x_size_adjust] += 1

    np_features = np.ones(x_size).astype(data_type)
    features = Tensor.from_numpy(np_features)

    ms.context.set_context(mode=mode,
                           device_target=DEVICE_TARGET)
    net = NetPSROIPooling(spatial_scale, group_size, output_dim)
    runtime_error_occurred = False
    try:
        output = net(features, rois)
    except RuntimeError:
        runtime_error_occurred = True
        if x_size_adjust is None:
            raise
    if x_size_adjust is not None:
        assert runtime_error_occurred, "Expected RuntimeError to occur, but it does not occur."
        return
    output_ms = output.asnumpy()

    output_tv = np.array([[[[1., 1.],
                            [1., 1.]],
                           [[1., 1.],
                            [1., 1.]],
                           [[1., 1.],
                            [1., 1.]],
                           [[1., 1.],
                            [1., 1.]]],
                          [[[1., 1.],
                            [1., 1.]],
                           [[1., 1.],
                            [1., 1.]],
                           [[1., 1.],
                            [1., 1.]],
                           [[1., 1.],
                            [1., 1.]]]], dtype=data_type)

    assert np.allclose(
        output_ms, output_tv,
        atol=ALL_CLOSE_CRITERION, rtol=ALL_CLOSE_CRITERION)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu
@pytest.mark.env_onecard
def test_ps_roi_pooling_dynamic_shape():
    """
    Feature: PSROIPooling op.
    Description: Test the dynamic shape behavior of PSROIPooing op.
    Expectation: success.
    """
    input_dx = Tensor(shape=[None, 16, None, None], dtype=ms.float32)
    input_roi = Tensor(shape=[None, 5, None], dtype=ms.float32)
    np_rois = np.array([[[0.0000],
                         [150.3563],
                         [200.1320],
                         [579.3563],
                         [602.3452]],
                        [[1.0000],
                         [657.1263],
                         [302.8564],
                         [762.4214],
                         [567.9854]]]).astype(np.float32)
    rois = Tensor.from_numpy(np_rois)

    batch_size = rois.shape[0]
    x_height = 80
    x_width = 48
    group_size = 2
    output_dim = 4
    spatial_scale = 1.0 / 16

    x_size = [batch_size, output_dim * group_size * group_size, x_height, x_width]

    np_features = np.ones(x_size).astype(np.float32)
    features = Tensor.from_numpy(np_features)

    ms.context.set_context(mode=CTX_MODE,
                           device_target=DEVICE_TARGET)
    net = NetPSROIPooling(spatial_scale, group_size, output_dim)
    net.set_inputs(input_dx, input_roi)
    output = net(features, rois)
    output_ms = output.asnumpy()

    output_tv = np.array([[[[1., 1.],
                            [1., 1.]],
                           [[1., 1.],
                            [1., 1.]],
                           [[1., 1.],
                            [1., 1.]],
                           [[1., 1.],
                            [1., 1.]]],
                          [[[1., 1.],
                            [1., 1.]],
                           [[1., 1.],
                            [1., 1.]],
                           [[1., 1.],
                            [1., 1.]],
                           [[1., 1.],
                            [1., 1.]]]], dtype=np.float32)

    assert np.allclose(
        output_ms, output_tv,
        atol=ALL_CLOSE_CRITERION, rtol=ALL_CLOSE_CRITERION)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu
@pytest.mark.env_onecard
def test_ps_roi_pooling_mind_ir():
    """
    Feature: PSROIPooling op.
    Description: Test the MindIR behavior of PSROIPooing op.
    Expectation: success.
    """
    ms.context.set_context(mode=CTX_MODE, device_target=DEVICE_TARGET)
    data_type = np.float32
    rois_np = np.array([[[0.0000],
                         [150.3563],
                         [200.1320],
                         [579.3563],
                         [602.3452]],
                        [[1.0000],
                         [657.1263],
                         [302.8564],
                         [762.4214],
                         [567.9854]]]).astype(data_type)
    rois_ms = ms.Tensor(rois_np)

    batch_size = rois_ms.shape[0]
    x_height = 80
    x_width = 48
    group_size = 2
    output_dim = 4
    spatial_scale = 1.0 / 16

    x_size = [batch_size, output_dim * group_size * group_size, x_height, x_width]
    np_features = np.ones(x_size).astype(data_type)
    features = Tensor.from_numpy(np_features)

    net = NetPSROIPooling(spatial_scale, group_size, output_dim)
    old_out = net(features, rois_ms)
    ms.export(
        net, features, rois_ms,
        file_name="ps_roi_pooling",
        file_format='MINDIR')

    graph = ms.load("ps_roi_pooling.mindir")
    new_net = nn.GraphCell(graph)
    new_out = new_net(features, rois_ms)
    assert np.allclose(
        old_out.asnumpy(), new_out.asnumpy(),
        atol=ALL_CLOSE_CRITERION, rtol=ALL_CLOSE_CRITERION)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu
@pytest.mark.env_onecard
def test_ps_roi_pooling_graph_mode():
    """
    Feature: PSROIPooling op.
    Description: Test the normal behavior of PSROIPooing op.
    Expectation: success.
    """
    _ps_roi_pooling_case(
        data_type=np.float32,
        mode=CTX_MODE)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu
@pytest.mark.env_onecard
def test_ps_roi_pooling_x_1_shape_wrong():
    """
    Feature: PSROIPooling op.
    Description: Test the validation of shape.
    Expectation: success.
    """
    _ps_roi_pooling_case(
        data_type=np.float32,
        mode=CTX_MODE,
        x_size_adjust=1
    )


@pytest.mark.level1
@pytest.mark.platform_x86_gpu
@pytest.mark.env_onecard
def test_ps_roi_pooling_pynative_mode():
    """
    Feature: PSROIPooling op.
    Description: Test the normal behavior of PSROIPooing op.
    Expectation: success.
    """
    _ps_roi_pooling_case(
        data_type=np.float32,
        mode=ms.context.PYNATIVE_MODE)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu
@pytest.mark.env_onecard
def test_ps_roi_pooling_spatial_scale_attr_type_wrong():
    """
    Feature: PSROIPooling op.
    Description: Test the validation of attrs.
    Expectation: success.
    """
    _check_attr_validation(arg_name="spatial_scale", arg_value=object())


@pytest.mark.level1
@pytest.mark.platform_x86_gpu
@pytest.mark.env_onecard
def test_ps_roi_pooling_spatial_scale_attr_range_wrong():
    """
    Feature: PSROIPooling op.
    Description: Test the validation of attrs.
    Expectation: success.
    """
    _check_attr_validation(arg_name="spatial_scale", arg_value=-0.15)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu
@pytest.mark.env_onecard
def test_ps_roi_pooling_group_size_attr_type_wrong():
    """
    Feature: PSROIPooling op.
    Description: Test the validation of attrs.
    Expectation: success.
    """
    _check_attr_validation(arg_name="group_size", arg_value=7.1)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu
@pytest.mark.env_onecard
def test_ps_roi_pooling_group_size_attr_range_wrong():
    """
    Feature: PSROIPooling op.
    Description: Test the validation of attrs.
    Expectation: success.
    """
    _check_attr_validation(arg_name="group_size", arg_value=-1)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu
@pytest.mark.env_onecard
def test_ps_roi_pooling_output_dim_attr_type_wrong():
    """
    Feature: PSROIPooling op.
    Description: Test the validation of attrs.
    Expectation: success.
    """
    _check_attr_validation(arg_name="output_dim", arg_value=7.1)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu
@pytest.mark.env_onecard
def test_ps_roi_pooling_output_dim_attr_range_wrong():
    """
    Feature: PSROIPooling op.
    Description: Test the validation of attrs.
    Expectation: success.
    """
    _check_attr_validation(arg_name="output_dim", arg_value=-1)


def _check_attr_validation(arg_name, arg_value):
    ms.context.set_context(mode=CTX_MODE,
                           device_target=DEVICE_TARGET)
    op_kwargs = {
        "spatial_scale": 1 / 16,
        "group_size": 7,
        "output_dim": 3
    }
    op_kwargs[arg_name] = arg_value
    try:
        G.PSROIPooling(**op_kwargs)
    except TypeError:
        assert True
    except ValueError:
        assert True
    else:
        assert False


@pytest.mark.level1
@pytest.mark.platform_x86_gpu
@pytest.mark.env_onecard
def test_ps_roi_pooling_input_args_num():
    """
    Feature: PSROIPooling op.
    Description: Test the validation of input args num.
    Expectation: success.
    """
    ms.context.set_context(mode=CTX_MODE,
                           device_target=DEVICE_TARGET)
    op = G.PSROIPooling(
        spatial_scale=1/16,
        group_size=2,
        output_dim=4
    )
    np_rois = np.array([[[0.0000],
                         [150.3563],
                         [200.1320],
                         [579.3563],
                         [602.3452]],
                        [[1.0000],
                         [657.1263],
                         [302.8564],
                         [762.4214],
                         [567.9854]]])
    rois = ms.Tensor(np_rois)
    np_features = np.random.randn(2, 4 * 2 * 2, 80, 48)
    features = Tensor.from_numpy(np_features)
    try:
        op(
            ms.Tensor(features, dtype=ms.float32),
            ms.Tensor(rois, dtype=ms.float32),
            ms.Tensor(features, dtype=ms.float32)
        )
        _pynative_executor.sync()
    except TypeError:
        return
    else:
        assert False, "Expected ValueError to occur, but it does not occur."


@pytest.mark.level1
@pytest.mark.platform_x86_gpu
@pytest.mark.env_onecard
def test_ps_roi_pooling_input_type_unsupported1():
    """
    Feature: PSROIPooling op.
    Description: Test the validation of input type.
    Expectation: success.
    """
    ms.context.set_context(mode=CTX_MODE,
                           device_target=DEVICE_TARGET)
    op = G.PSROIPooling(
        spatial_scale=1 / 16,
        group_size=2,
        output_dim=4
    )
    np_rois = np.array([[[0.0000],
                         [150.3563],
                         [200.1320],
                         [579.3563],
                         [602.3452]],
                        [[1.0000],
                         [657.1263],
                         [302.8564],
                         [762.4214],
                         [567.9854]]])
    rois = ms.Tensor(np_rois)
    try:
        op(
            1.0,
            ms.Tensor(rois, dtype=ms.float32)
        )
    except TypeError:
        return
    else:
        assert False, "Expected TypeError to occur, but it does not occur."


@pytest.mark.level1
@pytest.mark.platform_x86_gpu
@pytest.mark.env_onecard
def test_ps_roi_pooling_input_type_unsupported2():
    """
    Feature: PSROIPooling op.
    Description: Test the validation of input type.
    Expectation: success.
    """
    ms.context.set_context(mode=CTX_MODE,
                           device_target=DEVICE_TARGET)
    op = G.PSROIPooling(
        spatial_scale=1 / 16,
        group_size=2,
        output_dim=4
    )
    np_features = np.random.randn(2, 4 * 2 * 2, 80, 48)
    features = Tensor.from_numpy(np_features)
    try:
        op(
            ms.Tensor(features, dtype=ms.float32),
            1.0
        )
    except TypeError:
        return
    else:
        assert False, "Expected TypeError to occur, but it does not occur."
