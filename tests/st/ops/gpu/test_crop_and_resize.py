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
from tests.mark_utils import arg_mark
import numpy as np
import pytest

from mindspore import context, Tensor
from mindspore.ops import operations as P
from mindspore import nn


class NetCropAndResize(nn.Cell):
    def __init__(self, method_="bilinear", extrapolation_value_=0.0):
        super(NetCropAndResize, self).__init__()
        self.op = P.CropAndResize(
            method=method_, extrapolation_value=extrapolation_value_)

    def construct(self, image, boxes, box_index, channel):
        return self.op(image, boxes, box_index, channel)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_crop_and_resize_int8_bilinear(datatype=np.int8):
    """
    Feature: crop_and_resize kernel
    Description: test crop_and_resize int8
    Expectation: case pass
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    batch_size = 2
    image_height = 32
    image_width = 18
    channels = 2
    crop_size = (5, 3)
    total_values = batch_size * image_height * image_width * channels
    input_data = np.arange(0, total_values).reshape(
        (batch_size, image_height, image_width, channels))
    input_boxes = np.array(
        [[0, 0.5, 0.5, 0.0], [0, 0, 0.75, 1.75]]).astype(np.float32)
    input_box_index = np.array([1, 0]).astype(np.int32)
    input_data_tensor = Tensor(input_data.astype(datatype))
    input_boxes_tensor = Tensor(input_boxes)
    input_box_index_tensor = Tensor(input_box_index)
    net = NetCropAndResize("bilinear", 0.5)
    output = net(input_data_tensor, input_boxes_tensor,
                 input_box_index_tensor, crop_size)
    output_ms = output.asnumpy()
    expected_output = np.array([[[[-111.0, -110.0], [-119.5, -118.5], [-128.0, -127.0]],
                                 [[28.5, 29.5], [20.0, 21.0], [11.5, 12.5]],
                                 [[-88.0, -87.0], [-96.5, -95.5], [-41.0, -40.0]],
                                 [[51.5, 52.5], [43.0, 44.0], [34.5, 35.5]],
                                 [[-65.0, -64.0], [-73.5, -72.5], [-82.0, -81.0]]],
                                [[[0.0, 1.0], [29.75, 30.75], [0.5, 0.5]],
                                 [[-46.75, -45.75], [-17.0, -16.0], [0.5, 0.5]],
                                 [[-93.5, -92.5], [-63.75, -62.75], [0.5, 0.5]],
                                 [[3.75, 4.75], [-110.5, -109.5], [0.5, 0.5]],
                                 [[69.0, 70.0], [98.75, 99.75], [0.5, 0.5]]]]).astype(np.float32)
    assert np.allclose(output_ms, expected_output, 1e-5, 1e-5)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_crop_and_resize_int16_nearest(datatype=np.int16):
    """
    Feature: crop_and_resize kernel
    Description: test crop_and_resize int16
    Expectation: case pass
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    batch_size = 2
    image_height = 32
    image_width = 18
    channels = 2
    crop_size = (5, 3)
    total_values = batch_size * image_height * image_width * channels
    input_data = np.arange(0, total_values).reshape(
        (batch_size, image_height, image_width, channels))
    input_boxes = np.array(
        [[0, 0.5, 0.5, 0.0], [0, 0, 0.75, 1.75]]).astype(np.float32)
    input_box_index = np.array([1, 0]).astype(np.int32)
    input_data_tensor = Tensor(input_data.astype(datatype))
    input_boxes_tensor = Tensor(input_boxes)
    input_box_index_tensor = Tensor(input_box_index)
    net = NetCropAndResize("nearest", 0.5)
    output = net(input_data_tensor, input_boxes_tensor,
                 input_box_index_tensor, crop_size)
    output_ms = output.asnumpy()
    expected_output = np.array([[[[1170.0, 1171.0], [1160.0, 1161.0], [1152.0, 1153.0]],
                                 [[1314.0, 1315.0], [1304.0, 1305.0], [1296.0, 1297.0]],
                                 [[1458.0, 1459.0], [1448.0, 1449.0], [1440.0, 1441.0]],
                                 [[1602.0, 1603.0], [1592.0, 1593.0], [1584.0, 1585.0]],
                                 [[1746.0, 1747.0], [1736.0, 1737.0], [1728.0, 1729.0]]],
                                [[[0.0, 1.0], [30.0, 31.0], [0.5, 0.5]],
                                 [[216.0, 217.0], [246.0, 247.0], [0.5, 0.5]],
                                 [[432.0, 433.0], [462.0, 463.0], [0.5, 0.5]],
                                 [[612.0, 613.0], [642.0, 643.0], [0.5, 0.5]],
                                 [[828.0, 829.0], [858.0, 859.0], [0.5, 0.5]]]]).astype(np.float32)
    assert np.allclose(output_ms, expected_output, 1e-5, 1e-5)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_crop_and_resize_int32_bilinear_v2(datatype=np.int32):
    """
    Feature: crop_and_resize kernel
    Description: test crop_and_resize int32
    Expectation: case pass
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    batch_size = 2
    image_height = 32
    image_width = 18
    channels = 2
    crop_size = (5, 3)
    offset = 8795
    total_values = batch_size * image_height * image_width * channels
    input_data = np.arange(0 + offset, total_values + offset).reshape(
        (batch_size, image_height, image_width, channels))
    input_boxes = np.array(
        [[0, 0.5, 0.5, 0.0], [0, 0, 0.75, 1.75]]).astype(np.float32)
    input_box_index = np.array([1, 0]).astype(np.int32)
    input_data_tensor = Tensor(input_data.astype(datatype))
    input_boxes_tensor = Tensor(input_boxes)
    input_box_index_tensor = Tensor(input_box_index)
    net = NetCropAndResize("bilinear_v2", 0.369)
    output = net(input_data_tensor, input_boxes_tensor,
                 input_box_index_tensor, crop_size)
    output_ms = output.asnumpy()
    expected_output = np.array([[[[9964.0, 9965.0], [9955.5, 9956.5], [9947.0, 9948.0]],
                                 [[10103.5, 10104.5], [10095.0, 10096.0], [10086.5, 10087.5]],
                                 [[10243.0, 10244.0], [10234.5, 10235.5], [10226.0, 10227.0]],
                                 [[10382.5, 10383.5], [10374.0, 10375.0], [10365.5, 10366.5]],
                                 [[10522.0, 10523.0], [10513.5, 10514.5], [10505.0, 10506.0]]],
                                [[[8795.0, 8796.0], [8824.75, 8825.75], [0.368999987, 0.368999987]],
                                 [[9004.25, 9005.25], [9034.0, 9035.0], [0.368999987, 0.368999987]],
                                 [[9213.5, 9214.5], [9243.25, 9244.25], [0.368999987, 0.368999987]],
                                 [[9422.75, 9423.75], [9452.5, 9453.5], [0.368999987, 0.368999987]],
                                 [[9632.0, 9633.0], [9661.75, 9662.75], [0.368999987, 0.368999987]]]]).astype(
                                     np.float32)
    assert np.allclose(output_ms, expected_output, 1e-5, 1e-5)

@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_crop_and_resize_float16_nearest(datatype=np.float16):
    """
    Feature: crop_and_resize kernel
    Description: test crop_and_resize int16
    Expectation: case pass
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    batch_size = 2
    image_height = 50
    image_width = 40
    channels = 3
    crop_size = (5, 3)
    offset = 0
    total_values = batch_size * image_height * image_width * channels
    input_data = np.arange(0 + offset, total_values + offset).reshape(
        (batch_size, image_height, image_width, channels))
    input_boxes = np.array(
        [[0.23, 0.5, 0.75, 0.0], [0, 0.1, 0.75, 1.75]]).astype(np.float32)
    input_box_index = np.array([1, 0]).astype(np.int32)
    input_data_tensor = Tensor(input_data.astype(datatype))
    input_boxes_tensor = Tensor(input_boxes)
    input_box_index_tensor = Tensor(input_box_index)
    net = NetCropAndResize("nearest", 0.0)
    output = net(input_data_tensor, input_boxes_tensor,
                 input_box_index_tensor, crop_size)
    output_ms = output.asnumpy()
    expected_output = np.array([[[[7380.0, 7380.0, 7384.0], [7352.0, 7352.0, 7352.0],
                                  [7320.0, 7320.0, 7320.0]],
                                 [[8224.0, 8224.0, 8224.0], [8192.0, 8192.0, 8192.0],
                                  [8160.0, 8160.0, 8160.0]],
                                 [[8944.0, 8944.0, 8944.0], [8912.0, 8912.0, 8912.0],
                                  [8880.0, 8880.0, 8880.0]],
                                 [[9664.0, 9664.0, 9664.0], [9632.0, 9632.0, 9632.0],
                                  [9600.0, 9600.0, 9600.0]],
                                 [[10496.0, 10504.0, 10504.0], [10472.0, 10472.0, 10472.0],
                                  [10440.0, 10440.0, 10440.0]]],
                                [[[12.0, 13.0, 14.0], [108.0, 109.0, 110.0], [0.0, 0.0, 0.0]],
                                 [[1092.0, 1093.0, 1094.0], [1188.0, 1189.0, 1190.0], [0.0, 0.0, 0.0]],
                                 [[2172.0, 2172.0, 2174.0], [2268.0, 2268.0, 2270.0], [0.0, 0.0, 0.0]],
                                 [[3372.0, 3372.0, 3374.0], [3468.0, 3468.0, 3470.0], [0.0, 0.0, 0.0]],
                                 [[4452.0, 4452.0, 4456.0], [4548.0, 4548.0, 4552.0],
                                  [0.0, 0.0, 0.0]]]]).astype(np.float32)
    assert np.allclose(output_ms, expected_output, 1e-5, 1e-5)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_crop_and_resize_float32_bilinear(datatype=np.float32):
    """
    Feature: crop_and_resize kernel
    Description: test crop_and_resize float32
    Expectation: case pass
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    batch_size = 2
    image_height = 512
    image_width = 256
    channels = 3
    crop_size = (5, 3)
    offset = 5000
    total_values = batch_size * image_height * image_width * channels
    input_data = np.arange(0 + offset, total_values + offset).reshape(
        (batch_size, image_height, image_width, channels))
    input_boxes = np.array(
        [[0.23, 0.5, 0.75, 0.0], [0, 0.1, 0.75, 1.75]]).astype(np.float32)
    input_box_index = np.array([1, 0]).astype(np.int32)
    input_data_tensor = Tensor(input_data.astype(datatype))
    input_boxes_tensor = Tensor(input_boxes)
    input_box_index_tensor = Tensor(input_box_index)
    net = NetCropAndResize("bilinear", 0.0)
    output = net(input_data_tensor, input_boxes_tensor,
                 input_box_index_tensor, crop_size)
    output_ms = output.asnumpy()
    expected_output = np.array([[[[488861.53, 488862.53, 488863.53],
                                  [488670.28, 488671.28, 488672.28],
                                  [488479.03, 488480.03, 488481.03]],
                                 [[539879.75, 539880.75, 539881.75],
                                  [539688.5, 539689.5, 539690.5],
                                  [539497.25, 539498.25, 539499.25]],
                                 [[590898.0, 590899.0, 590900.0], [590706.75, 590707.75, 590708.75],
                                  [590515.5, 590516.5, 590517.5]],
                                 [[641916.25, 641917.25, 641918.25], [641725.0, 641726.0, 641727.0],
                                  [641533.75, 641534.75, 641535.75]],
                                 [[692934.5, 692935.5, 692936.5], [692743.25, 692744.25, 692745.25],
                                  [692552.0, 692553.0, 692554.0]]],
                                [[[5076.5, 5077.5, 5078.5], [5707.625, 5708.625, 5709.625], [0.0, 0.0, 0.0]],
                                 [[78660.5, 78661.5, 78662.5], [79291.625, 79292.625, 79293.625], [0.0, 0.0, 0.0]],
                                 [[152244.5, 152245.5, 152246.5], [152875.62, 152876.62, 152877.62],
                                  [0.0, 0.0, 0.0]],
                                 [[225828.5, 225829.5, 225830.5], [226459.62, 226460.62, 226461.62],
                                  [0.0, 0.0, 0.0]],
                                 [[299412.5, 299413.5, 299414.5], [300043.62, 300044.62, 300045.62],
                                  [0.0, 0.0, 0.0]]]]).astype(np.float32)
    assert np.allclose(output_ms, expected_output, 1e-5, 1e-5)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_crop_and_resize_float64_nearest(datatype=np.float64):
    """
    Feature: crop_and_resize kernel
    Description: test crop_and_resize float64
    Expectation: case pass
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    batch_size = 2
    image_height = 50
    image_width = 25
    channels = 3
    crop_size = (5, 3)
    offset = 7549
    total_values = batch_size * image_height * image_width * channels
    input_data = np.arange(0 + offset, total_values + offset).reshape(
        (batch_size, image_height, image_width, channels))
    input_boxes = np.array(
        [[0.23, 0.5, 0.75, 0.0], [0, 0.1, 0.75, 1.75]]).astype(np.float32)
    input_box_index = np.array([1, 0]).astype(np.int32)
    input_data_tensor = Tensor(input_data.astype(datatype))
    input_boxes_tensor = Tensor(input_boxes)
    input_box_index_tensor = Tensor(input_box_index)
    net = NetCropAndResize("nearest", 0.0)
    output = net(input_data_tensor, input_boxes_tensor,
                 input_box_index_tensor, crop_size)
    output_ms = output.asnumpy()
    expected_output = np.array([[[[12160.0, 12161.0, 12162.0], [12142.0, 12143.0, 12144.0],
                                  [12124.0, 12125.0, 12126.0]],
                                 [[12685.0, 12686.0, 12687.0], [12667.0, 12668.0, 12669.0],
                                  [12649.0, 12650.0, 12651.0]],
                                 [[13135.0, 13136.0, 13137.0], [13117.0, 13118.0, 13119.0],
                                  [13099.0, 13100.0, 13101.0]],
                                 [[13585.0, 13586.0, 13587.0], [13567.0, 13568.0, 13569.0],
                                  [13549.0, 13550.0, 13551.0]],
                                 [[14110.0, 14111.0, 14112.0], [14092.0, 14093.0, 14094.0],
                                  [14074.0, 14075.0, 14076.0]]],
                                [[[7555.0, 7556.0, 7557.0], [7615.0, 7616.0, 7617.0], [0.0, 0.0, 0.0]],
                                 [[8230.0, 8231.0, 8232.0], [8290.0, 8291.0, 8292.0], [0.0, 0.0, 0.0]],
                                 [[8905.0, 8906.0, 8907.0], [8965.0, 8966.0, 8967.0], [0.0, 0.0, 0.0]],
                                 [[9655.0, 9656.0, 9657.0], [9715.0, 9716.0, 9717.0], [0.0, 0.0, 0.0]],
                                 [[10330.0, 10331.0, 10332.0], [10390.0, 10391.0, 10392.0],
                                  [0.0, 0.0, 0.0]]]]).astype(np.float32)
    assert np.allclose(output_ms, expected_output, 1e-5, 1e-5)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_crop_and_resize_int64_bilinearv2(datatype=np.int64):
    """
    Feature: crop_and_resize kernel
    Description: test crop_and_resize int64
    Expectation: case pass
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    batch_size = 2
    image_height = 50
    image_width = 25
    channels = 3
    crop_size = (5, 3)
    offset = 7549
    total_values = batch_size * image_height * image_width * channels
    input_data = np.arange(0 + offset, total_values + offset).reshape(
        (batch_size, image_height, image_width, channels))
    input_boxes = np.array(
        [[0.23, 0.5, 0.75, 0.0], [0, 0.1, 0.75, 1.75]]).astype(np.float32)
    input_box_index = np.array([1, 0]).astype(np.int32)
    input_data_tensor = Tensor(input_data.astype(datatype))
    input_boxes_tensor = Tensor(input_boxes)
    input_box_index_tensor = Tensor(input_box_index)
    net = NetCropAndResize("bilinear_v2", 0.0)
    output = net(input_data_tensor, input_boxes_tensor,
                 input_box_index_tensor, crop_size)
    output_ms = output.asnumpy()
    expected_output = np.array([[[[12180.25, 12181.25, 12182.25], [12162.25, 12163.25, 12164.25],
                                  [12144.25, 12145.25, 12146.25]],
                                 [[12658.0, 12659.0, 12660.0], [12640.0, 12641.0, 12642.0],
                                  [12622.0, 12623.0, 12624.0]],
                                 [[13135.75, 13136.75, 13137.75], [13117.75, 13118.75, 13119.75],
                                  [13099.75, 13100.75, 13101.75]],
                                 [[13613.5, 13614.5, 13615.5], [13595.5, 13596.5, 13597.5],
                                  [13577.5, 13578.5, 13579.5]],
                                 [[14091.25, 14092.25, 14093.25], [14073.25, 14074.25, 14075.25],
                                  [14055.25, 14056.25, 14057.25]]],
                                [[[7556.2001953125, 7557.2001953125, 7558.2001953125],
                                  [7615.60009765625, 7616.60009765625, 7617.60009765625],
                                  [0.0, 0.0, 0.0]],
                                 [[8245.2626953125, 8246.2626953125, 8247.2626953125],
                                  [8304.662109375, 8305.662109375, 8306.662109375],
                                  [0.0, 0.0, 0.0]],
                                 [[8934.3251953125, 8935.3251953125, 8936.3251953125],
                                  [8993.724609375, 8994.724609375, 8995.724609375], [0.0, 0.0, 0.0]],
                                 [[9623.3876953125, 9624.3876953125, 9625.3876953125],
                                  [9682.787109375, 9683.787109375, 9684.787109375],
                                  [0.0, 0.0, 0.0]],
                                 [[10312.4501953125, 10313.4501953125, 10314.4501953125],
                                  [10371.849609375, 10372.849609375, 10373.849609375],
                                  [0.0, 0.0, 0.0]]]]).astype(np.float32)
    assert np.allclose(output_ms, expected_output, 1e-5, 1e-5)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_crop_and_resize_uint8_nearest(datatype=np.uint8):
    """
    Feature: crop_and_resize kernel
    Description: test crop_and_resize uint8
    Expectation: case pass
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    batch_size = 2
    image_height = 7
    image_width = 5
    channels = 2
    crop_size = (5, 3)
    offset = 0
    total_values = batch_size * image_height * image_width * channels
    input_data = np.arange(0 + offset, total_values + offset).reshape(
        (batch_size, image_height, image_width, channels))
    input_boxes = np.array(
        [[0.23, 0.5, 0.75, 0.0], [0, 0.1, 0.75, 1.75]]).astype(np.float32)
    input_box_index = np.array([1, 0]).astype(np.int32)
    input_data_tensor = Tensor(input_data.astype(datatype))
    input_boxes_tensor = Tensor(input_boxes)
    input_box_index_tensor = Tensor(input_box_index)
    net = NetCropAndResize("nearest", 0.0)
    output = net(input_data_tensor, input_boxes_tensor,
                 input_box_index_tensor, crop_size)
    output_ms = output.asnumpy()
    expected_output = np.array([[[[84.0, 85.0], [82.0, 83.0], [80.0, 81.0]],
                                 [[94.0, 95.0], [92.0, 93.0], [90.0, 91.0]],
                                 [[104.0, 105.0], [102.0, 103.0], [100.0, 101.0]],
                                 [[114.0, 115.0], [112.0, 113.0], [110.0, 111.0]],
                                 [[124.0, 125.0], [122.0, 123.0], [120.0, 121.0]]],
                                [[[0.0, 1.0], [8.0, 9.0], [0.0, 0.0]],
                                 [[10.0, 11.0], [18.0, 19.0], [0.0, 0.0]],
                                 [[20.0, 21.0], [28.0, 29.0], [0.0, 0.0]],
                                 [[30.0, 31.0], [38.0, 39.0], [0.0, 0.0]],
                                 [[50.0, 51.0], [58.0, 59.0], [0.0, 0.0]]]]).astype(np.float32)
    assert np.allclose(output_ms, expected_output, 1e-5, 1e-5)

@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_crop_and_resize_uint16_bilinear(datatype=np.uint16):
    """
    Feature: crop_and_resize kernel
    Description: test crop_and_resize uint16
    Expectation: case pass
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    batch_size = 2
    image_height = 50
    image_width = 30
    channels = 3
    crop_size = (5, 3)
    offset = 0
    total_values = batch_size * image_height * image_width * channels
    input_data = np.arange(0 + offset, total_values + offset).reshape(
        (batch_size, image_height, image_width, channels))
    input_boxes = np.array(
        [[0.23, 0.5, 0.75, 0.0], [0, 0.1, 0.75, 1.75]]).astype(np.float32)
    input_box_index = np.array([1, 0]).astype(np.int32)
    input_data_tensor = Tensor(input_data.astype(datatype))
    input_boxes_tensor = Tensor(input_boxes)
    input_box_index_tensor = Tensor(input_box_index)
    net = NetCropAndResize("bilinear", 0.0)
    output = net(input_data_tensor, input_boxes_tensor,
                 input_box_index_tensor, crop_size)
    output_ms = output.asnumpy()
    expected_output = np.array([[[[5557.7998046875, 5558.7998046875, 5559.7998046875],
                                  [5536.0498046875, 5537.0498046875, 5538.0498046875],
                                  [5514.2998046875, 5515.2998046875, 5516.2998046875]],
                                 [[6131.10009765625, 6132.10009765625, 6133.10009765625],
                                  [6109.35009765625, 6110.35009765625, 6111.35009765625],
                                  [6087.60009765625, 6088.60009765625, 6089.60009765625]],
                                 [[6704.39990234375, 6705.39990234375, 6706.39990234375],
                                  [6682.64990234375, 6683.64990234375, 6684.64990234375],
                                  [6660.89990234375, 6661.89990234375, 6662.89990234375]],
                                 [[7277.7001953125, 7278.7001953125, 7279.7001953125],
                                  [7255.9501953125, 7256.9501953125, 7257.9501953125],
                                  [7234.2001953125, 7235.2001953125, 7236.2001953125]],
                                 [[7851.0, 7852.0, 7853.0], [7829.25, 7830.25, 7831.25],
                                  [7807.5, 7808.5, 7809.5]]],
                                [[[8.700000762939453, 9.700000762939453, 10.700000762939453],
                                  [80.4749984741211, 81.4749984741211, 82.4749984741211],
                                  [0.0, 0.0, 0.0]],
                                 [[835.5750122070312, 836.5750122070312, 837.5750122070312],
                                  [907.3499755859375, 908.3499755859375, 909.3499755859375], [0.0, 0.0, 0.0]],
                                 [[1662.449951171875, 1663.449951171875, 1664.449951171875],
                                  [1734.2249755859375, 1735.2249755859375, 1736.2249755859375],
                                  [0.0, 0.0, 0.0]],
                                 [[2489.324951171875, 2490.324951171875, 2491.324951171875],
                                  [2561.10009765625, 2562.10009765625, 2563.10009765625], [0.0, 0.0, 0.0]],
                                 [[3316.199951171875, 3317.199951171875, 3318.199951171875],
                                  [3387.97509765625, 3388.97509765625, 3389.97509765625],
                                  [0.0, 0.0, 0.0]]]]).astype(np.float32)
    assert np.allclose(output_ms, expected_output, 1e-5, 1e-5)
