# Copyright 2020 Huawei Technologies Co., Ltd
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
""" test_vm """
import numpy as np

from .....vm_impl import vm


def test_avg_pooling():
    """ test_avg_pooling """
    input_data = np.array([[[[-4., -3., 1., 9.],
                             [-9., -1., 3., 4.],
                             [1., -1., -3., -6.],
                             [-2., -1., -2., -15.]]]]).astype(np.float32)
    out = vm.avg_pooling(input_data, pool_h=2, pool_w=2, stride=1)
    expect_out = [[[[-4.25, 0.0, 4.25],
                    [-2.5, -0.5, -0.5],
                    [-0.75, -1.75, -6.5]]]]
    assert (expect_out == out).all()


def test_avg_pool_grad():
    """ test_avg_pool_grad """
    # To do
    input_data = np.array([[[[1., 2, 3, 4],
                             [5, 6, 7, 8],
                             [9, 10, 11, 12],
                             [13, 14, 15, 16]]]]).astype(np.float32)
    dout = vm.avg_pooling(input_data, pool_h=2, pool_w=2, stride=1)
    print("vm.avg_pooling dout: ", dout)
    out = vm.avg_pool_grad(dout, input_data.shape, 2, 2, 1)
    print("vm.avg_pool_grad: ", out)
    assert True


def test_batch_norm():
    """ test_batch_norm """
    input_data = np.random.randint(0, 255, [1, 3, 224, 224])
    print("input_data.shape: ", input_data.shape)
    print("input_data: ", input_data)
    output = vm.batch_norm(input_data)
    print("vm.batch_norm: ", output)


def test_conv2d():
    """ test_conv2d """
    x = np.array([[[
        [3, 0, 1, 2, 7, 4],
        [1, 5, 8, 9, 3, 1],
        [2, 7, 2, 5, 1, 3],
        [0, 1, 3, 1, 7, 8],
        [4, 2, 1, 6, 2, 8],
        [2, 4, 5, 2, 3, 9]]]]).astype(np.float32)
    weight = np.array([[[[1, 0, -1], [1, 0, -1], [1, 0, -1]]]]).astype(np.float32)
    out = vm.conv2d(x, weight)
    expect_out = np.array([[[
        [-5., -4., 0., 8.],
        [-10., -2., 2., 3.],
        [0., -2., -4., -7.],
        [-3., -2., -3., -16.]]]]).astype(np.float32)
    assert (expect_out == out).all()


def test_conv2d_with_bias():
    """ test_conv2d_with_bias """
    x = np.array([[[
        [3, 0, 1, 2, 7, 4],
        [1, 5, 8, 9, 3, 1],
        [2, 7, 2, 5, 1, 3],
        [0, 1, 3, 1, 7, 8],
        [4, 2, 1, 6, 2, 8],
        [2, 4, 5, 2, 3, 9]]]]).astype(np.float32)
    weight = np.array([[[[1, 0, -1], [1, 0, -1], [1, 0, -1]]]]).astype(np.float32)
    bias = np.array([1]).astype(np.float32)
    out = vm.conv2d(x, weight, bias)
    expect_out = np.array([[[
        [-4., -3., 1., 9.],
        [-9., -1., 3., 4.],
        [1., -1., -3., -6.],
        [-2., -1., -2., -15.]]]]).astype(np.float32)
    assert (expect_out == out).all()


def test_conv2d_backprop_filter():
    """ test_conv2d_backprop_filter """
    x = np.array([[[
        [3, 0, 1, 2, 7, 4],
        [1, 5, 8, 9, 3, 1],
        [2, 7, 2, 5, 1, 3],
        [0, 1, 3, 1, 7, 8],
        [4, 2, 1, 6, 2, 8],
        [2, 4, 5, 2, 3, 9]]]]).astype(np.float32)
    weight = np.array([[[[1, 0, -1], [1, 0, -1], [1, 0, -1]]]]).astype(np.float32)
    out = vm.conv2d(x, weight)
    backprop_filter = vm.conv2d_backprop_filter(out, x, weight.shape)
    print(backprop_filter)
    assert True


def test_conv2d_backprop_input():
    """ test_conv2d_backprop_input """
    x = np.array([[[
        [3, 0, 1, 2, 7, 4],
        [1, 5, 8, 9, 3, 1],
        [2, 7, 2, 5, 1, 3],
        [0, 1, 3, 1, 7, 8],
        [4, 2, 1, 6, 2, 8],
        [2, 4, 5, 2, 3, 9]]]]).astype(np.float32)
    weight = np.array([[[[1, 0, -1], [1, 0, -1], [1, 0, -1]]]]).astype(np.float32)
    out = vm.conv2d(x, weight)
    grad = vm.conv2d_backprop_input(out, x.shape, weight)
    print(grad)
    assert True


def test_flatten():
    """ test_flatten """
    x = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float32)
    y = vm.flatten(x)
    assert ([1, 2, 3, 4, 5, 6] == y.T).all()
    assert np.float32 == y.dtype


def test_flatten2():
    """ test_flatten2 """
    x = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float32)
    y = vm.flatten2(x)
    assert ([1, 2, 3, 4, 5, 6] == y).all()
    assert (1, 6) == y.shape
    assert np.float32 == y.dtype


def test_flatten_batch():
    """ test_flatten_batch """
    x = np.array([[[9, 4, 14, 1],
                   [7, 10, 14, 13],
                   [1, 9, 16, 7],
                   [15, 16, 0, 4]],
                  [[16, 13, 13, 10],
                   [0, 12, 5, 9],
                   [15, 0, 11, 1],
                   [4, 16, 4, 1]],
                  [[2, 8, 1, 13],
                   [5, 15, 4, 11],
                   [8, 2, 17, 16],
                   [5, 13, 0, 2]],
                  [[14, 8, 6, 8],
                   [0, 8, 6, 15],
                   [9, 1, 8, 5],
                   [12, 6, 13, 8]],
                  [[13, 11, 6, 3],
                   [8, 6, 16, 5],
                   [7, 10, 0, 8],
                   [17, 17, 17, 3]]]).astype(np.float32)
    y = vm.flatten_batch(x)
    expect_out = np.array(
        [[9, 4, 14, 1, 7, 10, 14, 13, 1, 9, 16, 7, 15, 16, 0, 4],
         [16, 13, 13, 10, 0, 12, 5, 9, 15, 0, 11, 1, 4, 16, 4, 1],
         [2, 8, 1, 13, 5, 15, 4, 11, 8, 2, 17, 16, 5, 13, 0, 2],
         [14, 8, 6, 8, 0, 8, 6, 15, 9, 1, 8, 5, 12, 6, 13, 8],
         [13, 11, 6, 3, 8, 6, 16, 5, 7, 10, 0, 8, 17, 17, 17, 3]]).astype(np.float32)
    assert (expect_out == y).all()
    assert expect_out.shape == y.shape
    assert np.float32 == y.dtype


def test_im2col():
    """ test_im2col """
    img = np.ones([1, 1, 32, 32]).astype(np.float32) * 0.01
    print("input img: ", img)
    col = vm.im2col(img, 2, 3, 1, 1)
    print("output col.shape : ", col.shape)
    print("output col: ", col)
    print("output col.dtype: ", col.dtype)
    assert np.float32 == col.dtype


def test_matmul():
    """ test_matmul """
    x = np.array([1, 2, 3]).astype(np.float32)
    w = np.array([0, 1, 0.5]).astype(np.float32)
    y = vm.matmul(x, w)
    assert y == 3.5
    assert np.float32 == y.dtype


def test_max_pooling():
    """ test_max_pooling """
    input_data = np.array([[[
        [-4., -3., 1., 9.],
        [-9., -1., 3., 4.],
        [1., -1., -3., -6.],
        [-2., -1., -2., -15.]]]]).astype(np.float32)
    out = vm.max_pooling(input_data, pool_h=2, pool_w=2, stride=1)
    expect_out = [[[[-1., 3., 9.],
                    [1., 3., 4.],
                    [1., -1., -2.]]]]
    assert (expect_out == out).all()
    assert np.float32 == out.dtype


def test_np_convolve():
    """ test_np_convolve """
    out = np.convolve([1, 2, 3], [0, 1, 0.5]).astype(np.float32)
    assert ([0.0, 1.0, 2.5, 4.0, 1.5] == out).all()
    assert np.float32 == out.dtype


def test_np_convolve_same():
    """ test_np_convolve_same """
    out = np.convolve([1, 2, 3], [0, 1, 0.5], 'same').astype(np.float32)
    assert ([1.0, 2.5, 4.0] == out).all()
    assert np.float32 == out.dtype


def test_np_convolve_valid():
    """ test_np_convolve_valid """
    out = np.convolve([1, 2, 3], [0, 1, 0.5], 'valid').astype(np.float32)
    assert ([2.5] == out).all()
    assert np.float32 == out.dtype


def test_relu():
    """ test_relu """
    x = np.array([-0.32208174, 0.33999891]).astype(np.float32)
    y = vm.relu(x)
    assert np.allclose([-0., 0.33999891], y)
    assert np.float32 == y.dtype

    y = vm.relu_grad(y)
    assert (y == [0., 1.]).all()
    assert np.float32 == y.dtype


def test_softmax():
    """ test_softmax """
    logits = 2.84806275 * np.ones([1, 10]).astype(np.float32)
    y = vm.softmax(logits)
    assert np.allclose([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], y)
    assert np.float32 == y.dtype

    logits = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float32)
    y = vm.softmax(logits, axis=1)
    labels = [[0.09003057, 0.24472847, 0.66524096], [0.09003057, 0.24472847, 0.66524096]]
    assert np.allclose(labels, y)
    assert np.float32 == y.dtype


def test_softmax_cross_entropy_with_logit():
    """ test_softmax_cross_entropy_with_logit """
    logits = np.array([[1, 2, 3, 4, 2, 1, 0, 2, 1, 1], [1, 2, 4, 1, 0, 5, 0, 2, 1, 3]], dtype=np.float32)
    labels = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32)
    loss, dx = vm.softmax_cross_entropy_with_logits(logits, labels)
    print("logits.shape: ", logits.shape)
    print("logits: ", logits)
    print("softmax: ", vm.softmax(logits))
    print("labels: ", labels)
    print("loss: ", loss)
    print("dx: ", dx)
    assert np.float32 == loss.dtype
    assert np.float32 == dx.dtype
