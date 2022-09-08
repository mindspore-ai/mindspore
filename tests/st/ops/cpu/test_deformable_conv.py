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

import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
import mindspore.ops as ops
import mindspore.common.dtype as mstype
from mindspore.ops import functional as F
from mindspore.ops.operations import nn_ops as NN

context.set_context(device_target='CPU')


class Net(nn.Cell):
    def construct(self, x, weight, offsets, kh, kw, strides=(1, 1, 1, 1), padding=(0, 0, 0, 0), bias=None,
                  dilations=(1, 1, 1, 1)):
        return ops.deformable_conv2d(x, weight, offsets, (kh, kw), strides, padding, bias, dilations)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_deformable_conv2d():
    """"
    Feature: deformable_conv2d function.
    Description: Test case for simplest deformable_conv2d.
    Expectation: The results are as expected.
    """
    kh, kw = 1, 1
    # x shape [1, 1, 1, 2]
    x = np.array([[[[-0.41675785, -0.05626683]]]]).astype(np.float32)
    x = Tensor(x, mstype.float32)
    # weight shape [1, 1, 1, 1]
    weight = np.array([[[[-2.1361961]]]]).astype(np.float32)
    weight = Tensor(weight, mstype.float32)
    # offsets shape [1, 3, 1, 2]
    offsets = np.array([[[[1.6402708, -1.7934356]],
                         [[-0.84174734, 0.5028814]],
                         [[-1.2452881, -1.0579522]]]]).astype(np.float32)
    offsets = Tensor(offsets, mstype.float32)
    out = Net()(x, weight, offsets, kh, kw)
    # expected output: [1, 1, 1, 2]
    expected = np.array([[[[-0.00852099, -0.09671781]]]]).astype(np.float32)
    assert np.allclose(out.asnumpy(), expected)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_required_inputs():
    """"
    Feature: deformable_conv2d function.
    Description: Test case for simplest deformable_conv2d.
    Expectation: The results are as expected.
    """
    x = Tensor(np.arange(2 * 3 * 5 * 5).reshape(2, 3, 5, 5), mstype.float32)
    kh, kw = 3, 3
    weight = Tensor(np.arange(5 * 3 * kh * kw).reshape(5, 3, kh, kw), mstype.float32)
    offsets = Tensor(np.ones((2, 3 * kh * kw, 3, 3)), mstype.float32)
    output = Net()(x, weight, offsets, kh, kw)
    expect = np.array([[[[17325., 17676., 11547.],
                         [19080., 19431., 12672.],
                         [11991., 12198., 7920.]],

                        [[44298., 45378., 30258.],
                         [49698., 50778., 33813.],
                         [33618., 34311., 22824.]],

                        [[71271., 73080., 48969.],
                         [80316., 82125., 54954.],
                         [55245., 56424., 37728.]],

                        [[98244., 100782., 67680.],
                         [110934., 113472., 76095.],
                         [76872., 78537., 52632.]],

                        [[125217., 128484., 86391.],
                         [141552., 144819., 97236.],
                         [98499., 100650., 67536.]]],

                       [[[43650., 44001., 28422.],
                         [45405., 45756., 29547.],
                         [27516., 27723., 17820.]],

                        [[125298., 126378., 83583.],
                         [130698., 131778., 87138.],
                         [85593., 86286., 57024.]],

                        [[206946., 208755., 138744.],
                         [215991., 217800., 144729.],
                         [143670., 144849., 96228.]],

                        [[288594., 291132., 193905.],
                         [301284., 303822., 202320.],
                         [201747., 203412., 135432.]],

                        [[370242., 373509., 249066.],
                         [386577., 389844., 259911.],
                         [259824., 261975., 174636.]]]]).astype(np.float32)
    assert np.allclose(output.asnumpy(), expect)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_with_bias():
    """"
    Feature: deformable_conv2d function.
    Description: Test case with bias input.
    Expectation: The results are as expected.
    """
    x = Tensor(np.arange(2 * 3 * 5 * 5).reshape(2, 3, 5, 5), mstype.float32)
    kh, kw = 3, 3
    weight = Tensor(np.arange(5 * 3 * kh * kw).reshape(5, 3, kh, kw), mstype.float32)
    bias = Tensor(np.ones((5,)), mstype.float32)
    offsets = Tensor(np.ones((2, 3 * kh * kw, 3, 3)), mstype.float32)
    output = Net()(x, weight, offsets, kh, kw, bias=bias)
    expect = np.array([[[[17326., 17677., 11548.],
                         [19081., 19432., 12673.],
                         [11992., 12199., 7921.]],

                        [[44299., 45379., 30259.],
                         [49699., 50779., 33814.],
                         [33619., 34312., 22825.]],

                        [[71272., 73081., 48970.],
                         [80317., 82126., 54955.],
                         [55246., 56425., 37729.]],

                        [[98245., 100783., 67681.],
                         [110935., 113473., 76096.],
                         [76873., 78538., 52633.]],

                        [[125218., 128485., 86392.],
                         [141553., 144820., 97237.],
                         [98500., 100651., 67537.]]],

                       [[[43651., 44002., 28423.],
                         [45406., 45757., 29548.],
                         [27517., 27724., 17821.]],

                        [[125299., 126379., 83584.],
                         [130699., 131779., 87139.],
                         [85594., 86287., 57025.]],

                        [[206947., 208756., 138745.],
                         [215992., 217801., 144730.],
                         [143671., 144850., 96229.]],

                        [[288595., 291133., 193906.],
                         [301285., 303823., 202321.],
                         [201748., 203413., 135433.]],

                        [[370243., 373510., 249067.],
                         [386578., 389845., 259912.],
                         [259825., 261976., 174637.]]]]).astype(np.float32)
    assert np.allclose(output.asnumpy(), expect)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_with_strides():
    """"
    Feature: deformable_conv2d function.
    Description: Test case with strides input.
    Expectation: The results are as expected.
    """
    x = Tensor(np.arange(2 * 3 * 5 * 5).reshape(2, 3, 5, 5), mstype.float32)
    kh, kw = 3, 3
    weight = Tensor(np.arange(5 * 3 * kh * kw).reshape(5, 3, kh, kw), mstype.float32)
    offsets = Tensor(np.ones((2, 3 * kh * kw, 2, 2)), mstype.float32)
    output = Net()(x, weight, offsets, kh, kw, (1, 1, 2, 2))
    expect = np.array([[[[17325., 11547.],
                         [11991., 7920.]],

                        [[44298., 30258.],
                         [33618., 22824.]],

                        [[71271., 48969.],
                         [55245., 37728.]],

                        [[98244., 67680.],
                         [76872., 52632.]],

                        [[125217., 86391.],
                         [98499., 67536.]]],

                       [[[43650., 28422.],
                         [27516., 17820.]],

                        [[125298., 83583.],
                         [85593., 57024.]],

                        [[206946., 138744.],
                         [143670., 96228.]],

                        [[288594., 193905.],
                         [201747., 135432.]],

                        [[370242., 249066.],
                         [259824., 174636.]]]]).astype(np.float32)
    assert np.allclose(output.asnumpy(), expect)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_with_padding():
    """"
    Feature: deformable_conv2d function.
    Description: Test case with padding input.
    Expectation: The results are as expected.
    """
    x = Tensor(np.arange(2 * 3 * 5 * 5).reshape(2, 3, 5, 5), mstype.float32)
    kh, kw = 3, 3
    weight = Tensor(np.arange(5 * 3 * kh * kw).reshape(5, 3, kh, kw), mstype.float32)
    offsets = Tensor(np.ones((2, 3 * kh * kw, 5, 7)), mstype.float32)
    output = Net()(x, weight, offsets, kh, kw, padding=(1, 1, 2, 2))
    expect = np.array([[[[10296., 15219., 15570., 15921., 10422., 5112., 0.],
                         [11511., 16974., 17325., 17676., 11547., 5652., 0.],
                         [12726., 18729., 19080., 19431., 12672., 6192., 0.],
                         [8040., 11784., 11991., 12198., 7920., 3852., 0.],
                         [3768., 5496., 5586., 5676., 3666., 1773., 0.]],

                        [[25119., 37818., 38898., 39978., 26703., 13374., 0.],
                         [28764., 43218., 44298., 45378., 30258., 15129., 0.],
                         [32409., 48618., 49698., 50778., 33813., 16884., 0.],
                         [21972., 32925., 33618., 34311., 22824., 11385., 0.],
                         [11139., 16674., 17007., 17340., 11523., 5742., 0.]],

                        [[39942., 60417., 62226., 64035., 42984., 21636., 0.],
                         [46017., 69462., 71271., 73080., 48969., 24606., 0.],
                         [52092., 78507., 80316., 82125., 54954., 27576., 0.],
                         [35904., 54066., 55245., 56424., 37728., 18918., 0.],
                         [18510., 27852., 28428., 29004., 19380., 9711., 0.]],

                        [[54765., 83016., 85554., 88092., 59265., 29898., 0.],
                         [63270., 95706., 98244., 100782., 67680., 34083., 0.],
                         [71775., 108396., 110934., 113472., 76095., 38268., 0.],
                         [49836., 75207., 76872., 78537., 52632., 26451., 0.],
                         [25881., 39030., 39849., 40668., 27237., 13680., 0.]],

                        [[69588., 105615., 108882., 112149., 75546., 38160., 0.],
                         [80523., 121950., 125217., 128484., 86391., 43560., 0.],
                         [91458., 138285., 141552., 144819., 97236., 48960., 0.],
                         [63768., 96348., 98499., 100650., 67536., 33984., 0.],
                         [33252., 50208., 51270., 52332., 35094., 17649., 0.]]],

                       [[[28521., 41544., 41895., 42246., 27297., 13212., 0.],
                         [29736., 43299., 43650., 44001., 28422., 13752., 0.],
                         [30951., 45054., 45405., 45756., 29547., 14292., 0.],
                         [18840., 27309., 27516., 27723., 17820., 8577., 0.],
                         [8493., 12246., 12336., 12426., 7941., 3798., 0.]],

                        [[79794., 118818., 119898., 120978., 80028., 39699., 0.],
                         [83439., 124218., 125298., 126378., 83583., 41454., 0.],
                         [87084., 129618., 130698., 131778., 87138., 43209., 0.],
                         [57072., 84900., 85593., 86286., 57024., 28260., 0.],
                         [28014., 41649., 41982., 42315., 27948., 13842., 0.]],

                        [[131067., 196092., 197901., 199710., 132759., 66186., 0.],
                         [137142., 205137., 206946., 208755., 138744., 69156., 0.],
                         [143217., 214182., 215991., 217800., 144729., 72126., 0.],
                         [95304., 142491., 143670., 144849., 96228., 47943., 0.],
                         [47535., 71052., 71628., 72204., 47955., 23886., 0.]],

                        [[182340., 273366., 275904., 278442., 185490., 92673., 0.],
                         [190845., 286056., 288594., 291132., 193905., 96858., 0.],
                         [199350., 298746., 301284., 303822., 202320., 101043., 0.],
                         [133536., 200082., 201747., 203412., 135432., 67626., 0.],
                         [67056., 100455., 101274., 102093., 67962., 33930., 0.]],

                        [[233613., 350640., 353907., 357174., 238221., 119160., 0.],
                         [244548., 366975., 370242., 373509., 249066., 124560., 0.],
                         [255483., 383310., 386577., 389844., 259911., 129960., 0.],
                         [171768., 257673., 259824., 261975., 174636., 87309., 0.],
                         [86577., 129858., 130920., 131982., 87969., 43974., 0.]]]]).astype(np.float32)
    assert np.allclose(output.asnumpy(), expect)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_with_dilations():
    """"
    Feature: deformable_conv2d function.
    Description: Test case with dilations input.
    Expectation: The results are as expected.
    """
    x = Tensor(np.arange(2 * 3 * 5 * 5).reshape(2, 3, 5, 5), mstype.float32)
    kh, kw = 3, 3
    weight = Tensor(np.arange(5 * 3 * kh * kw).reshape(5, 3, kh, kw), mstype.float32)
    offsets = Tensor(np.ones((2, 3 * kh * kw, 1, 1)), mstype.float32)
    output = Net()(x, weight, offsets, kh, kw, dilations=(1, 1, 2, 2))
    expect = np.array([[[[6780.]],

                        [[18768.]],

                        [[30756.]],

                        [[42744.]],

                        [[54732.]]],

                       [[[16680.]],

                        [[52968.]],

                        [[89256.]],

                        [[125544.]],

                        [[161832.]]]]).astype(np.float32)
    assert np.allclose(output.asnumpy(), expect)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_vmap():
    """"
    Feature: deformable_conv2d function.
    Description: Test case with vmap.
    Expectation: The results are as expected.
    """
    kh, kw = 3, 3

    def cal_deformable_offsets(x, offsets):
        deformable_offsets = NN.DeformableOffsets((1, 1, 1, 1), (0, 0, 0, 0), (kh, kw))
        return deformable_offsets(x, offsets)

    x = Tensor(np.arange(2 * 2 * 3 * 5 * 5).reshape(2, 2, 3, 5, 5), mstype.float32)
    offsets = Tensor(np.ones((2, 2, 3 * kh * kw, 3, 3)), mstype.float32)
    vmap_deformable_offsets = F.vmap(cal_deformable_offsets, in_axes=(0, 0), out_axes=0)
    out1 = vmap_deformable_offsets(x, offsets)

    def manually_batched(x, offsets):
        output = []
        for i in range(x.shape[0]):
            output.append(cal_deformable_offsets(x[i], offsets[i]))
        return F.stack(output)

    out2 = manually_batched(x, offsets)
    assert np.allclose(out1.asnumpy(), out2.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_deformable_conv2d_dynamic_shape():
    """"
    Feature: deformable_conv2d function.
    Description: Test case for dynamic shape support of deformable_conv2d.
    Expectation: The results are as expected.
    """
    kh, kw = 1, 1
    # x shape [1, 1, 1, 2]
    x = np.array([[[[-0.41675785, -0.05626683]]]]).astype(np.float32)
    x = Tensor(x, mstype.float32)
    # weight shape [1, 1, 1, 1]
    weight = np.array([[[[-2.1361961]]]]).astype(np.float32)
    weight = Tensor(weight, mstype.float32)
    # offsets shape [1, 3, 1, 2]
    offsets = np.array([[[[1.6402708, -1.7934356]],
                         [[-0.84174734, 0.5028814]],
                         [[-1.2452881, -1.0579522]]]]).astype(np.float32)
    offsets = Tensor(offsets, mstype.float32)
    offsets_dynamic = Tensor(shape=[None, None, None, None], dtype=mstype.float32)
    x_dynamic = Tensor(shape=[None, None, None, None], dtype=mstype.float32)
    net = Net()
    net.set_inputs(x_dynamic, weight, offsets_dynamic, kh, kw)
    out = net(x, weight, offsets, kh, kw)
    # expected output: [1, 1, 1, 2]
    expected = np.array([[[[-0.00852099, -0.09671781]]]]).astype(np.float32)
    assert np.allclose(out.asnumpy(), expected)
