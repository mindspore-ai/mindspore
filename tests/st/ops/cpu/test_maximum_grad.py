# Copyright 2022-2023 Huawei Technologies Co., Ltd
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
import mindspore.context as context
from mindspore import Tensor
from mindspore.nn import Cell
from mindspore.ops import operations as P
from mindspore.ops.functional import vmap
from tests.st.pynative.utils import GradOfAllInputs
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE, enable_graph_kernel=True, device_target="CPU")


class Maximum(Cell):
    def __init__(self):
        super(Maximum, self).__init__()
        self.max = P.Maximum()

    def construct(self, inputa, inputb):
        return self.max(inputa, inputb)


class MaxmumGradNet(Cell):
    def __init__(self):
        super(MaxmumGradNet, self).__init__()
        self.maximum_grad = GradOfAllInputs(Maximum())

    def construct(self, x, y, dy):
        return self.maximum_grad(x, y, dy)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_maximum_grad_random():
    np.random.seed(0)
    input_x = np.random.normal(0, 1, [2, 3]).astype(np.float32)
    input_y = np.random.normal(0, 1, [2, 3]).astype(np.float32)
    input_dout = np.maximum(input_x, input_y).astype(np.float32)
    net = MaxmumGradNet()
    result = net(Tensor(input_x), Tensor(input_y), Tensor(input_dout))
    dx = input_dout * (input_x >= input_y)
    dy = input_dout - dx
    assert np.allclose(result[0].asnumpy(), dx, rtol=1.e-4, atol=1.e-8, equal_nan=True)
    assert np.allclose(result[1].asnumpy(), dy, rtol=1.e-4, atol=1.e-8, equal_nan=True)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_broadcast_grad_cpu_type():
    """
    Feature: ALL To ALL
    Description: test cases for broadcast_grad of two tensors
    Expectation: the result match to numpy
    """
    np.random.seed(1)
    input_x = np.arange(2 * 3 * 2).reshape((2, 3, 2))
    input_y = np.arange(88, 2 * 3 * 2 + 88).reshape((2, 3, 2))
    input_dout = np.maximum(input_x, input_y)
    net = MaxmumGradNet()
    dtypes = (np.int16, np.int32, np.int64, np.float16,
              np.float32, np.float64, np.uint16, np.uint32)
    for dtype in dtypes:
        result = net(Tensor(input_x.astype(dtype)), Tensor(input_y.astype(dtype)),
                     Tensor(input_dout.astype(dtype)))
        dx = input_dout * (input_x >= input_y)
        dy = input_dout - dx
        assert np.allclose(result[0].asnumpy(), dx, rtol=1.e-4, atol=1.e-8, equal_nan=True)
        assert np.allclose(result[1].asnumpy(), dy, rtol=1.e-4, atol=1.e-8, equal_nan=True)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_broadcast_grad_cpu():
    x = np.array([[[[0.659578],
                    [0.49113268],
                    [0.75909054],
                    [0.71681815],
                    [0.30421826]]],
                  [[[0.30322495],
                    [0.02858258],
                    [0.06398096],
                    [0.09519596],
                    [0.12498625]]],
                  [[[0.7347768],
                    [0.166469],
                    [0.328553],
                    [0.54908437],
                    [0.23673844]]]]).astype(np.float32)
    y = np.array([[[[0.9154968, 0.29014662, 0.6492294, 0.39918253, 0.1648203, 0.00861965]],
                   [[0.996885, 0.24152198, 0.3601213, 0.51664376, 0.7933056, 0.84706444]],
                   [[0.75606346, 0.974512, 0.3939527, 0.69697475, 0.83400667, 0.6348955]],
                   [[0.68492866, 0.24609096, 0.4924665, 0.22500521, 0.38474053, 0.5586104]]]]).astype(np.float32)
    dout = np.array([[[[0.42891738, 0.03434946, 0.06192983, 0.21216309, 0.37450036, 0.6619524],
                       [0.8583447, 0.5765161, 0.1468952, 0.9975385, 0.6908136, 0.4903796],
                       [0.68952006, 0.39336833, 0.9049695, 0.66886294, 0.2338471, 0.913618],
                       [0.0428149, 0.6243054, 0.8519898, 0.12088962, 0.9735885, 0.45661286],
                       [0.41563734, 0.41607043, 0.4754915, 0.32207987, 0.33823156, 0.47422352]],

                      [[0.64478457, 0.22430937, 0.7682554, 0.46082005, 0.8938723, 0.20490853],
                       [0.44393885, 0.08278944, 0.4734108, 0.5543551, 0.39428464, 0.44424313],
                       [0.12612297, 0.76566416, 0.71133816, 0.81280327, 0.20583127, 0.54058075],
                       [0.41341263, 0.48118508, 0.00401995, 0.37259838, 0.05435474, 0.5240658],
                       [0.4081956, 0.48718935, 0.9132831, 0.67969185, 0.0119757, 0.8328054]],

                      [[0.91695577, 0.95370644, 0.263782, 0.7477626, 0.6448147, 0.8080634],
                       [0.15576603, 0.9104615, 0.3778708, 0.6912833, 0.2092224, 0.67462957],
                       [0.7087075, 0.7888326, 0.4672294, 0.98221505, 0.25210258, 0.98920417],
                       [0.7466197, 0.22702982, 0.01991269, 0.6846591, 0.7515228, 0.5890395],
                       [0.04531088, 0.21740614, 0.8406235, 0.36480767, 0.37733936, 0.02914464]],

                      [[0.33069974, 0.5497569, 0.9896345, 0.4167176, 0.78057563, 0.04659131],
                       [0.7747768, 0.21427679, 0.29893255, 0.7706969, 0.9755185, 0.42388415],
                       [0.3910244, 0.39381978, 0.37065396, 0.15558061, 0.05012341, 0.15870963],
                       [0.17791101, 0.47219893, 0.13899496, 0.32323205, 0.3628809, 0.02580585],
                       [0.30274773, 0.62890774, 0.11024303, 0.6980051, 0.35346958, 0.062852]]],

                     [[[0.6925081, 0.74668753, 0.80145043, 0.06598313, 0.665123, 0.15073007],
                       [0.11784806, 0.6385372, 0.5228278, 0.5349848, 0.84671104, 0.8096436],
                       [0.09516156, 0.63298017, 0.52382874, 0.36734378, 0.66497755, 0.6019127],
                       [0.46438488, 0.0194377, 0.9388292, 0.7286089, 0.29178405, 0.11872514],
                       [0.22101837, 0.6164887, 0.6139798, 0.11711904, 0.6227745, 0.09701069]],

                      [[0.80480653, 0.90034056, 0.8633447, 0.97415197, 0.08309154, 0.8446033],
                       [0.9473769, 0.791024, 0.26339203, 0.01155075, 0.2673186, 0.7116369],
                       [0.9687511, 0.24281934, 0.37777108, 0.09802654, 0.2421312, 0.87095344],
                       [0.6311381, 0.23368953, 0.0998995, 0.4364419, 0.9187446, 0.5043872],
                       [0.35226053, 0.09357589, 0.41317305, 0.85930043, 0.16249318, 0.5478765]],

                      [[0.14338651, 0.24859418, 0.4246941, 0.73034066, 0.47172204, 0.8717199],
                       [0.05415315, 0.78556925, 0.99214983, 0.7415298, 0.673708, 0.87817156],
                       [0.616975, 0.42843062, 0.05179814, 0.1566958, 0.04536059, 0.70166487],
                       [0.15493333, 0.776598, 0.4361967, 0.40253627, 0.89210516, 0.8144414],
                       [0.04816005, 0.29696834, 0.4586605, 0.3419852, 0.5595613, 0.74093205]],

                      [[0.1388035, 0.9168704, 0.64287645, 0.83864623, 0.48026922, 0.78323376],
                       [0.12724937, 0.83034366, 0.42557436, 0.50578654, 0.25630295, 0.15349793],
                       [0.27256685, 0.04547984, 0.5385756, 0.39270344, 0.7661698, 0.23722854],
                       [0.24620503, 0.25431684, 0.71564585, 0.01161419, 0.846467, 0.7043044],
                       [0.63272387, 0.11857849, 0.3772076, 0.16758402, 0.46743023, 0.05919575]]],

                     [[[0.18827082, 0.8912264, 0.6841404, 0.74436826, 0.9582085, 0.1083683],
                       [0.60695344, 0.09742349, 0.25074378, 0.87940735, 0.21116392, 0.39418384],
                       [0.744686, 0.35679692, 0.01308284, 0.45166633, 0.68166, 0.8634658],
                       [0.7331758, 0.21113694, 0.3935488, 0.87934476, 0.70728546, 0.09309767],
                       [0.12128611, 0.93696386, 0.81177396, 0.85402405, 0.5827289, 0.9776509]],

                      [[0.54069614, 0.66651285, 0.10646132, 0.17342485, 0.88795924, 0.03551182],
                       [0.25531697, 0.87946486, 0.74267226, 0.89230734, 0.95171434, 0.94697934],
                       [0.3708397, 0.507355, 0.97099817, 0.4918163, 0.17212386, 0.5008048],
                       [0.62530744, 0.25210327, 0.73966664, 0.71555346, 0.82484317, 0.6094874],
                       [0.4589691, 0.1386695, 0.27448782, 0.20373994, 0.27805242, 0.23292768]],

                      [[0.7414099, 0.2270226, 0.90431255, 0.47035843, 0.9581062, 0.5359226],
                       [0.79603523, 0.45549425, 0.80858237, 0.7705133, 0.017761, 0.98001194],
                       [0.06013146, 0.99240226, 0.33515573, 0.04110833, 0.41470334, 0.7130743],
                       [0.5687417, 0.5788611, 0.00722461, 0.6603336, 0.3420471, 0.75181854],
                       [0.4699261, 0.51390815, 0.343182, 0.81498754, 0.8942413, 0.46532857]],

                      [[0.4589523, 0.5534698, 0.2825786, 0.8205943, 0.78258514, 0.43154418],
                       [0.27020997, 0.01667354, 0.60871965, 0.90670526, 0.3208025, 0.96995634],
                       [0.85337156, 0.9711295, 0.1381724, 0.53670496, 0.7347996, 0.73380876],
                       [0.6137464, 0.54751194, 0.9037335, 0.23134394, 0.61411524, 0.26583543],
                       [0.70770144, 0.01813207, 0.24718016, 0.70329237, 0.7062925, 0.14399007]]]]).astype(np.float32)

    dx = np.array([[[[6.6534014],
                     [5.649811],
                     [10.071739],
                     [6.6798244],
                     [3.0426278]]],
                   [[[4.2183976],
                     [0.8096436],
                     [0.6019127],
                     [0.11872514],
                     [0.09701069]]],
                   [[[9.573029],
                     [0.60534775],
                     [3.917112],
                     [5.9021177],
                     [2.263672]]]]).astype(np.float32)

    dy = np.array([[[[6.4205275, 2.941831, 5.492452, 4.3212175, 2.4262471, 0.]],
                    [[7.991917, 2.3792431, 4.9190216, 5.2013817, 6.348791, 8.351772]],
                    [[5.518505, 8.401285, 4.691043, 6.463884, 7.504318, 7.620938]],
                    [[5.2708025, 1.2835244, 4.1031275, 1.9843934, 4.9320035, 4.537787]]]]).astype(np.float32)

    net = MaxmumGradNet()
    result = net(Tensor(x), Tensor(y), Tensor(dout))
    assert np.allclose(result[0].asnumpy(), dx, rtol=1.e-4, atol=1.e-8, equal_nan=True)
    assert np.allclose(result[1].asnumpy(), dy, rtol=1.e-4, atol=1.e-8, equal_nan=True)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_max_tensor_grad_with_same_input():
    """
    Feature: test maximumgrad on CPU
    Description: test maximumgrad with same input.
    Expectation: result match to expected result.
    """
    x_np = np.array([1.7, 2.3, 5.8]).astype(np.float32)
    y_np = np.array([1.7, 2.3, 5.8]).astype(np.float32)
    dout = np.array([1.0, -1.0, 0]).astype(np.float32)
    net = MaxmumGradNet()
    output = net(Tensor(x_np), Tensor(y_np), Tensor(dout))
    print(output[0].asnumpy())
    print(output[1].asnumpy())
    expect0 = np.array([0.5, -0.5, 0.])
    expect1 = np.array([0.5, -0.5, 0.])
    assert np.allclose(output[0].asnumpy(), expect0, rtol=1e-6, atol=1e-4)
    assert np.allclose(output[1].asnumpy(), expect1, rtol=1e-6, atol=1e-4)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_max_tensor_grad_with_input_nan():
    """
    Feature: test maximumgrad on CPU
    Description: test maximumgrad with input nan.
    Expectation: result match to expected result.
    """
    x_np = np.full((3,), np.nan).astype(np.float32)
    y_np = np.array([1.7, 2.3, 5.8]).astype(np.float32)
    dout = np.array([1.28, -0.23, 0.96]).astype(np.float32)
    net = MaxmumGradNet()
    output = net(Tensor(x_np), Tensor(y_np), Tensor(dout))
    print(output[0].asnumpy())
    print(output[1].asnumpy())
    expect0 = np.array([1.28, -0.23, 0.96])
    expect1 = np.array([1.28, -0.23, 0.96])
    assert np.allclose(output[0].asnumpy(), expect0, rtol=1e-6, atol=1e-4)
    assert np.allclose(output[1].asnumpy(), expect1, rtol=1e-6, atol=1e-4)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_max_grad_vmap():
    """
    Feature: maximumgrad vmap
    Description: test the maximumgrad vmap when in_axes=(0, 0, 0).
    Expectation: match to np benchmark.
    """
    net = MaxmumGradNet()
    vmap_net = vmap(net, in_axes=(0, 0, 0))
    np.random.seed(20)
    x = Tensor(np.random.rand(2, 4))
    y = Tensor(np.random.rand(2, 4))
    sens = Tensor(np.random.rand(2, 4))
    dx, dy = vmap_net(x, y, sens)

    expect0 = np.array([[0., 0.75128073, 0.23921822, 0.25480601],
                        [0., 0., 0., 0.17878053]])
    expect1 = np.array([[0.11669374, 0., 0., 0.],
                        [0.85762554, 0.94977903, 0.56168687, 0.]])

    assert np.allclose(dx.asnumpy(), expect0, rtol=1e-6, atol=1e-4)
    assert np.allclose(dy.asnumpy(), expect1, rtol=1e-6, atol=1e-4)
