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

import pytest
import numpy as np
import mindspore
from mindspore import context
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore import nn


class RandomPoissonTEST(nn.Cell):
    def __init__(self, seed, seed2) -> None:
        super(RandomPoissonTEST, self).__init__()
        self.random_poisson = P.random_ops.RandomPoisson(seed, seed2)

    def construct(self, shape, rate):
        return self.random_poisson(shape, rate)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_random_poisson_op_case1():
    """
    Feature: Random poisson gpu kernel
    Description: test the correctness of shape and result
    Expectation: match to tensorflow benchmark.
    """
    random_poisson_test = RandomPoissonTEST(seed=10, seed2=20)
    shape = Tensor(np.array([1, 4]), mindspore.int32)
    expect_shape = np.array([1, 4, 2])
    rate_type_list = [mindspore.int32, mindspore.int64, mindspore.float16, mindspore.float32, mindspore.float64]
    for rate_type in rate_type_list:
        rate = Tensor(np.array([5, 10]), rate_type)
        expect_result = Tensor(np.array([[[7, 8], [4, 10], [2, 14], [5, 13]]]), rate_type)

        context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
        output = random_poisson_test(shape, rate)
        assert np.all(output.shape == expect_shape)
        assert np.all(output.asnumpy() == expect_result.asnumpy())

        context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
        output = random_poisson_test(shape, rate)
        assert np.all(output.shape == expect_shape)
        assert np.all(output.asnumpy() == expect_result.asnumpy())


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_random_poisson_op_case2():
    """
    Feature: Random poisson gpu kernel
    Description: test the correctness of shape and result
    Expectation: match to tensorflow benchmark.
    """
    random_poisson_test = RandomPoissonTEST(seed=10, seed2=20)
    shape = Tensor(np.array([1, 2, 3]), mindspore.int64)
    expect_shape = np.array([1, 2, 3, 3])
    rate_type_list = [mindspore.int32, mindspore.int64, mindspore.float16, mindspore.float32, mindspore.float64]
    for rate_type in rate_type_list:
        rate = Tensor(np.array([5, 10, 15]), rate_type)
        expect_result = Tensor(np.array([[[[7, 8, 12],
                                           [5, 6, 20],
                                           [5, 13, 17]],
                                          [[5, 12, 21],
                                           [6, 4, 10],
                                           [10, 11, 4]]]]), rate_type)

        context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
        output = random_poisson_test(shape, rate)
        assert np.all(output.shape == expect_shape)
        assert np.all(output.asnumpy() == expect_result.asnumpy())

        context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
        output = random_poisson_test(shape, rate)
        assert np.all(output.shape == expect_shape)
        assert np.all(output.asnumpy() == expect_result.asnumpy())
