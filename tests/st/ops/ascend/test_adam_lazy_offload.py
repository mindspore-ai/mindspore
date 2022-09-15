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
from mindspore import Tensor, Parameter
from mindspore.nn import Dense
from mindspore.nn import TrainOneStepCell, WithLossCell
from mindspore.nn.optim import Adam
from mindspore.ops import operations as P


class NetAdam(nn.Cell):
    def __init__(self):
        super(NetAdam, self).__init__()
        self.batch_size = 1
        self.reshape = P.Reshape()
        weight = Tensor(np.ones([10, 16]).astype(np.float32) * 0.01)
        self.fc1 = Dense(16, 10, weight_init=weight)

    def construct(self, input_x):
        output = self.reshape(input_x, (self.batch_size, -1))
        output = self.fc1(output)
        return output


class NetWithSparseGatherV2(nn.Cell):
    def __init__(self):
        super(NetWithSparseGatherV2, self).__init__()
        self.weight1 = Parameter(Tensor(np.ones([3, 1, 2]).astype(np.float32)), name="weight1")
        self.weight2 = Parameter(Tensor(np.ones([3, 1, 2]).astype(np.float32)), name="weight2")
        self.axis = 0
        self.gather = P.SparseGatherV2()

    def construct(self, indices, label):
        return self.gather(self.weight1, indices, self.axis) + self.weight2


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_lazy_adam_converge():
    """
    Feature: LazyAdam optimizer
    Description: Verify if the loss is converged
    Expectation: success
    """
    context.set_context(mode=context.GRAPH_MODE)
    epoch = 3
    net = NetAdam()
    optimizer = Adam(filter(lambda x: x.requires_grad,
                            net.get_parameters()), learning_rate=0.01, use_lazy=True)
    criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    net_with_criterion = WithLossCell(net, criterion)
    train_network = TrainOneStepCell(net_with_criterion, optimizer)
    train_network.set_train()

    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    losses1 = []
    for _ in range(epoch):
        data = Tensor(np.arange(0, 16).reshape((1, 1, 4, 4)).astype(np.float32) * 0.01)
        label = Tensor(np.array([0]).astype(np.int32))
        loss = train_network(data, label)
        losses1.append(loss.asnumpy())
    assert losses1[0] > losses1[1]
    assert losses1[1] > losses1[2]


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_lazy_adam_acc():
    """
    Feature: LazyAdam optimizer
    Description: Verify if the result is correct
    Expectation: success
    """
    context.set_context(mode=context.GRAPH_MODE)
    indices = Tensor(np.array([0, 0, 1]).astype(np.int32))
    label = Tensor(np.zeros([2, 1, 2]).astype(np.float32))
    net = NetWithSparseGatherV2()

    output = []
    optimizer = Adam(net.trainable_params(), learning_rate=0.1, use_lazy=True)
    optimizer.target = 'Ascend'
    for _ in range(2):
        train_network = TrainOneStepCell(net, optimizer)
        output = train_network(indices, label)
    expected_output = np.array([[[1.8, 1.8]], [[1.8, 1.8]], [[1.8000001, 1.8000001]]]).astype(np.float32)
    assert np.allclose(output.asnumpy(), expected_output)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_adam_offload_acc():
    """
    Feature: AdamOffload optimizer
    Description: Verify if the loss is the same as the original AdamOffload
    Expectation: success
    """
    context.set_context(mode=context.GRAPH_MODE)
    epoch = 3
    net = NetAdam()
    optimizer = Adam(filter(lambda x: x.requires_grad, net.get_parameters()), learning_rate=0.01, use_offload=True)
    criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    net_with_criterion = WithLossCell(net, criterion)
    train_network = TrainOneStepCell(net_with_criterion, optimizer)
    train_network.set_train()

    context.set_context(mode=context.GRAPH_MODE)
    losses1 = []
    for _ in range(epoch):
        data = Tensor(np.arange(0, 16).reshape((1, 1, 4, 4)).astype(np.float32) * 0.01)
        label = Tensor(np.array([0]).astype(np.int32))
        loss = train_network(data, label)
        losses1.append(loss.asnumpy())

    assert np.array_equal(losses1[-1], np.array(2.2237475, np.float32))
