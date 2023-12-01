from __future__ import absolute_import
import pytest
import numpy as np
import mindspore
from mindspore import nn
from mindspore import Tensor, context
from mindspore.experimental.optim import RAdam
from mindspore.experimental.optim.lr_scheduler import StepLR


class Network(nn.Cell):
    def __init__(self, lin_weight, lin_bias):
        super().__init__()
        self.lin = nn.Dense(2, 3, weight_init=lin_weight, bias_init=lin_bias)
        self.relu = nn.ReLU()

    def construct(self, x):
        out = self.lin(x)
        out = self.relu(out)
        return out


class RAdamFactory():
    def __init__(self, group=True, lr_dynamic=False, if_change=False, dtype=np.float32):
        super().__init__()
        self.lin_weight_np = np.array([[0.15, 0.18], [0.2, 0.5], [0.19, 0.17]]).astype(dtype)
        self.lin_bias_np = np.array([0.1, 0.3, 0.2]).astype(dtype)

        self.group = group
        self.lr_dynamic = lr_dynamic
        self.if_change = if_change
        self.data = np.array([[0.1, 0.1], [0.2, 0.2]]).astype(np.float32)
        self.label = np.array([[0.1, 0.1, 0.8], [0.75, 0.24, 0.88]]).astype(np.float32)
        self.epochs = 3
        self.steps = 1
        self.lr = 0.002

        self.betas = (0.9, 0.999)
        self.eps = 1e-8
        self.weight_decay = 0
        self.amsgrad = False
        self.maximize = False

    def forward_mindspore_impl(self):
        lin_weight = Tensor(self.lin_weight_np.copy())
        lin_bias = Tensor(self.lin_bias_np.copy())
        model = Network(lin_weight, lin_bias)

        data = Tensor(self.data)
        label = Tensor(self.label)

        if not self.group:
            optimizer = RAdam(model.trainable_params(), self.lr)
        else:
            bias_params = list(filter(lambda x: 'bias' in x.name, model.trainable_params()))
            no_bias_params = list(filter(lambda x: 'bias' not in x.name, model.trainable_params()))
            group_params = [{'params': bias_params, 'weight_decay': 0.1, 'betas': (0.8, 0.88)},
                            {'params': no_bias_params, 'lr': 0.66, "eps": 1e5}]
            optimizer = RAdam(params=group_params, lr=self.lr)

        criterion = nn.MAELoss(reduction="mean")
        lr_scheduler = StepLR(optimizer, 2, gamma=0.5, last_epoch=-1)

        def forward_fn(data, label):
            logits = model(data)
            loss = criterion(logits, label)
            return loss, logits

        grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

        def train_step(data, label):
            (loss, _), grads = grad_fn(data, label)
            optimizer(grads)
            return loss

        def train(epochs, steps, lr_dynamic, if_change):
            for _ in range(epochs):
                for _ in range(steps):
                    train_step(data, label)
                if lr_dynamic:
                    lr_scheduler.step()
                if if_change:
                    optimizer.param_groups[1]["betas"] = (0.7, 0.87)

        train(self.epochs, self.steps, self.lr_dynamic, self.if_change)
        output = model(data)
        return output.asnumpy()


def _count_unequal_element(data_expected, data_me, rtol, atol):
    assert data_expected.shape == data_me.shape
    total_count = len(data_expected.flatten())
    error = np.abs(data_expected - data_me)
    greater = np.greater(error, atol + np.abs(data_me) * rtol)
    loss_count = np.count_nonzero(greater)
    assert (loss_count / total_count) < rtol, \
        "\ndata_expected_std:{0}\ndata_me_error:{1}\nloss:{2}". \
            format(data_expected[greater], data_me[greater], error[greater])


def allclose_nparray(data_expected, data_me, rtol, atol, equal_nan=True):
    if np.any(np.isnan(data_expected)) or np.any(np.isnan(data_me)):
        assert np.allclose(data_expected, data_me, rtol, atol, equal_nan=equal_nan)
    elif not np.allclose(data_expected, data_me, rtol, atol, equal_nan=equal_nan):
        _count_unequal_element(data_expected, data_me, rtol, atol)
    else:
        assert np.array(data_expected).shape == np.array(data_me).shape


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_radam_basic(mode):
    """
    Feature: Test radam.
    Description: Test radam with default parameter.
    Expectation: success.
    """
    mindspore.set_context(mode=mode, jit_syntax_level=mindspore.STRICT)
    fact = RAdamFactory(False, False)
    out = fact.forward_mindspore_impl()
    loss_expect = [[0.13302001, 0.36793998, 0.23806], [0.16604, 0.43787998, 0.27412]]
    allclose_nparray(loss_expect, out, 0.005, 0.005)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.PYNATIVE_MODE])
def test_radam_group(mode):
    """
    Feature: Test radam.
    Description: Test radam with grouped params.
    Expectation: success.
    """
    mindspore.set_context(mode=mode, jit_syntax_level=mindspore.STRICT)
    fact = RAdamFactory(True, False)
    out = fact.forward_mindspore_impl()
    loss_expect = [[0.13954002, 0.3480203, 0.2576798], [0.17914003, 0.39822027, 0.31347978]]
    allclose_nparray(loss_expect, out, 0.005, 0.005)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_radam_lr_dynamic(mode):
    """
    Feature: Test radam.
    Description: Test radam when lr is dynamic.
    Expectation: success.
    """
    mindspore.set_context(mode=mode, jit_syntax_level=mindspore.STRICT)
    fact = RAdamFactory(False, True)
    out = fact.forward_mindspore_impl()
    loss_expect = [[0.13301668, 0.36828333, 0.23771667], [0.16603333, 0.43823332, 0.27376667]]
    allclose_nparray(loss_expect, out, 0.005, 0.005)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_radam_group_lr_dynamic(mode):
    """
    Feature: Test radam.
    Description: Test radam with grouped params when lr is dynamic.
    Expectation: success.
    """
    mindspore.set_context(mode=mode, jit_syntax_level=mindspore.STRICT)
    fact = RAdamFactory(True, True)
    out = fact.forward_mindspore_impl()
    loss_expect = [[0.13845001, 0.3516835, 0.25406653], [0.17695001, 0.4051835, 0.30656654]]
    allclose_nparray(loss_expect, out, 0.005, 0.005)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.PYNATIVE_MODE])
def test_radam_group_lr_dynamic_change_param(mode):
    """
    Feature: Test radam.
    Description: Test radam with grouped params when optimizer params are changed.
    Expectation: success.
    """
    mindspore.set_context(mode=mode, jit_syntax_level=mindspore.STRICT)
    fact = RAdamFactory(True, True, True)
    out = fact.forward_mindspore_impl()
    loss_expect = [[0.13768202, 0.35398751, 0.25176254], [0.17541403, 0.4097915, 0.30195853]]
    allclose_nparray(loss_expect, out, 0.005, 0.005)
