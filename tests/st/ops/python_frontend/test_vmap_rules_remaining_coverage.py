# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Remaining vmap contracts: real CPU numerics and explicit executor protocols.

The graph preprocess and trailing monad are adapted at their eager boundaries.
The original rule closures, tensor transformations and numerical kernels execute.
Missing primitive metadata APIs and malformed product calls are not fabricated.
"""
import importlib
import random

import numpy as np
import pytest
import mindspore as ms
from mindspore import dataset
from mindspore._c_expression import UMonad, FormatEnum, ReductionEnum
from mindspore.ops import operations as P, auto_generate as Gen
from mindspore.ops.operations import nn_ops as NN, _grad_ops as G, image_ops as IMG
from mindspore.ops.primitive import Primitive
from mindspore.common.generator import default_generator
from mindspore import log as ms_log
from mindspore.ops._vmap import vmap_base as B, vmap_nn_ops as N, vmap_grad_nn_ops as D
from mindspore.ops._vmap import vmap_array_ops as V, vmap_math_ops as W
from mindspore.ops._vmap import vmap_other_ops as O, vmap_debug_ops as Debug
from mindspore.ops._vmap import vmap_image_ops as I, vmap_random_ops as R
from mindspore.ops._vmap import vmap_convolution_ops as Conv
from mindspore.ops._vmap import vmap_grad_math_ops as GradMath
from tests.mark_utils import arg_mark


def tensor(value, dtype=np.float32):
    return ms.Tensor(np.asarray(value, dtype=dtype))


def assert_tensor(actual, expected, dtype=ms.float32):
    expected = np.asarray(expected)
    assert isinstance(actual, ms.Tensor)
    assert actual.dtype == dtype and actual.shape == expected.shape
    np.testing.assert_allclose(actual.asnumpy(), expected, rtol=3e-5, atol=2e-6)


def eager_preprocess(prim, *arguments):
    """Follow the C++ bdim branch order; invoke the first argument only for all-unmapped inputs."""
    if isinstance(prim, str):
        prim = Primitive(prim)

    def leaf(value):
        return isinstance(value, tuple) and len(value) == 2 and (value[1] is None or isinstance(value[1], int))

    def unmapped(value):
        return value[1] is None if leaf(value) else all(unmapped(item) for item in value)

    def unbind(value):
        return value[0] if leaf(value) else tuple(unbind(item) for item in value)

    if getattr(prim, 'name', None) in ('MirrorPad', 'MirrorPadGrad', 'PadV3') and len(arguments) == 1:
        arguments = arguments[0]
    if all(unmapped(value) for value in arguments):
        # The real C++ metagraph only uses its first argument in this all-unmapped branch.
        if not callable(prim):
            raise TypeError('all-unmapped preprocessing requires a callable primitive')
        return True, B.vmap_bind_all_none(prim(*(unbind(value) for value in arguments)))
    return False, None


@pytest.fixture(autouse=True)
def isolated_cpu(monkeypatch):
    # Snapshot private RNG and logging state so native execution stays isolated.
    # pylint: disable=protected-access
    previous = {key: ms.get_context(key) for key in ('mode', 'device_target', 'pynative_synchronize')}
    python_state, numpy_state = random.getstate(), np.random.get_state()
    dataset_seed, state, global_seed = dataset.config.get_seed(), ms.get_rng_state(), ms.get_seed()
    manual = default_generator._generator.attrs['manual_seed']
    seed_module = importlib.import_module('mindspore.common.seed')
    op_seeds = seed_module._KERNEL_SEED.copy()
    monkeypatch.setattr(ms_log._LogActionOnce, 'is_logged', ms_log._LogActionOnce.is_logged.copy())
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target='CPU', pynative_synchronize=True)
    for module in (B, N, D, V, W, I, R, Conv, GradMath):
        monkeypatch.setattr(module, 'vmap_general_preprocess', eager_preprocess)
    yield
    ms.set_rng_state(state)
    default_generator._generator.add_prim_attr('manual_seed', manual)
    dataset.config.set_seed(dataset_seed)
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    seed_module._GLOBAL_SEED = global_seed
    seed_module._KERNEL_SEED.clear()
    seed_module._KERNEL_SEED.update(op_seeds)
    ms.set_context(**previous)


# Explicit native argument order, mutable slots, and public rule result slots.
OPTIMIZERS = [
    ('ada_max', 'ApplyAdaMax', 'var m v beta1_power lr beta1 beta2 epsilon grad', 'var m v', 'var m v'),
    ('adadelta', 'ApplyAdadelta', 'var accum accum_update lr rho epsilon grad',
     'var accum accum_update', 'var accum accum_update'),
    ('ftrl', 'ApplyFtrl', 'var accum linear grad lr l1 l2 lr_power', 'var accum linear', 'var'),
    ('proximal_adagrad', 'ApplyProximalAdagrad', 'var accum lr l1 l2 grad', 'var accum', 'var accum'),
    ('gradient_descent', 'ApplyGradientDescent', 'var alpha delta', 'var', 'var'),
    ('proximal_gradient_descent', 'ApplyProximalGradientDescent', 'var alpha l1 l2 delta', 'var', 'var'),
    ('adam_with_amsgrad', 'ApplyAdamWithAmsgrad', 'var m v vhat beta1_power beta2_power lr grad',
     'var m v vhat', 'var m v vhat'),
    ('adam_with_amsgrad_v2', 'ApplyAdamWithAmsgradV2',
     'var m v vhat beta1_power beta2_power lr beta1 beta2 epsilon grad', 'var m v vhat', 'var m v vhat'),
    ('adam', 'Adam', 'var m v beta1_power beta2_power lr beta1 beta2 epsilon grad', 'var m v', 'var m v'),
    ('power_sign', 'ApplyPowerSign', 'var m lr logbase sign_decay beta grad', 'var m', 'var m'),
    ('adagrad_v2', 'ApplyAdagradV2', 'var accum lr grad', 'var accum', 'var accum'),
    ('adagrad_da', 'ApplyAdagradDA', 'var gradient_accumulator gradient_squared_accumulator grad lr l1 l2 global_step',
     'var gradient_accumulator gradient_squared_accumulator', 'var'),
    ('rmsprop', 'ApplyRMSProp', 'var mean_square moment lr grad decay momentum epsilon',
     'var mean_square moment', 'var'),
    ('centered_rmsprop', 'ApplyCenteredRMSProp',
     'var mean_grad mean_square mom grad lr rho momentum eps', 'var mean_grad mean_square mom', 'var'),
    ('sparse_adagrad', 'SparseApplyAdagrad', 'var accum grad indices', 'var accum', 'var accum'),
    ('sparse_ftrl', 'SparseApplyFtrl', 'var accum linear grad indices', 'var accum linear', 'var accum linear'),
]


def optimizer_primitive(name):
    if name == 'ApplyAdagradV2':
        return P.ApplyAdagradV2(.02)
    if name == 'ApplyAdamWithAmsgrad':
        return P.ApplyAdamWithAmsgrad(.8, .7, .02)
    if name == 'SparseApplyAdagrad':
        return NN.SparseApplyAdagrad(.15)
    if name == 'SparseApplyFtrl':
        return NN.SparseApplyFtrl(.15, .03, .02, -.5)
    return getattr(P, name)()


def optimizer_rule(kind, prim):
    factories = {
        'ada_max': N.get_apply_ada_max_rule, 'adadelta': N.get_apply_adadelta_rule,
        'ftrl': N.get_apply_ftrl_rule, 'proximal_adagrad': N.get_apply_proximal_adagrad_rule,
        'gradient_descent': N.get_apply_gradient_descent_rule,
        'proximal_gradient_descent': N.get_apply_proximal_gradient_descent_rule,
        'adam_with_amsgrad': N.get_apply_adam_with_amsgrad_rule,
        'adam_with_amsgrad_v2': N.get_apply_adam_with_amsgrad_v2_rule, 'adam': N.get_adam_rule,
        'power_sign': N.get_apply_power_sign_rule, 'adagrad_v2': N.get_apply_adagrad_v2_vmap_rule,
        'adagrad_da': N.get_apply_adagrad_da_vmap_rule, 'rmsprop': N.get_rmsprop_vmap_rule,
        'centered_rmsprop': N.get_apply_centered_rmsprop_vmap_rule,
        'sparse_adagrad': N.get_sparse_apply_adagrad_vmap_rule, 'sparse_ftrl': N.get_sparse_apply_ftrl_vmap_rule}
    return factories[kind](prim, 2)


def numpy_optimizer_step(kind, values):
    """One independent scalar/vector step from the documented optimization equations."""
    s = {key: np.asarray(value).copy() for key, value in values.items()}
    x = s['var'].copy()
    g = s.get('grad', s.get('delta'))
    if kind in ('ftrl', 'sparse_ftrl'):
        lr, l1, l2, power = (s['lr'], s['l1'], s['l2'], s['lr_power']) if kind == 'ftrl' else (.15, .03, .02, -.5)
        old = s['accum'].copy()
        s['accum'] += g * g
        s['linear'] += g - (s['accum'] ** -power - old ** -power) / lr * x
        s['var'] = np.where(np.abs(s['linear']) > l1,
                            (np.sign(s['linear']) * l1 - s['linear']) / (s['accum'] ** -power / lr + 2*l2), 0)
    elif kind in ('proximal_adagrad', 'adagrad_v2', 'sparse_adagrad'):
        s['accum'] += g*g
        lr = .15 if kind == 'sparse_adagrad' else s['lr']
        rate = lr / (np.sqrt(s['accum']) + (.02 if kind == 'adagrad_v2' else 0))
        prox = x - rate*g
        s['var'] = (np.sign(prox)*np.maximum(np.abs(prox)-rate*s['l1'], 0)/(1+rate*s['l2'])
                    if kind == 'proximal_adagrad' else prox)
    elif kind == 'gradient_descent':
        s['var'] = x - s['alpha']*g
    elif kind == 'proximal_gradient_descent':
        prox = x-s['alpha']*g
        s['var'] = np.sign(prox)*np.maximum(np.abs(prox)-s['alpha']*s['l1'], 0)/(1+s['alpha']*s['l2'])
    elif kind == 'ada_max':
        s['m'] = s['beta1']*s['m']+(1-s['beta1'])*g
        s['v'] = np.maximum(s['beta2']*s['v'], np.abs(g))
        s['var'] = x-s['lr']/(1-s['beta1_power'])*s['m']/(s['v']+s['epsilon'])
    elif kind == 'adadelta':
        s['accum'] = s['rho']*s['accum']+(1-s['rho'])*g*g
        update = np.sqrt(s['accum_update']+s['epsilon'])/np.sqrt(s['accum']+s['epsilon'])*g
        s['accum_update'] = s['rho']*s['accum_update']+(1-s['rho'])*update*update
        s['var'] = x-s['lr']*update
    elif kind.startswith('adam'):
        b1, b2, epsilon = (.8, .7, .02) if kind == 'adam_with_amsgrad' else (s['beta1'], s['beta2'], s['epsilon'])
        s['m'] = b1*s['m']+(1-b1)*g
        s['v'] = b2*s['v']+(1-b2)*g*g
        variance = s['v']
        if 'amsgrad' in kind:
            s['vhat'] = np.maximum(s['vhat'], s['v'])
            variance = s['vhat']
        rate = s['lr']*np.sqrt(1-s['beta2_power'])/(1-s['beta1_power'])
        s['var'] = x-rate*s['m']/(np.sqrt(variance)+epsilon)
    elif kind == 'power_sign':
        s['m'] = s['beta']*s['m']+(1-s['beta'])*g
        s['var'] = x-s['lr']*np.exp(s['logbase']*s['sign_decay']*np.sign(g)*np.sign(s['m']))*g
    elif kind == 'adagrad_da':
        s['gradient_accumulator'] += g
        s['gradient_squared_accumulator'] += g*g
        prox = np.sign(s['gradient_accumulator'])* \
                       np.maximum(np.abs(s['gradient_accumulator'])-s['l1']*s['global_step'], 0)
        s['var'] = -s['lr']*prox/(s['l2']*s['global_step']*s['lr']+np.sqrt(s['gradient_squared_accumulator']))
    elif kind in ('rmsprop', 'centered_rmsprop'):
        rho = s['decay'] if kind == 'rmsprop' else s['rho']
        s['mean_square'] = rho*s['mean_square']+(1-rho)*g*g
        variance = s['mean_square']
        slot = 'moment' if kind == 'rmsprop' else 'mom'
        epsilon = s['epsilon'] if kind == 'rmsprop' else s['eps']
        if kind == 'centered_rmsprop':
            s['mean_grad'] = rho*s['mean_grad']+(1-rho)*g
            variance = variance-s['mean_grad']**2
        s[slot] = s['momentum']*s[slot]+s['lr']*g/np.sqrt(variance+epsilon)
        s['var'] = x-s[slot]
    else:
        raise AssertionError('oracle missing for '+kind)
    return s


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('kind,name,order,slots,returned', OPTIMIZERS, ids=[item[0] for item in OPTIMIZERS])
@pytest.mark.parametrize('mapped_scalars', [False, True])
def test_optimizer_mapped_state_real_cpu_updates(
    monkeypatch, request, kind, name, order, slots, returned, mapped_scalars):
    """
    Feature: Remaining vmap dispatch, mutation and numerical contracts
    Description: Removing only the graph monad lets every actual CPU optimizer execute its update.
    Expectation: One native optimizer update matches independent state formulas, returns original state Tensors and
        binds axis zero.
    """
    prim = optimizer_primitive(name)
    if kind == 'sparse_ftrl':
        prim.add_prim_attr('batch_rank', 0)
    original_attrs = prim.attrs.copy()
    defects = {
        'ftrl': 'CPU Ftrl uses truthiness rather than sign(linear) for its L1 term and ignores mapped hyperparameters',
        'adam': 'CPU eager Adam returns tensors without the updated mutable values',
        'adagrad_da': 'CPU AdagradDA computes abs(accumulator-l1*step), instead of max(abs(accumulator)-l1*step,0)',
        'centered_rmsprop': 'CPU eager CenteredRMSProp return tensor does not contain updated variable values',
        'rmsprop': 'CPU eager RMSProp return tensor does not contain updated variable values',
    }
    if kind in defects or (kind == 'adam_with_amsgrad_v2' and mapped_scalars):
        reason = defects.get(kind, 'CPU AmsgradV2 dereferences beta1/beta2/epsilon at the first mapped sample')
        request.applymarker(pytest.mark.xfail(strict=True, raises=AssertionError, reason=reason))
    names, slot_names, return_names = order.split(), slots.split(), returned.split()
    scalar = {'lr': .15, 'alpha': .12, 'beta1': .8, 'beta2': .7, 'beta1_power': .64, 'beta2_power': .49,
                  'epsilon': .02, 'eps': .02, 'rho': .7, 'l1': .03, 'l2': .02, 'lr_power': -.5,
                  'logbase': np.log(2), 'sign_decay': .2, 'beta': .6, 'decay': .7, 'momentum': .1, 'global_step': 3}
    x = np.array([[1., -2., 3.], [4., -5., 6.]], np.float32)
    grad = np.array([[.5, -.3, 1.2], [-.7, .4, -.9]], np.float32)
    values, operands = {}, {}
    for key in names:
        if key in slot_names:
            array = x.copy() if key == 'var' else np.abs(x)/3+1 if key in ('v', 'accum', 'accum_update',
                           'mean_square', 'gradient_squared_accumulator') else np.abs(x)/5 if key == 'vhat' else x/20
            values[key] = array
            operands[key] = (ms.Parameter(tensor(array), name='remaining_'+key), 0)
        elif key in ('grad', 'delta'):
            array = grad[:, :2] if kind.startswith('sparse_') else grad
            values[key] = array
            # RMSProp requires the same physical source axis for grad and var.
            dim = 0 if kind == 'rmsprop' else 1
            operands[key] = (tensor(np.moveaxis(array, 0, dim)), dim)
        elif key == 'indices':
            values[key] = np.array([[2, 0], [2, 0]], np.int32)
            operands[key] = (tensor(values[key][0], np.int32), None)
        else:
            # RMSProp's decay/momentum/epsilon are native scalar attributes.
            mapped = mapped_scalars and not (kind == 'rmsprop' and key in ('decay', 'momentum', 'epsilon'))
            array = np.array([scalar[key], scalar[key]*.8] if mapped else scalar[key],
                             np.int64 if key == 'global_step' else np.float32)
            values[key] = array if mapped else np.repeat(array[None], 2)
            operands[key] = (tensor(array, array.dtype), 0 if mapped else None)
    expected = {key: values[key].copy() for key in slot_names}
    for batch in range(2):
        sample = {key: value[batch] for key, value in values.items()}
        indices = sample.get('indices')
        if indices is not None:
            for key in slot_names:
                sample[key] = sample[key][indices]
        updated = numpy_optimizer_step(kind, sample)
        for key in slot_names:
            if indices is None:
                expected[key][batch] = updated[key]
            else:
                expected[key][batch, indices] = updated[key]
    original_call = type(prim).__call__
    monad, calls, native_outputs = UMonad(), [], []
    attrs = dict(prim.attrs, batch_rank=1)

    def eager_native(actual_prim, *args):
        assert actual_prim is not prim and actual_prim.name == name
        assert actual_prim.attrs == attrs
        assert len(args) == len(names)+1 and args[-1] is monad
        for key, actual in zip(names, args[:-1]):
            if kind == 'rmsprop' and key in ('mean_square', 'moment'):
                assert_tensor(actual, values[key])
            elif key in slot_names:
                assert actual is operands[key][0]
            elif kind == 'rmsprop' and key in ('decay', 'momentum', 'epsilon'):
                assert actual is operands[key][0]
            else:
                dtype = ms.int32 if key == 'indices' else ms.int64 if key == 'global_step' else ms.float32
                assert_tensor(actual, values[key], dtype)
        calls.append(actual_prim)
        result = original_call(actual_prim, *args[:-1])
        native_outputs.append(result)
        return result

    monkeypatch.setattr(type(prim), '__call__', eager_native)
    result = optimizer_rule(kind, prim)(*(operands[key] for key in names), monad)
    assert len(calls) == 1 and prim.attrs == original_attrs
    results = [result] if len(return_names) == 1 else result
    native_results = [native_outputs[0]] if len(return_names) == 1 else native_outputs[0]
    assert len(results) == len(return_names)
    for (value, dim), key, native_value in zip(results, return_names, native_results):
        assert dim == 0 and value is native_value
        assert_tensor(value, expected[key])
    for key in slot_names:
        assert_tensor(operands[key][0], expected[key])


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('kind', ['gradient_descent', 'proximal_gradient_descent', 'proximal_adagrad',
                                 'adam_with_amsgrad', 'adam_with_amsgrad_v2', 'sparse_adagrad', 'sparse_ftrl'])
def test_optimizer_unmapped_state_real_cpu(monkeypatch, kind):
    """
    Feature: Remaining vmap dispatch, mutation and numerical contracts
    Description: Check optimizer unmapped state real CPU.
    Expectation: One native update mutates each state to its independent expected value and retains None output
        axes.
    """
    _, name, order, slots, returned = next(item for item in OPTIMIZERS if item[0] == kind)
    prim = optimizer_primitive(name)
    names, slot_names, return_names = order.split(), slots.split(), returned.split()
    x = np.array([1., -2., 3.], np.float32)
    scalar = {'alpha': .12, 'lr': .15, 'l1': .03, 'l2': .02, 'beta1': .8, 'beta2': .7,
                  'beta1_power': .64, 'beta2_power': .49, 'epsilon': .02}
    operands, values = {}, {}
    for key in names:
        if key in slot_names:
            values[key] = x.copy() if key == 'var' else np.abs(x)/3+1 if key in ('v',
                                 'accum') else np.abs(x)/5 if key == 'vhat' else x/20
            operands[key] = ms.Parameter(tensor(values[key]), name='unmapped_'+key)
        elif key in ('grad', 'delta'):
            values[key] = np.array([.5, -.3, 1.2], np.float32)
            operands[key] = tensor(values[key])
        elif key == 'indices':
            values[key] = np.arange(3, dtype=np.int32)
            operands[key] = tensor(values[key], np.int32)
        else:
            values[key] = np.array(scalar[key], np.float32)
            operands[key] = tensor(values[key])
    expected = numpy_optimizer_step(kind, values)
    monad, calls, original = UMonad(), [], type(prim).__call__
    attrs = prim.attrs.copy()
    def native(actual_prim, *args):
        assert actual_prim is prim and actual_prim.attrs == attrs
        assert len(args) == len(names)+1 and args[-1] is monad
        assert all(value is operands[key] for key, value in zip(names, args[:-1]))
        calls.append(args)
        return original(actual_prim, *args[:-1])
    monkeypatch.setattr(type(prim), '__call__', native)
    result = optimizer_rule(kind, prim)(*((operands[key], None) for key in names), monad)
    results = [result] if len(return_names) == 1 else result
    assert len(calls) == 1 and len(results) == len(return_names)
    for (value, dim), key in zip(results, return_names):
        assert dim is None
        assert_tensor(value, expected[key])
    for key in slot_names:
        assert_tensor(operands[key], expected[key])


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('kind', ['ftrl', 'adam', 'rmsprop', 'centered_rmsprop'])
def test_optimizer_unmapped_dispatch_protocol(monkeypatch, kind):
    """
    Feature: Remaining vmap dispatch, mutation and numerical contracts
    Description: Exact all-unmapped dispatch before CPU executors with independently reproduced numeric defects.
    Expectation: All original operands and outputs pass through exactly once with None axes before defective CPU
        kernels are reached.
    """
    _, name, order, slots, returned = next(item for item in OPTIMIZERS if item[0] == kind)
    prim = optimizer_primitive(name)
    names = order.split()
    operands = [tensor([1., 2.]) if key in slots.split() or key in ('grad', 'delta') else
                tensor([0], np.int32) if key == 'indices' else tensor(-.5 if key == 'lr_power' else .1)
                for key in names]
    outputs = [tensor([3.+index, 4.+index]) for index, _ in enumerate(returned.split())]
    monad, calls = UMonad(), []
    original_attrs = prim.attrs.copy()

    def native(actual_prim, *args):
        assert actual_prim is prim and actual_prim.attrs == original_attrs
        assert len(args) == len(operands)+1 and args[-1] is monad
        assert all(actual is expected for actual, expected in zip(args[:-1], operands))
        calls.append(args)
        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    monkeypatch.setattr(type(prim), '__call__', native)
    result = optimizer_rule(kind, prim)(*((value, None) for value in operands), monad)
    results = [result] if len(outputs) == 1 else result
    assert len(calls) == 1 and len(results) == len(outputs)
    assert all(value is expected and dim is None for (value, dim), expected in zip(results, outputs))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('kind', ['ftrl', 'adam', 'adagrad_da', 'centered_rmsprop', 'rmsprop', 'adam_with_amsgrad_v2'])
def test_optimizer_with_defective_cpu_kernel_mapped_executor_protocol(monkeypatch, kind):
    """
    Feature: Remaining vmap dispatch, mutation and numerical contracts
    Description: Verify batching before the independently reproduced faulty CPU execution boundary.
    Expectation: Aligned state/gradient values and native attributes reach one executor call; output identities and
        axis zero are retained.
    """
    _, name, order, slots, returned = next(item for item in OPTIMIZERS if item[0] == kind)
    prim = optimizer_primitive(name)
    names, slot_names = order.split(), slots.split()
    args, expected, originals = [], [], []
    for index, key in enumerate(names):
        dtype = np.int64 if key == 'global_step' else np.float32
        if key in slot_names:
            array = np.arange(6, dtype=dtype).reshape(2, 3)+index+1
            value = ms.Parameter(tensor(array, dtype), name='protocol_'+key)
            args.append((value, 0))
            originals.append((value, array.copy()))
            expected.append((array, value))
        elif key == 'grad':
            array = np.arange(6, dtype=dtype).reshape(2, 3)+index+1
            args.append((tensor(array if kind == 'rmsprop' else array.T, dtype), 0 if kind == 'rmsprop' else 1))
            expected.append((array, None))
        else:
            scalars = {'global_step': 3, 'lr_power': -.5, 'lr': .1, 'l1': .03, 'l2': .02,
                           'beta1': .8, 'beta2': .7, 'beta1_power': .64, 'beta2_power': .49,
                           'epsilon': .02, 'eps': .02, 'decay': .7, 'momentum': .1, 'rho': .7}
            array = np.array(scalars[key], dtype)
            args.append((tensor(array, dtype), None))
            expected.append((array if kind == 'rmsprop' and key in ('decay', 'momentum', 'epsilon')
                             else np.repeat(array[None], 2), None))
    output = [object() for _ in returned.split()]
    monad, calls = UMonad(), []
    attrs = dict(prim.attrs, batch_rank=1)
    def native(actual_prim, *received):
        assert actual_prim is not prim and actual_prim.attrs == attrs
        assert len(received) == len(names)+1 and received[-1] is monad
        for key, actual, (array, ref) in zip(names, received[:-1], expected):
            dtype = ms.int64 if key == 'global_step' else ms.float32
            assert_tensor(actual, array, dtype)
            if ref is not None and not (kind == 'rmsprop' and key in ('mean_square', 'moment')):
                assert actual is ref
        calls.append(received)
        return output[0] if len(output) == 1 else tuple(output)
    monkeypatch.setattr(type(prim), '__call__', native)
    result = optimizer_rule(kind, prim)(*args, monad)
    result = [result] if len(output) == 1 else result
    assert len(calls) == 1 and len(result) == len(output)
    assert all(value is wanted and axis == 0 for (value, axis), wanted in zip(result, output))
    for value, array in originals:
        assert_tensor(value, array)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.xfail(strict=True, raises=ValueError,
                   reason='PowerSign all-unmapped guard checks bdim pairs rather than their None axes')
def test_power_sign_unmapped_state_expected_cpu():
    """
    Feature: Remaining vmap dispatch, mutation and numerical contracts
    Description: Check power sign unmapped state expected CPU.
    Expectation: PowerSign state and moment should match their explicit CPU formulas while retaining None axes.
    """
    prim = P.ApplyPowerSign()
    operands = [ms.Parameter(tensor([1., 2.])), ms.Parameter(tensor([.1, .2]))]
    operands += [tensor(value) for value in (.1, np.log(2), .2, .6)]
    operands += [tensor([.5, -.5])]
    value, moment = N.get_apply_power_sign_rule(prim, 2)(*((x, None) for x in operands), UMonad())
    assert value[1] is moment[1] is None
    expected_m = .6*np.array([.1, .2])+.4*np.array([.5, -.5])
    expected_x = np.array([1., 2.])-.1*np.exp(np.log(2)*.2*np.sign([.5, -.5])*np.sign(expected_m))*[.5, -.5]
    assert_tensor(value[0], expected_x)
    assert_tensor(moment[0], expected_m)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('factory,name', [(O.get_identity_vmap_rule, 'Identity'), (O.get_load_vmap_rule, 'Load')])
@pytest.mark.parametrize('dim', [None, 1])
def test_identity_and_load_real_values(monkeypatch, factory, name, dim):
    """
    Feature: Remaining vmap dispatch, mutation and numerical contracts
    Description: Check identity and load real values.
    Expectation: Identity/Load preserve values and mapped metadata; Load receives the original reference and monad
        exactly once.
    """
    x = ms.Parameter(tensor([[1., 2.], [3., 4.]]), name='load_source')
    if name == 'Load':
        # Load is a graph-only native op: enforce the exact reference/monad protocol.
        prim, monad, calls = Primitive(name), UMonad(), []
        def native(self, ref, token):
            assert self is prim and ref is x and token is monad
            calls.append(ref)
            return x
        monkeypatch.setattr(Primitive, '__call__', native)
        result = factory(prim, 2)((x, dim), monad)
        assert calls == [x]
    else:
        result = factory(name, 2)((x, dim))
    value, axis = result
    assert axis == dim
    assert_tensor(value, [[1., 2.], [3., 4.]])


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('name', ['TupleGetItem', 'list_getitem'])
def test_sequence_getitem_native_protocol(monkeypatch, name):
    """
    Feature: Remaining vmap dispatch, mutation and numerical contracts
    Description: Check sequence getitem native protocol.
    Expectation: The native sequence helper receives index one once and returns the exact second packed input.
    """
    prim = Primitive(name)
    items = ((tensor([1., 2.]), 0), (tensor([3., 4.]), None))
    inputs = list(items) if name == 'list_getitem' else items
    calls = []
    def native(self, sequence, index):
        assert self is prim and sequence is inputs and index == 1
        calls.append(index)
        return sequence[index]
    monkeypatch.setattr(Primitive, '__call__', native)
    result = O.get_seq_get_item_vmap_rule(prim, 2)(inputs, (1, None))
    assert result is inputs[1] and calls == [1]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('name', ['Partial', 'Switch'])
@pytest.mark.parametrize('as_string', [False, True])
def test_control_flow_exact_unmapped_protocol(monkeypatch, name, as_string):
    """
    Feature: Remaining vmap dispatch, mutation and numerical contracts
    Description: Check control flow exact unmapped protocol.
    Expectation: Unmapped control-flow inputs dispatch once unchanged; mapped data raises ValueError without another
        native call.
    """
    prim, calls = Primitive(name), []
    output = object()
    def branch(value, flag=True):
        return value+int(flag)
    marker = branch if name == 'Partial' else True
    x = tensor([1., 2.]) if name == 'Partial' else branch
    last = True if name == 'Partial' else lambda value: value-1
    def native(self, *args):
        assert self.name == name and len(args) == 3
        assert args[0] is marker and args[1] is x and args[2] is last
        calls.append(args)
        return output
    monkeypatch.setattr(Primitive, '__call__', native)
    rule = O.get_partical_vmap_rule(name if as_string else prim, 2)
    assert rule(marker, (x, None), (last, None)) is output
    assert len(calls) == 1
    with pytest.raises(ValueError, match='source axis.*must be None'):
        rule(marker, (x, 0))
    assert len(calls) == 1


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('as_string', [False, True])
def test_print_graph_protocol_includes_each_axis_and_monad(monkeypatch, as_string):
    """
    Feature: Remaining vmap dispatch, mutation and numerical contracts
    Description: Check print graph protocol includes each axis and monad.
    Expectation: Print dispatch includes label, Tensor axis, scalar and monad once; unsupported metadata raises
        before dispatch.
    """
    prim, calls = P.Print(), []
    x, token, output = tensor([1., 2.]), UMonad(), object()
    def native(self, *args):
        assert self.name == 'Print' and len(args) == 8
        assert args[0] == 'label' and args[1] == '(' and args[2] is x
        assert args[3:7] == (', dim: ', 1, ')', 7) and args[-1] is token
        calls.append(args)
        return output
    monkeypatch.setattr(Primitive if as_string else type(prim), '__call__', native)
    rule = Debug.get_print_vmap_rule('Print' if as_string else prim, 2)
    assert rule(('label', None), (x, 1), (7, None), token) is output
    assert len(calls) == 1
    with pytest.raises(ValueError, match='does not contain axis'):
        rule('missing-axis', token)
    assert len(calls) == 1


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason='Tile computes output batch axis before padding short multiples')
def test_tile_short_multiples_preserve_unmentioned_dimensions():
    """
    Feature: Remaining vmap dispatch, mutation and numerical contracts
    Description: Check tile short multiples preserve unmentioned dimensions.
    Expectation: Short repeats affect only the final sample dimension and retain leading batch axis zero.
    """
    x = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    value, dim = V.get_tile_vmap_rule(P.Tile(), 2)((tensor(x), 0), ((2,), None))
    assert dim == 0
    assert_tensor(value, np.tile(x, (1, 1, 2)))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_concat_rejects_non_tuple_list_sequence():
    """
    Feature: Remaining vmap dispatch, mutation and numerical contracts
    Description: Check concat rejects non tuple list sequence.
    Expectation: A non-tuple/list sequence raises the container-contract ValueError.
    """
    class OtherSequence:
        def __iter__(self):
            return iter(((tensor([1., 2.]), 0),))
    rule = V.get_concat_vmap_rule(P.Concat(), 2)
    with pytest.raises(ValueError, match='neither tuple nor list'):
        rule(OtherSequence(), (0, None))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_gatherd_rejects_dynamic_dimension_before_native_dispatch():
    """
    Feature: Remaining vmap dispatch, mutation and numerical contracts
    Description: Check gatherd rejects dynamic dimension before native dispatch.
    Expectation: A Tensor dimension raises ValueError requiring a static integer before native execution.
    """
    with pytest.raises(ValueError, match='must be a int'):
        V.get_gatherd_vmap_rule(P.GatherD(), 2)((tensor([[1., 2.], [3., 4.]]), 0),
            (tensor(0, np.int32), None), (tensor([[0, 1], [1, 0]], np.int32), 0))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_unmapped_fills_missing_cpu_kernel_native_protocol(monkeypatch):
    """
    Feature: Remaining vmap dispatch, mutation and numerical contracts
    Description: Check unmapped fills missing CPU kernel native protocol.
    Expectation: Fills receives the original data and scalar three once, returning the exact output with None axis.
    """
    prim = V.Fills()
    data, output, calls = tensor([1., 2.]), object(), []
    def native(actual_prim, x, value):
        assert actual_prim is prim and x is data and value == 3.
        calls.append((x, value))
        return output
    monkeypatch.setattr(type(prim), '__call__', native)
    value, axis = V.get_fills_vmap_rule(prim, 2)((data, None), (3., None))
    assert value is output and axis is None and calls == [(data, 3.)]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_inplace_add_alpha_graph_executor_protocol(monkeypatch):
    """
    Feature: Remaining vmap dispatch, mutation and numerical contracts
    Description: The CPU lacks InplaceAddExt; validate alpha, monad and aligned operands only.
    Expectation: The executor receives aligned operands, alpha and monad once; its output binds axis zero and source
        values stay intact.
    """
    prim = Gen.InplaceAddExt()
    variable = tensor([[1., 2.], [3., 4.]])
    value = tensor([5., 6.])
    token, output, calls = UMonad(), object(), []
    def native(actual_prim, x, y, alpha, monad):
        assert actual_prim is prim and x is variable and alpha == 3 and monad is token
        assert_tensor(y, [[5., 6.], [5., 6.]])
        calls.append(y)
        return output
    monkeypatch.setattr(type(prim), '__call__', native)
    result, axis = B.get_assign_vmap_rule(prim, 2)((variable, 0), (value, None), (3, None), token)
    assert result is output and axis == 0 and len(calls) == 1
    assert_tensor(variable, [[1., 2.], [3., 4.]])


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('name', ['MinimumGrad', 'MaximumGrad'])
@pytest.mark.xfail(strict=True, raises=TypeError,
                   reason='String extremum-gradient factory calls generated three-input wrapper with five arguments')
def test_extremum_gradient_string_constructor_real_cpu(name):
    """
    Feature: Remaining vmap dispatch, mutation and numerical contracts
    Description: Check extremum gradient string constructor real CPU.
    Expectation: String-created Maximum/Minimum gradients route each nonuniform sensitivity to the NumPy-selected
        operand.
    """
    x = np.array([[1., 4., 2.], [5., 1., 3.]], np.float32)
    y = np.array([[3., 2., 6.], [2., 4., 1.]], np.float32)
    g = np.arange(1, 7, dtype=np.float32).reshape(2, 3)
    result = GradMath.get_broadcast_binary_op_grad_vmap_rule(name, 2)(
        (tensor(x), 0), (tensor(y), 0), (tensor(g), 0), (True, None), (True, None))
    select = x < y if name == 'MinimumGrad' else x > y
    assert len(result) == 2 and result[0][1] == result[1][1] == 0
    assert_tensor(result[0][0], np.where(select, g, 0))
    assert_tensor(result[1][0], np.where(select, 0, g))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_multinomial_rejects_mapped_sample_count():
    """
    Feature: Remaining vmap dispatch, mutation and numerical contracts
    Description: Check multinomial rejects mapped sample count.
    Expectation: Mapped num_samples metadata raises ValueError requiring a None source axis.
    """
    with pytest.raises(ValueError, match='source axis.*must be None'):
        R.get_multinomial_vmap_rule(P.Multinomial(seed=3), 2)(
            (tensor([[.3, .7], [.6, .4]]), 0), (2, 0))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('factory,prim,args', [
    (N.get_bce_with_logits_loss_vamp_rule, NN.BCEWithLogitsLoss(),
     [(tensor([[1., 2.], [3., 4.]]), 0)]*4+[(99, None)]),
    (N.get_random_categorical_vmap_rule, P.RandomCategorical(),
     [(tensor(np.ones((2, 2, 3))), 0), (2, 0), (3, None)]),
    (N.get_pad_v3_vmap_rule, NN.PadV3(), [(tensor([[1., 2.], [3., 4.]]), 0)]),
])
def test_nn_invalid_configuration_stops_before_native(factory, prim, args):
    """
    Feature: Remaining vmap dispatch, mutation and numerical contracts
    Description: Check nn invalid configuration stops before native.
    Expectation: Invalid reduction, sample count or parameter metadata raises its configuration error before native
        execution.
    """
    with pytest.raises((RuntimeError, ValueError), match='reduction|num_sample|params'):
        factory(prim, 2)(*args)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_softmax_dynamic_axis_metadata_is_rejected(monkeypatch):
    """
    Feature: Remaining vmap dispatch, mutation and numerical contracts
    Description: Check softmax dynamic axis metadata is rejected.
    Expectation: A single constantness query detects dynamic axis metadata and raises ValueError.
    """
    axis = ms.mutable(-1)
    calls = []
    def isconstant(value):
        assert value is axis
        calls.append(value)
        return False
    monkeypatch.setattr(N.F, 'isconstant', isconstant)
    with pytest.raises(ValueError):
        N.get_softmax_vmap_rule(P.Softmax(), 2)((tensor([[1., 2.], [3., 4.]]), 0), (axis, None))
    assert len(calls) == 1


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('dim', [0, 1])
def test_log_softmax_shifted_physical_axis_real_numpy(dim):
    """
    Feature: Remaining vmap dispatch, mutation and numerical contracts
    Description: Check log softmax shifted physical axis real numpy.
    Expectation: Log-softmax matches NumPy after physical-axis shifting and preserves the original mapped dimension.
    """
    data = np.array([[1., -2., 3.], [4., 1., -3.]], np.float32)
    physical = np.moveaxis(data, 0, dim)
    value, axis = N.get_log_softmax_vmap_rule(P.LogSoftmax(), 2)((tensor(physical), dim), (-1, None))
    expected = data-np.log(np.exp(data).sum(axis=-1, keepdims=True))
    assert axis == dim
    assert_tensor(value, np.moveaxis(expected, 0, dim))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.xfail(strict=True, raises=ValueError,
                   reason='Pdist CPU rejects the rank-three input emitted by its batching rule')
def test_pdist_string_rule_real_cpu_batch_distances():
    """
    Feature: Remaining vmap dispatch, mutation and numerical contracts
    Description: Check pdist string rule real CPU batch distances.
    Expectation: String-created Pdist returns distance five for each sample with output axis zero.
    """
    data = np.array([[[0., 0.], [3., 4.]], [[1., 2.], [4., 6.]]], np.float32)
    value, axis = N.get_pdist_vmap_rule('Pdist', 2)((tensor(data), 0))
    assert axis == 0
    assert_tensor(value, [[5.], [5.]])


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_pdist_missing_batched_kernel_native_protocol(monkeypatch):
    """
    Feature: Remaining vmap dispatch, mutation and numerical contracts
    Description: Check pdist missing batched kernel native protocol.
    Expectation: The normalized batched operand reaches exactly one native call and its output identity binds axis
        zero.
    """
    data = np.array([[[0., 0.], [3., 4.]], [[1., 2.], [4., 6.]]], np.float32)
    output, calls = object(), []
    def native(prim, x):
        assert prim.name == 'Pdist' and prim.attrs == {'p': 2.0}
        assert_tensor(x, data)
        calls.append(x)
        return output
    monkeypatch.setattr(Primitive, '__call__', native)
    value, axis = N.get_pdist_vmap_rule('Pdist', 2)((tensor(np.moveaxis(data, 0, 1)), 1))
    assert value is output and axis == 0 and len(calls) == 1


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_unmapped_pad_and_lrn_preserve_none_axis_real_cpu():
    """
    Feature: Remaining vmap dispatch, mutation and numerical contracts
    Description: Check unmapped pad and lrn preserve none axis real CPU.
    Expectation: Padding and local response normalization match independent formulas and retain None axes.
    """
    # Generated native rule factories expose dynamic call signatures.
    # pylint: disable=too-many-function-args
    x = tensor([[1., 2.], [3., 4.]])
    value, axis = N.get_pad_vmap_rule(P.Pad(((1, 0), (0, 1))), 2)((x, None))
    assert axis is None
    assert_tensor(value, np.pad([[1., 2.], [3., 4.]], ((1, 0), (0, 1))))
    array = np.array([1., 2., 3.], np.float32).reshape(1, 3, 1, 1)
    value, axis = N.get_lrn_vmap_rule(P.LRN(depth_radius=1, bias=2., alpha=.5, beta=.75), 2)((tensor(array), None))
    expected = array.copy()
    for channel in range(3):
        expected[:, channel] /= (2+.5*(array[:, max(0, channel-1):channel+2]**2).sum(axis=1))**.75
    assert axis is None
    assert_tensor(value, expected)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('kind', ['mirror', 'mirror_grad', 'pad_v3'])
def test_unmapped_variadic_padding_real_cpu(kind):
    """
    Feature: Remaining vmap dispatch, mutation and numerical contracts
    Description: Check unmapped variadic padding real CPU.
    Expectation: Reflect, mirror-gradient and constant padding match the specified arrays and retain unmapped
        metadata.
    """
    data = np.array([[1., 2.], [3., 4.]], np.float32)
    pads = tensor([[1, 1], [1, 1]], np.int64)
    if kind == 'mirror':
        rule = N.get_mirror_pad_vmap_rule(P.MirrorPad(mode='REFLECT'), 2)
        args = (tensor(data), pads)
        expected = np.pad(data, ((1, 1), (1, 1)), mode='reflect')
    elif kind == 'mirror_grad':
        rule = D.get_mirror_pad_grad_grad_vmap_rule(G.MirrorPadGrad(mode='REFLECT'), 2)
        args = (tensor(np.ones((4, 4))), pads)
        expected = np.full((2, 2), 4.)
    else:
        rule = N.get_pad_v3_vmap_rule(NN.PadV3(mode='constant'), 2)
        args = (tensor(data), tensor([1, 1], np.int64), tensor(-2.))
        expected = np.pad(data, ((0, 0), (1, 1)), constant_values=-2)
    value, axis = rule(*((arg, None) for arg in args))
    assert axis is None
    assert_tensor(value, expected)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_unmapped_dense_and_logsoftmax_real_cpu():
    """
    Feature: Remaining vmap dispatch, mutation and numerical contracts
    Description: Check unmapped dense and logsoftmax real CPU.
    Expectation: Dense, log-softmax and softmax equal their NumPy formulas with None output axes.
    """
    data = np.array([[1., 2.], [3., 4.]], np.float32)
    weight = np.array([[2., 1.], [-1., 3.]], np.float32)
    value, axis = N.get_dense_vmap_rule(P.Dense(), 2)((tensor(data), None), (tensor(weight), None),
                                                    (tensor([.5, -.5]), None))
    assert axis is None
    assert_tensor(value, data@weight.T+[.5, -.5])
    value, axis = N.get_log_softmax_vmap_rule(Primitive('LogSoftmax'), 2)((tensor(data), None), (-1, None))
    assert axis is None
    assert_tensor(value, data-np.log(np.exp(data).sum(axis=-1, keepdims=True)))
    value, axis = N.get_softmax_vmap_rule(Primitive('Softmax'), 2)((tensor(data), None), (-1, None))
    assert axis is None
    assert_tensor(value, np.exp(data)/np.exp(data).sum(axis=-1, keepdims=True))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_unmapped_categorical_real_cpu_deterministic_support():
    """
    Feature: Remaining vmap dispatch, mutation and numerical contracts
    Description: Check unmapped categorical real CPU deterministic support.
    Expectation: Extreme logits produce only categories zero and one for their respective rows and retain None
        metadata.
    """
    data = tensor([[0., -1000.], [-1000., 0.]])
    value, axis = N.get_random_categorical_vmap_rule(P.RandomCategorical(), 2)(
        (data, None), (3, None), (71, None))
    assert axis is None
    assert_tensor(value, [[0, 0, 0], [1, 1, 1]], ms.int64)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_bce_invalid_reduction_after_broadcast_preparation():
    """
    Feature: Remaining vmap dispatch, mutation and numerical contracts
    Description: Check BCE invalid reduction after broadcast preparation.
    Expectation: Binary cross entropy rejects an unknown reduction after preparing broadcast operands.
    """
    logits, labels = tensor(np.ones((2, 3))), tensor(np.zeros((2, 3)))
    with pytest.raises(RuntimeError, match='reduction must in'):
        N.get_bce_with_logits_loss_vamp_rule(NN.BCEWithLogitsLoss(), 2)(
            (logits, 0), (labels, 0), (tensor([1., 2., 3.]), None),
            (tensor([1., 1., 1.]), None), (99, None))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason='KLDivLossGrad mean reduction includes the mapped batch dimension in its denominator')
def test_kl_mean_gradient_string_factory_per_sample_expected_cpu():
    """
    Feature: Remaining vmap dispatch, mutation and numerical contracts
    Description: Check KL mean gradient string factory per sample expected CPU.
    Expectation: The string gradient factory should return -2 times target divided by three with mapped axis zero.
    """
    data = np.log(np.array([[.2, .3, .5], [.1, .4, .5]], np.float32))
    target = np.array([[.3, .2, .5], [.2, .3, .5]], np.float32)
    value, axis = N.get_kl_div_loss_grad_vmap_rule('KLDivLossGrad', 2)(
        (tensor(2.), None), (tensor(data), 0), (tensor(target), 0))
    assert axis == 0
    assert_tensor(value, -2*target/3)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_kl_gradient_string_factory_rejects_integer_target_dtype():
    """
    Feature: Remaining vmap dispatch, mutation and numerical contracts
    Description: Check KL gradient string factory rejects integer target dtype.
    Expectation: Integer target Tensors raise TypeError identifying floating/same-dtype requirements.
    """
    with pytest.raises(TypeError, match='dtype|Float|same'):
        N.get_kl_div_loss_grad_vmap_rule('KLDivLossGrad', 2)(
            (tensor(2.), None), (tensor(np.ones((2, 3))), 0),
            (tensor(np.ones((2, 3)), np.int32), 0))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_unmapped_lrn_gradient_alpha_zero_analytic_cpu():
    """
    Feature: Remaining vmap dispatch, mutation and numerical contracts
    Description: Check unmapped lrn gradient alpha zero analytic CPU.
    Expectation: With alpha zero, each sensitivity is divided by 2**0.75 and the output axis stays None.
    """
    # Generated native rule factories expose dynamic call signatures.
    # pylint: disable=too-many-function-args
    data = np.array([1., 2., 3.], np.float32).reshape(1, 3, 1, 1)
    sensitivity = data+1
    prim = G.LRNGrad(depth_radius=1, bias=2., alpha=0., beta=.75)
    value, axis = N.get_lrn_grad_vmap_rule(prim, 2)((tensor(sensitivity), None),
        (tensor(data), None), (tensor(data/2**.75), None))
    assert axis is None
    assert_tensor(value, sensitivity/2**.75)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_bias_add_nhwc_single_channel_real_cpu():
    """
    Feature: Remaining vmap dispatch, mutation and numerical contracts
    Description: Check bias add NHWC single channel real CPU.
    Expectation: The shared channel bias adds two to every element while preserving axis zero.
    """
    data = np.arange(12, dtype=np.float32).reshape(2, 1, 2, 3, 1)
    value, axis = N.get_bias_add_vmap_rule(P.BiasAdd(), 2)(
        (tensor(data), 0), (tensor([2.]), None), (FormatEnum.NHWC, None))
    assert axis == 0
    assert_tensor(value, data+2)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.xfail(strict=True, raises=ValueError, reason='NHWC BiasAdd uses rank as its channel index')
def test_bias_add_nhwc_distinct_channels_expected_cpu():
    """
    Feature: Remaining vmap dispatch, mutation and numerical contracts
    Description: Check bias add NHWC distinct channels expected CPU.
    Expectation: NHWC channels should receive their distinct biases [2, -3] independently with mapped axis zero.
    """
    data = np.arange(24, dtype=np.float32).reshape(2, 1, 2, 3, 2)
    value, axis = N.get_bias_add_vmap_rule(P.BiasAdd(), 2)(
        (tensor(data), 0), (tensor([2., -3.]), None), (FormatEnum.NHWC, None))
    assert axis == 0
    assert_tensor(value, data+[2., -3.])


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.xfail(strict=True, raises=AssertionError, reason='NHWC BiasAddGrad reduces the channel axis too')
def test_bias_add_gradient_nhwc_preserves_channels_expected_cpu():
    """
    Feature: Remaining vmap dispatch, mutation and numerical contracts
    Description: Check bias add gradient NHWC preserves channels expected CPU.
    Expectation: The NHWC bias gradient should reduce spatial/sample axes and retain separate channel sums.
    """
    data = np.arange(24, dtype=np.float32).reshape(2, 1, 2, 3, 2)
    value, axis = N.get_bias_add_grad_vmap_rule(G.BiasAddGrad(), 2)(
        (tensor(data), 0), (FormatEnum.NHWC, None))
    assert axis == 0
    assert_tensor(value, data.sum(axis=(1, 2, 3)))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('mapped', [False, True])
def test_instance_norm_forward_complete_native_protocol(monkeypatch, mapped):
    """
    Feature: Remaining vmap dispatch, mutation and numerical contracts
    Description: Batched InstanceNorm lacks a CPU kernel: validate only its exact executor contract.
    Expectation: One native call receives the complete normalized operand/attribute contract and preserves all three
        output identities/axes.
    """
    prim = NN.InstanceNorm(epsilon=.001, momentum=.2)
    data = np.arange(16, dtype=np.float32).reshape(
        2, 1, 2, 2, 2) if mapped else np.arange(8, dtype=np.float32).reshape(1, 2, 2, 2)
    x = tensor(np.moveaxis(data, 0, 1) if mapped else data)
    stats = [tensor(np.full((2, 2) if mapped else (2,), value)) for value in (1., 0., .2, 2.)]
    outputs, token, calls = [object(), object(), object()], UMonad(), []
    attrs = dict(prim.attrs, **({'batch_rank': 1} if mapped else {}))
    def native(actual_prim, actual_x, gamma, beta, mean, variance, monad):
        assert actual_prim.attrs == attrs and monad is token
        assert (actual_prim is prim) is (not mapped)
        assert_tensor(actual_x, data)
        assert all(actual is expected for actual, expected in zip((gamma, beta, mean, variance), stats))
        calls.append(actual_x)
        return tuple(outputs)
    monkeypatch.setattr(type(prim), '__call__', native)
    result = N.get_instance_norm_rule(prim, 2)((x, 1 if mapped else None),
        *((value, 0 if mapped else None) for value in stats), token)
    assert len(calls) == 1 and len(result) == 3
    assert all(value is expected and axis == (0 if mapped else None)
               for (value, axis), expected in zip(result, outputs))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason='ReverseSequence with mapped axis after sample batch returns transposed sample dimensions')
def test_reverse_sequence_with_batch_after_sequence_real_cpu():
    """
    Feature: Remaining vmap dispatch, mutation and numerical contracts
    Description: Check reverse sequence with batch after sequence real CPU.
    Expectation: Each sequence prefix reverses according to its own length while the mapped axis remains zero.
    """
    data = np.arange(12, dtype=np.float32).reshape(2, 3, 2, 1)
    lengths = np.array([[1, 3], [2, 1]], np.int32)
    expected = data.copy()
    for mapped in range(2):
        for batch in range(2):
            length = lengths[mapped, batch]
            expected[mapped, :length, batch] = data[mapped, :length, batch][::-1]
    value, axis = V.get_reverse_sequence_vmap_rule(P.ReverseSequence(seq_dim=0, batch_dim=1), 2)(
        (tensor(np.moveaxis(data, 0, 2)), 2), (tensor(lengths, np.int32), 0))
    assert axis == 0
    assert_tensor(value, expected)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_reverse_sequence_rejects_lengths_exceeding_sequence_extent():
    """
    Feature: Remaining vmap dispatch, mutation and numerical contracts
    Description: Check reverse sequence rejects lengths exceeding sequence extent.
    Expectation: Lengths larger than the sequence extent raise the native Invalid seq_lengths RuntimeError.
    """
    data = np.arange(12, dtype=np.float32).reshape(2, 3, 2, 1)
    lengths = np.array([[4, 1], [2, 1]], np.int32)
    with pytest.raises(RuntimeError, match='Invalid seq_lengths'):
        V.get_reverse_sequence_vmap_rule(P.ReverseSequence(seq_dim=0, batch_dim=1), 2)(
            (tensor(np.moveaxis(data, 0, 2)), 2), (tensor(lengths, np.int32), 0))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_fft_string_factory_mapped_input_real_cpu():
    """
    Feature: Remaining vmap dispatch, mutation and numerical contracts
    Description: Check FFT string factory mapped input real CPU.
    Expectation: String-created FFT equals NumPy complex64 Fourier transforms with mapped axis zero.
    """
    data = np.array([[1+1j, 2-1j, 3+2j, -1j], [2j, 1+2j, -3j, 4+1j]], np.complex64)
    value, axis = W.get_fft_with_size_vmap_rule('FFTWithSize', 2)((tensor(data, np.complex64), 0))
    assert axis == 0
    assert_tensor(value, np.fft.fft(data), ms.complex64)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('shape', [(2, 1, 1, 2, 2, 2), (2, 1, 1, 1, 2, 2, 2)])
def test_adaptive_max_pool3d_high_rank_reshape_real_cpu(shape, monkeypatch):
    """
    Feature: Remaining vmap dispatch, mutation and numerical contracts
    Description: Check adaptive max pool3d high rank reshape real CPU.
    Expectation: All leading dimensions are reconstructed, each pooled value matches its maximum and argmax equals
        seven.
    """
    real_reshape = N.F.reshape

    def reshape_with_ready_shape(value, output_shape):
        # Preserve the graph rule's shape-value dependency at this eager boundary.
        # Materialize before CPU shape conversion, then forward the original Tensor.
        if isinstance(output_shape, ms.Tensor):
            output_shape.asnumpy()
        return real_reshape(value, output_shape)

    monkeypatch.setattr(N.F, 'reshape', reshape_with_ready_shape)
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    result = N.get_adaptive_max_pool3d_vmap_rule(P.AdaptiveMaxPool3D(), 2)(
        (tensor(data), 0), (tensor([1, 1, 1], np.int32), None))
    expected = data.max(axis=(-3, -2, -1), keepdims=True)
    (value, dim), (indices, index_dim) = result
    assert dim == index_dim == 0
    assert_tensor(value, expected)
    assert_tensor(indices, np.full(expected.shape, 7), ms.int32)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('factory,prim,args', [
    (D.get_nll_loss_grad_vmap_rule, G.NLLLossGrad(),
     [(tensor(np.ones((2, 1, 3))), 0), (tensor([1., 2.]), 0),
      (tensor([[0], [1]], np.int32), 0), (tensor(np.ones((2, 3))), 0),
      (tensor(2.), None), (0, None), (-100, None)]),
    (D.get_batchnorm_grad_vmap_rule, G.BatchNormGrad(),
     [(tensor(np.ones((2, 3))), 0)]*6+[(False, None), (.001, None), (FormatEnum.NCHW, None)]),
])
def test_gradient_argument_rank_and_weight_contract(factory, prim, args):
    """
    Feature: Remaining vmap dispatch, mutation and numerical contracts
    Description: Check gradient argument rank and weight contract.
    Expectation: Invalid gradient ranks or mapped weight metadata raise ValueError before gradient execution.
    """
    with pytest.raises(ValueError, match='weight|dim of'):
        factory(prim, 2)(*args)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('existing_batch', [False, True])
def test_instance_norm_gradient_native_protocol(monkeypatch, existing_batch):
    """
    Feature: Remaining vmap dispatch, mutation and numerical contracts
    Description: CPU lacks batched InstanceNormGrad; check real clone, all operands and output axes.
    Expectation: A real clone and all original operands reach one native call, preserving three output objects and
        axis-zero metadata.
    """
    prim = G.InstanceNormGrad(epsilon=.001)
    if existing_batch:
        prim.add_prim_attr('batch_rank', 1)
    data = np.arange(16, dtype=np.float32).reshape(2, 1, 2, 2, 2)
    values = [data+1, data, np.array([[1., 2.], [3., 4.]], np.float32),
              np.array([[.1, .2], [.3, .4]], np.float32), np.ones((2, 2), np.float32)]
    args = [(tensor(np.moveaxis(value, 0, 1)), 1) for value in values]
    outputs = [tensor(data/2), tensor([[1., 2.], [3., 4.]]), tensor([[5., 6.], [7., 8.]])]
    calls = []
    expected_attrs = dict(prim.attrs, batch_rank=2 if existing_batch else 1)
    def native(actual_prim, *received):
        assert actual_prim is not prim and actual_prim.attrs == expected_attrs
        assert len(received) == 5
        for actual, expected in zip(received, values):
            assert_tensor(actual, expected)
        calls.append(received)
        return tuple(outputs)
    monkeypatch.setattr(type(prim), '__call__', native)
    result = D.get_instance_norm_grad_rule(prim, 2)(*args)
    assert len(calls) == 1 and len(result) == 3
    assert all(value is expected and axis == 0 for (value, axis), expected in zip(result, outputs))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('kind', ['max2d', 'avg3d', 'maxgrad2d',
    pytest.param('maxgrad3d', marks=pytest.mark.xfail(strict=True, raises=ValueError,
        reason='MaxPool3DGradGrad preprocessing forwards only y to a three-input primitive')),
    'adaptive2d', 'adaptive3d'])
def test_unmapped_pool_gradients_real_cpu(kind):
    """
    Feature: Remaining vmap dispatch, mutation and numerical contracts
    Description: An unmapped rule must preserve its None axis and exact nonuniform sensitivities.
    Expectation: Each pooling gradient matches its independently constructed sparse or uniform sensitivity
        distribution and retains None axis.
    """
    rank = 3 if kind in ('avg3d', 'maxgrad3d', 'adaptive3d') else 2
    shape = (1, 1)+(2,)*rank
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    maximum = tensor(np.full((1, 1)+(1,)*rank, data.max()))
    sensitivity = tensor(np.full((1, 1)+(1,)*rank, 3.))
    if kind == 'max2d':
        factory, prim = D.get_max_pool_grad_vmap_rule, G.MaxPoolGrad(2, 2)
        args = (tensor(data), maximum, sensitivity)
        expected = np.zeros_like(data)
        expected.reshape(-1)[-1] = 3
    elif kind == 'avg3d':
        factory, prim = D.get_avg_pool3d_grad_vmap_rule, G.AvgPool3DGrad(2, 2)
        args, expected = (shape, sensitivity), np.full(shape, 3/8, np.float32)
    elif kind.startswith('maxgrad'):
        factory = D.get_maxpool_grad_grad_vmap_rule if rank == 2 else D.get_maxpool_3d_grad_grad_vmap_rule
        prim = G.MaxPoolGradGrad(2, 2) if rank == 2 else G.MaxPool3DGradGrad(2, 2)
        args = (tensor(data), maximum, tensor(data+2))
        expected = np.full((1, 1)+(1,)*rank, data.max()+2)
    else:
        factory = D.get_adaptive_avgpool2d_vmap_rule
        prim = G.AdaptiveMaxPool2DGrad() if rank == 2 else G.AdaptiveMaxPool3DGrad()
        args = (sensitivity, tensor(data), tensor(
            np.full((1, 1)+(1,)*rank, data.size-1), np.int64 if rank == 2 else np.int32))
        expected = np.zeros_like(data)
        expected.reshape(-1)[-1] = 3
    value, axis = factory(prim, 2)(*((arg, None) for arg in args))
    assert axis is None
    assert_tensor(value, expected)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_unmapped_nll_gradient_real_cpu():
    """
    Feature: Remaining vmap dispatch, mutation and numerical contracts
    Description: Check unmapped NLL gradient real CPU.
    Expectation: Class weights and sensitivities scatter to [[0, 0, -4], [-3, 0, 0]] with None output axis.
    """
    args = [tensor([[-1., -2., -3.], [-4., -5., -6.]]), tensor([2., 3.]),
            tensor([2, 0], np.int32), tensor([1., .5, 2.]), tensor(3.), int(ReductionEnum.NONE), -100]
    value, axis = D.get_nll_loss_grad_vmap_rule(Primitive('NLLLossGrad'), 2)(*((arg, None) for arg in args))
    assert axis is None
    assert_tensor(value, [[0., 0., -4.], [-3., 0., 0.]])


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_unmapped_average_pool2d_gradient_real_cpu():
    """
    Feature: Remaining vmap dispatch, mutation and numerical contracts
    Description: Check unmapped average pool2d gradient real CPU.
    Expectation: Sensitivity three distributes as 0.75 across the four input cells with None output axis.
    """
    data = tensor(np.arange(4).reshape(1, 1, 2, 2))
    args = [data, tensor([[[[1.5]]]]), tensor([[[[3.]]]]), (2, 2), (2, 2), 2, 0]
    value, axis = D.get_avg_pool_grad_vmap_rule(Primitive('AvgPoolGrad'), 2)(*((arg, None) for arg in args))
    assert axis is None
    assert_tensor(value, np.full((1, 1, 2, 2), .75))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_unmapped_max_pool3d_argmax_gradient_real_cpu():
    """
    Feature: Remaining vmap dispatch, mutation and numerical contracts
    Description: Check unmapped max pool3d argmax gradient real CPU.
    Expectation: Only the recorded maximum position receives its sensitivity and output metadata remains None.
    """
    args = [tensor(np.arange(8).reshape(1, 1, 2, 2, 2)), tensor([[[[[3.]]]]]),
            tensor([[[[[7]]]]], np.int64)]
    value, axis = D.get_max_pool3d_grad_with_argmax_vmap_rule(G.MaxPool3DGradWithArgmax(2, 2, 0), 2)(
        *((arg, None) for arg in args))
    expected = np.zeros((1, 1, 2, 2, 2), np.float32)
    expected[..., -1, -1, -1] = 3
    assert axis is None
    assert_tensor(value, expected)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_unmapped_layernorm_gradient_analytic_cpu():
    """
    Feature: Remaining vmap dispatch, mutation and numerical contracts
    Description: Check unmapped layernorm gradient analytic CPU.
    Expectation: All three analytic layer-normalization derivatives match their NumPy expectations and retain None
        axes.
    """
    x = np.array([[1., 2., 4.], [-1., 1., 3.]], np.float32)
    dy = np.array([[2., 1., -1.], [1., -2., 3.]], np.float32)
    mean, variance = x.mean(axis=1, keepdims=True), x.var(axis=1, keepdims=True)
    gamma = np.array([1., 2., 3.], np.float32)
    normalized = (x-mean)/np.sqrt(variance+1e-12)
    weighted = dy*gamma
    dx = (weighted-weighted.mean(axis=1, keepdims=True)-normalized* \
          (weighted*normalized).mean(axis=1, keepdims=True))/np.sqrt(variance+1e-12)
    args = [tensor(x), tensor(dy), tensor(variance), tensor(mean), tensor(gamma), 1, 1]
    result = D.get_layernormgrad_vmap_rule(Primitive('LayerNormGrad'), 2)(*((arg, None) for arg in args))
    assert len(result) == 3 and all(axis is None for _, axis in result)
    for (actual, _), expected in zip(result, (dx, (dy*normalized).sum(axis=0), dy.sum(axis=0))):
        assert_tensor(actual, expected)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_unmapped_grid_sampler_gradient_bilinear_cpu():
    """
    Feature: Remaining vmap dispatch, mutation and numerical contracts
    Description: Check unmapped grid sampler gradient bilinear CPU.
    Expectation: The image gradient is 0.5 per cell and grid gradient is [1, 2], both with None axes.
    """
    args = [tensor([[[[2.]]]]), tensor(np.arange(4).reshape(1, 1, 2, 2)),
            tensor(np.zeros((1, 1, 1, 2))), 0, 0, True, (1, 1)]
    dx, dgrid = D.get_grid_sampler_grad_vmap_rule(Primitive('GridSampler2DGrad'), 2)(
        *((arg, None) for arg in args))
    assert dx[1] is dgrid[1] is None
    assert_tensor(dx[0], np.full((1, 1, 2, 2), .5))
    assert_tensor(dgrid[0], [[[[1., 2.]]]])


# Private convolution dispatchers are exercised directly for unsupported-operation validation.
@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('factory', [Conv._conv_transpose_vmap_rule, Conv._conv_backprop_filter_vmap_rule])  # pylint: disable=protected-access
def test_convolution_internal_dispatch_rejects_unknown_operation(factory):
    """
    Feature: Remaining vmap dispatch, mutation and numerical contracts
    Description: Check convolution internal dispatch rejects unknown operation.
    Expectation: An unsupported convolution operation raises the explicit dispatch ValueError.
    """
    # Internal helpers are the unit coverage targets here.
    # pylint: disable=protected-access
    prim = P.Conv2D(1, 1)
    data = (tensor(np.ones((2, 1, 1, 2, 2))), 0)
    weight = (tensor(np.ones((1, 1, 1, 1))), None)
    with pytest.raises(ValueError, match='Unsupported the operation'):
        factory(prim, 2, data, weight, ((1, 1, 2, 2), None), ['UnknownConv', 1, 'NCHW'])


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_one_hot_rejects_nonconstant_axis_metadata(monkeypatch):
    """
    Feature: Remaining vmap dispatch, mutation and numerical contracts
    Description: Check one hot rejects nonconstant axis metadata.
    Expectation: A single constantness query rejects a dynamic OneHot axis with ValueError.
    """
    axis, calls = ms.mutable(-1), []
    def isconstant(value):
        assert value is axis
        calls.append(value)
        return False
    monkeypatch.setattr(V.F, 'isconstant', isconstant)
    with pytest.raises(ValueError, match='must be constant'):
        V.get_one_hot_vmap_rule(P.OneHot(), 2)((tensor([[0, 1], [1, 2]], np.int32), 0),
            (3, None), (tensor(1.), None), (tensor(0.), None), (axis, None))
    assert len(calls) == 1


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_index_add_rejects_mapped_updates_to_unmapped_reference():
    """
    Feature: Remaining vmap dispatch, mutation and numerical contracts
    Description: Check index add rejects mapped updates to unmapped reference.
    Expectation: Mapped updates against an unmapped mutable reference raise ValueError before mutation.
    """
    rule = W.get_index_add_vmap_rule(P.IndexAdd(0), 2)
    with pytest.raises(ValueError, match="batch dimension of 'x'.*None"):
        rule((tensor([1., 2.]), None), (tensor([0], np.int32), None),
             (tensor([[3.], [4.]]), 0), UMonad())


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('axes', [(0, None), (0, 1), (0, 0)])
def test_index_add_aligns_real_updates_at_graph_executor_boundary(monkeypatch, axes):
    """
    Feature: Remaining vmap dispatch, mutation and numerical contracts
    Description: Check index add aligns real updates at graph executor boundary.
    Expectation: One executor call receives aligned updates, mutates the original reference and returns matching
        values with axis zero.
    """
    data = ms.Parameter(tensor([[1., 2., 3.], [4., 5., 6.]]), name='index_add_ref')
    update = np.array([[7.], [8.]], np.float32)
    update_arg = tensor(update[0] if axes[1] is None else np.moveaxis(update, 0, axes[1]))
    index, monad = tensor([1], np.int32), UMonad()
    original = P.IndexAdd.__call__
    calls = []
    def native(prim, x, indices, y, token):
        assert prim.axis == 1 and x is data and indices is index and token is monad
        expected_update = np.broadcast_to(update[0], update.shape) if axes[1] is None else update
        assert_tensor(y, expected_update)
        calls.append(y)
        return original(prim, x, indices, y)
    monkeypatch.setattr(P.IndexAdd, '__call__', native)
    value, axis = W.get_index_add_vmap_rule(P.IndexAdd(0), 2)((data, axes[0]), (index, None),
                                                            (update_arg, axes[1]), monad)
    expected = np.array([[1., 9., 3.], [4., 12. if axes[1] is None else 13., 6.]], np.float32)
    assert axis == 0 and len(calls) == 1
    assert_tensor(value, expected)
    assert_tensor(data, expected)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.xfail(strict=True, raises=TypeError, reason='tuple output fallback calls zip with keyword unpacking')
def test_general_tuple_output_fallback_expected_values():
    """
    Feature: Remaining vmap dispatch, mutation and numerical contracts
    Description: Check general tuple output fallback expected values.
    Expectation: The fallback should preserve two tuple outputs [[1], [3]] and [[2], [4]], both with mapped axis
        zero.
    """
    result = B.vmap_general_output_process(((tensor([1.]), tensor([2.])), (tensor([3.]), tensor([4.]))))
    assert len(result) == 2
    assert result[0][1] == result[1][1] == 0
    assert_tensor(result[0][0], [[1.], [3.]])
    assert_tensor(result[1][0], [[2.], [4.]])


def resize_native_graph_abi(monkeypatch, primitive, names, expected):
    """Bind the already supplied graph arguments once; execute the real native primitive."""
    wrapper = type(primitive).__call__
    native = Primitive.__call__
    attributes = primitive.attrs.copy()
    calls = []
    def invoke(current, *arguments):
        if current is not primitive:
            return wrapper(current, *arguments)
        assert len(arguments) == 2 + len(names)
        assert arguments[2:] == tuple(getattr(primitive, name) for name in names)
        for value in arguments[:2]:
            assert_tensor(value, expected)
        calls.append(arguments)
        # Generated __call__ would append the same flags a second time in eager mode.
        return native(current, *arguments)
    monkeypatch.setattr(type(primitive), '__call__', invoke)
    return calls, attributes


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_resize_bilinear_gradient_identity_expected_cpu(monkeypatch):
    """
    Feature: Remaining vmap dispatch, mutation and numerical contracts
    Description: Check resize bilinear gradient identity expected CPU.
    Expectation: Identity resizing should preserve gradient values and original primitive attributes after one
        native call.
    """
    data = np.arange(8, dtype=np.float32).reshape(2, 1, 1, 2, 2)
    primitive = G.ResizeBilinearGrad()
    calls, attributes = resize_native_graph_abi(monkeypatch, primitive,
                                               ('align_corners', 'half_pixel_centers'), data.reshape(2, 1, 2, 2))
    result = I.get_resize_bilinear_grad_vmap_rule(primitive, 2)(
        (tensor(data), 0), (tensor(data), 0), (False, None), (False, None))
    value, axis = result
    assert axis == 0
    assert_tensor(value, data)
    assert len(calls) == 1 and primitive.attrs == attributes


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_resize_linear_gradient_identity_expected_cpu(monkeypatch):
    """
    Feature: Remaining vmap dispatch, mutation and numerical contracts
    Description: Check resize linear gradient identity expected CPU.
    Expectation: Identity linear resizing should preserve gradient values and original primitive attributes after
        one native call.
    """
    data = np.arange(8, dtype=np.float32).reshape(2, 1, 1, 4)
    primitive = G.ResizeLinear1DGrad()
    calls, attributes = resize_native_graph_abi(monkeypatch, primitive,
                                               ('coordinate_transformation_mode',), data.reshape(2, 1, 4))
    result = I.get_resize_linear_1d_grad_vmap_rule(primitive, 2, (primitive.coordinate_transformation_mode, None))(
        (tensor(data), 0), (tensor(data), 0))
    value, axis = result
    assert axis == 0
    assert_tensor(value, data)
    assert len(calls) == 1 and primitive.attrs == attributes


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_crop_and_resize_full_images_expected_cpu():
    """
    Feature: Remaining vmap dispatch, mutation and numerical contracts
    Description: Check crop and resize full images expected CPU.
    Expectation: Cropping full-image boxes at native resolution should return the original mapped images with axis
        zero.
    """
    data = np.arange(8, dtype=np.float32).reshape(2, 1, 2, 2, 1)
    value, axis = I.get_crop_and_resize_vmap_rule(IMG.CropAndResize(), 2)(
        (tensor(data), 0), (tensor([[[0., 0., 1., 1.]], [[0., 0., 1., 1.]]]), 0),
        (tensor([[0], [0]], np.int32), 0), ((2, 2), None))
    assert axis == 0
    assert_tensor(value, data)
