# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Frontend device policies and real SDK callback classification; no device execution."""
import copy

import pytest
from mindspore import context, float32
from mindspore.ops import deprecated
from mindspore.ops.operations import array_ops, nn_ops, custom_ops


@pytest.fixture
def ascend_policy(monkeypatch):
    """Only frontend device discovery changes; the real runtime target remains untouched."""
    original = context.get_context
    target = original('device_target')
    def query(key):
        return 'Ascend' if key == 'device_target' else original(key)
    monkeypatch.setattr(context, 'get_context', query)
    yield
    assert original('device_target') == target


@pytest.mark.parametrize('constructor', [array_ops.SpaceToBatchND, deprecated.BatchToSpaceND])
def test_ascend_space_batch_policy_requires_two_spatial_axes(ascend_policy, constructor):
    with pytest.raises(ValueError, match='block_shape length'):
        constructor([2, 2, 2], [[0, 0]] * 3)
    operation = constructor([2, 3], [[0, 0], [0, 0]])
    assert operation.block_shape == [2, 3]


@pytest.mark.parametrize('constructor', [nn_ops.Conv3D, nn_ops.Conv3DTranspose])
def test_ascend_convolution_policy_rejects_multiple_groups(ascend_policy, constructor):
    options = {'in_channel': 4} if constructor is nn_ops.Conv3DTranspose else {}
    with pytest.raises(ValueError, match='group = 1'):
        constructor(out_channel=4, kernel_size=3, group=2, **options)
    assert constructor(out_channel=4, kernel_size=3, group=1, **options).group == 1


@pytest.mark.parametrize('supported', [False, True])
def test_allfinite_software_capability_policy(ascend_policy, monkeypatch, supported):
    calls = []
    class SoftwareVersion:
        def __init__(self, path):
            assert path is None
        def check_custom_version(self):
            calls.append('check_custom_version')
            return supported
    monkeypatch.setattr(nn_ops, 'AscendEnvChecker', SoftwareVersion)
    if supported:
        assert nn_ops.AllFinite().input_names == ['gradients']
    else:
        with pytest.raises(RuntimeError, match='does not support AllFinite'):
            nn_ops.AllFinite()
    assert calls == ['check_custom_version']


def elementwise_expression(data):
    """A real TE expression used as compiler callback metadata."""
    from te import tvm
    return tvm.compute(data.shape, lambda i: data[i] + 1, name='policy_elementwise')


def statement_expression(data):
    """A real TVM statement builder used as compiler callback metadata."""
    from te import tvm
    builder = tvm.tir.ir_builder.create()
    with builder.for_range(0, data.shape[0], name='index') as index:
        builder.emit(tvm.tir.Evaluate(data[index]))
    return builder.get()


@pytest.mark.parametrize('callback,kind', [(elementwise_expression, 'tvm_compute'),
                                         (statement_expression, 'ir_builder')])
def test_custom_classifies_real_sdk_callback_source(monkeypatch, tmp_path, callback, kind):
    sdk = pytest.importorskip('te', reason='Callback classification fixture uses the optional real TE SDK')
    for name in ('compiled_bin', 'registered_func', 'attr_dict', 'tbe_path_checked',
                 'tbe_path_failed', 'op_path_in_cache', 'custom_aot_warning'):
        monkeypatch.setattr(custom_ops.Custom, name, copy.deepcopy(getattr(custom_ops.Custom, name)))
    monkeypatch.setenv('MS_COMPILER_CACHE_PATH', str(tmp_path))
    data = sdk.tvm.placeholder((2,), name='input', dtype='float32')
    expression = callback(data)
    if kind == 'ir_builder':
        assert isinstance(expression, sdk.tvm.tir.Stmt)
    else:
        assert tuple(int(dim) for dim in expression.shape) == (2,)
        assert str(expression.dtype) == 'float32'
    operation = custom_ops.Custom(callback, (2,), float32, 'akg')
    assert operation.func_type == kind
    assert operation.func is callback
    assert operation.func_source_str == operation.get_attr_dict()['func_source_str']
    assert ('ir_builder' if kind == 'ir_builder' else 'compute') in operation.func_source_str
