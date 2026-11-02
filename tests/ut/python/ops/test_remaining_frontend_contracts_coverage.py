# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Frontend validation, constant folding and platform registration contracts."""
from types import SimpleNamespace
import importlib

import numpy as np
import pytest
import mindspore as ms
from mindspore.multiprocessing import reductions
from mindspore.ops._utils import arg_dtype_cast, utils
from mindspore.ops.operations import control_ops, random_ops
from mindspore.ops.operations._map_tensor_ops import MapTensorGet
from mindspore.ops.operations._scalar_ops import bool_not
from mindspore.ops.operations.manually_defined._inner import ScalarCast, TensorReport
from mindspore.ops.operations.manually_defined import ops_def
from mindspore.ops.operations.reshard_ops import Reshard
from mindspore.parallel.shard import Layout

registration = importlib.import_module('mindspore.ops.op_info_register')


@pytest.fixture(scope='module', autouse=True)
def cpu_context():
    previous = {key: ms.get_context(key) for key in ('mode', 'device_target')}
    ms.set_context(device_target='CPU', mode=ms.PYNATIVE_MODE)
    yield
    ms.set_context(**previous)


@pytest.mark.parametrize('dtype', [ms.int32, ms.int64, ms.float32])
def test_constant_range_uses_requested_dtype(dtype):
    result = utils.range_op(5, -2, -2, dtype)
    assert result.dtype == dtype and result.shape == (4,)
    np.testing.assert_array_equal(result.asnumpy(), [5, 3, 1, -1])


@pytest.mark.parametrize('value', [2, 2.5])
def test_scalar_argument_normalization_returns_a_scalar_tensor(value):
    result = arg_dtype_cast.scalar_to_tensor(value)
    assert result.shape == ()
    np.testing.assert_array_equal(result.asnumpy(), value)


def test_optional_enum_and_negative_sequence_values():
    assert arg_dtype_cast.StringToEnum()('Interpolation', 'mode', None) is None
    assert arg_dtype_cast.normalize_int_sequence('Example', 'shape', (-2,)) == (-2,)


@pytest.mark.parametrize('constructor', [random_ops.Gamma, random_ops.Poisson])
def test_random_frontend_rejects_unresolved_shape_value(constructor):
    shape = {'shape': (1,), 'dtype': ms.int64, 'value': None}
    argument = {'shape': (), 'dtype': ms.float32, 'value': None}
    args = (shape, argument, argument) if constructor is random_ops.Gamma else (shape, argument)
    with pytest.raises(ValueError, match='shape.*cannot be None'):
        constructor().__infer__(*args)


def test_registration_schema_checks_parameter_container_and_arity():
    schema = registration.TBERegOp('CoverageSchema')
    with pytest.raises(TypeError, match='list type'):
        schema._check_param((), [], [], {})
    with pytest.raises(ValueError, match='size'):
        schema._check_param([1], [], [lambda value: True], {})
    with pytest.raises(ValueError, match='value_depend'):
        schema.input(0, 'input', value_depend='invalid')
    custom = registration.CustomRegOp('CoverageCustom').input(0, 'x').output(0, 'y')
    with pytest.raises(ValueError, match='sum of input tensors and output tensors'):
        custom.dtype_format(registration.DataType.F32_Default)


@pytest.mark.parametrize('value', [False, True])
def test_scalar_boolean_negation_is_a_python_boolean(value):
    assert bool_not()(value) is (not value)


@pytest.mark.parametrize('constructor,args', [(control_ops.GeSwitch, (1, True)),
                                             (control_ops.Merge, ((1, 2),))])
def test_graph_control_operators_reject_direct_python_execution(constructor, args):
    with pytest.raises(NotImplementedError):
        constructor()(*args)


def test_map_lookup_constructor_records_memory_side_effect_and_default_policy():
    lookup = MapTensorGet(insert_default_value=False)
    assert lookup.input_names == ['map_tensor', 'key_tensor']
    assert lookup.output_names == ['output']
    assert lookup.side_effect_mem is True
    assert lookup.insert_default_value is False


@pytest.mark.parametrize('system', ['Windows', 'Darwin', 'Linux'])
def test_platform_reduction_registration_policy_is_isolated(monkeypatch, system):
    calls = []
    monkeypatch.setattr(reductions, 'platform', SimpleNamespace(system=lambda: system))
    monkeypatch.setattr(reductions.reduction, 'register', lambda kind, function: calls.append((kind, function)))
    reductions.init_reductions()
    expected = [] if system != 'Linux' else [(reductions.StoragePy, reductions.reduce_storage),
                                           (reductions.UntypedStorage, reductions.reduce_storage),
                                           (reductions.Tensor, reductions.reduce_tensor)]
    assert calls == expected


def test_reshard_frontend_retains_layout_and_tensor_identity():
    layout = Layout((2,), ('data',))('data')
    operation = Reshard((layout,), (layout,), ((2,),))
    data = ms.Tensor([1., 2.])
    assert operation.in_strategy == ((2,),)
    assert operation(data) is data


def test_bfloat16_concat_constant_folding_preserves_dtype():
    inputs = [ms.Tensor([1., 2.], ms.bfloat16), ms.Tensor([3., 4.], ms.bfloat16)]
    result = ops_def.infer_value_for_Concat(inputs, 0)
    assert result.dtype == ms.bfloat16 and result.shape == (4,)
    np.testing.assert_array_equal(result.float().asnumpy(), [1., 2., 3., 4.])


def test_legacy_scalar_cast_rejects_unsupported_destination_type():
    with pytest.raises(ValueError, match='supported type'):
        ScalarCast()(2, ms.int32)


def test_tensor_report_warns_about_unsupported_eager_execution(monkeypatch):
    from mindspore.ops.operations.manually_defined import _inner
    messages = []
    monkeypatch.setattr(_inner.logger, 'warning', messages.append)
    assert TensorReport()('unused', ms.Tensor([1.])) is None
    assert messages == ["TensorReport doesn't support pynative mode."]
