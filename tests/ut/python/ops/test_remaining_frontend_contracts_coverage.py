# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Frontend validation, constant folding and platform registration contracts."""
# Unit coverage intentionally exercises internal Python inference and wrapper helpers.
# pylint: disable=protected-access
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


@pytest.fixture(scope='module', autouse=True, name='cpu_context')
def fixture_cpu_context():
    previous = {key: ms.get_context(key) for key in ('mode', 'device_target')}
    ms.set_context(device_target='CPU', mode=ms.PYNATIVE_MODE)
    yield
    ms.set_context(**previous)


@pytest.mark.parametrize('dtype', [ms.int32, ms.int64, ms.float32])
def test_constant_range_uses_requested_dtype(dtype):
    """
    Feature: Constant range uses requested dtype.
    Description: Check constant range uses requested dtype.
    Expectation: The constant range returns [5, 3, 1, -1], shape (4,), and the requested dtype.
    """
    result = utils.range_op(5, -2, -2, dtype)
    assert result.dtype == dtype and result.shape == (4,)
    np.testing.assert_array_equal(result.asnumpy(), [5, 3, 1, -1])


@pytest.mark.parametrize('value', [2, 2.5])
def test_scalar_argument_normalization_returns_a_scalar_tensor(value):
    """
    Feature: Scalar argument normalization returns a scalar tensor.
    Description: Check scalar argument normalization returns a scalar tensor.
    Expectation: Scalar normalization returns a zero-dimensional tensor retaining the original numeric value.
    """
    result = arg_dtype_cast.scalar_to_tensor(value)
    assert result.shape == ()
    np.testing.assert_array_equal(result.asnumpy(), value)


def test_optional_enum_and_negative_sequence_values():
    """
    Feature: Optional enum and negative sequence values.
    Description: Check optional enum and negative sequence values.
    Expectation: An absent optional enum remains None and a supported negative sequence value remains (-2,).
    """
    assert arg_dtype_cast.StringToEnum()('Interpolation', 'mode', None) is None
    assert arg_dtype_cast.normalize_int_sequence('Example', 'shape', (-2,)) == (-2,)


@pytest.mark.parametrize('constructor', [random_ops.Gamma, random_ops.Poisson])
def test_random_frontend_rejects_unresolved_shape_value(constructor):
    """
    Feature: Random frontend rejects unresolved shape value.
    Description: Check random frontend rejects unresolved shape value.
    Expectation: Invalid random frontend unresolved shape value raise ValueError. Diagnostics identify
        shape.*cannot be None.
    """
    shape = {'shape': (1,), 'dtype': ms.int64, 'value': None}
    argument = {'shape': (), 'dtype': ms.float32, 'value': None}
    args = (shape, argument, argument) if constructor is random_ops.Gamma else (shape, argument)
    with pytest.raises(ValueError, match='shape.*cannot be None'):
        constructor().__infer__(*args)


def test_registration_schema_checks_parameter_container_and_arity():
    """
    Feature: Registration schema checks parameter container and arity.
    Description: Check registration schema checks parameter container and arity.
    Expectation: Malformed parameter containers, metadata sizes, value-depend fields, and dtype-format arities
        raise their specified schema errors.
    """
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
    """
    Feature: Scalar boolean negation is a python boolean.
    Description: Check scalar boolean negation is a python boolean.
    Expectation: Boolean negation returns the exact Python singleton corresponding to the opposite input value.
    """
    assert bool_not()(value) is (not value)


@pytest.mark.parametrize('constructor,args', [(control_ops.GeSwitch, (1, True)),
                                             (control_ops.Merge, ((1, 2),))])
def test_graph_control_operators_reject_direct_python_execution(constructor, args):
    """
    Feature: Graph control operators reject direct python execution.
    Description: Check graph control operators reject direct python execution.
    Expectation: Invalid graph control operators reject direct python execution raise NotImplementedError.
    """
    with pytest.raises(NotImplementedError):
        constructor()(*args)


def test_map_lookup_constructor_records_memory_side_effect_and_default_policy():
    """
    Feature: Map lookup constructor records memory side effect and default policy.
    Description: Check map lookup constructor records memory side effect and default policy.
    Expectation: Map lookup records the expected input/output names, a memory side effect, and disabled default
        insertion.
    """
    lookup = MapTensorGet(insert_default_value=False)
    assert lookup.input_names == ['map_tensor', 'key_tensor']
    assert lookup.output_names == ['output']
    assert lookup.side_effect_mem is True
    assert lookup.insert_default_value is False


@pytest.mark.parametrize('system', ['Windows', 'Darwin', 'Linux'])
def test_platform_reduction_registration_policy_is_isolated(monkeypatch, system):
    """
    Feature: Platform reduction registration policy is isolated.
    Description: Check platform reduction registration policy is isolated.
    Expectation: Registration invokes exactly the expected platform-specific reduction policy calls.
    """
    calls = []
    monkeypatch.setattr(reductions, 'platform', SimpleNamespace(system=lambda: system))
    monkeypatch.setattr(reductions.reduction, 'register', lambda kind, function: calls.append((kind, function)))
    reductions.init_reductions()
    expected = [] if system != 'Linux' else [(reductions.StoragePy, reductions.reduce_storage),
                                           (reductions.UntypedStorage, reductions.reduce_storage),
                                           (reductions.Tensor, reductions.reduce_tensor)]
    assert calls == expected


def test_reshard_frontend_retains_layout_and_tensor_identity():
    """
    Feature: Reshard frontend retains layout and tensor identity.
    Description: Check reshard frontend retains layout and tensor identity.
    Expectation: Reshard records input strategy ((2,),) and returns the original tensor by identity.
    """
    layout = Layout((2,), ('data',))('data')
    operation = Reshard((layout,), (layout,), ((2,),))
    data = ms.Tensor([1., 2.])
    assert operation.in_strategy == ((2,),)
    assert operation(data) is data


def test_bfloat16_concat_constant_folding_preserves_dtype():
    """
    Feature: Bfloat16 concat constant folding preserves dtype.
    Description: Check bfloat16 concat constant folding preserves dtype.
    Expectation: Concatenation folds to [1, 2, 3, 4] with bfloat16 dtype and shape (4,).
    """
    inputs = [ms.Tensor([1., 2.], ms.bfloat16), ms.Tensor([3., 4.], ms.bfloat16)]
    result = ops_def.infer_value_for_Concat(inputs, 0)
    assert result.dtype == ms.bfloat16 and result.shape == (4,)
    np.testing.assert_array_equal(result.float().asnumpy(), [1., 2., 3., 4.])


def test_legacy_scalar_cast_rejects_unsupported_destination_type():
    """
    Feature: Legacy scalar cast rejects unsupported destination type.
    Description: Check legacy scalar cast rejects unsupported destination type.
    Expectation: Invalid legacy scalar cast unsupported destination type raise ValueError. Diagnostics identify
        supported type.
    """
    with pytest.raises(ValueError, match='supported type'):
        ScalarCast()(2, ms.int32)


def test_tensor_report_warns_about_unsupported_eager_execution(monkeypatch):
    """
    Feature: Tensor report warns about unsupported eager execution.
    Description: Check tensor report warns about unsupported eager execution.
    Expectation: Eager TensorReport returns None and emits exactly its documented unsupported-mode warning.
    """
    # Import at the test boundary so fixture isolation is established first.
    # pylint: disable-next=import-outside-toplevel
    from mindspore.ops.operations.manually_defined import _inner
    messages = []
    monkeypatch.setattr(_inner.logger, 'warning', messages.append)
    assert TensorReport()('unused', ms.Tensor([1.])) is None
    assert messages == ["TensorReport doesn't support pynative mode."]
