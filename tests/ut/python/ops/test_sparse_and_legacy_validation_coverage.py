# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Final sparse/frontend validation and real legacy reduction graph contracts."""
import importlib

import numpy as np
import pytest
import mindspore as ms
from mindspore import _checkparam
from mindspore.ops.function._add_attr_func import AddAttr
from mindspore.ops._utils.arg_dtype_cast import scalar_to_tensor
from mindspore.ops.operations import Abs
from mindspore.ops.function import math_func


@pytest.fixture(scope='module', autouse=True)
def cpu_context():
    previous = {key: ms.get_context(key) for key in ('mode', 'device_target', 'pynative_synchronize')}
    ms.set_context(device_target='CPU', mode=ms.PYNATIVE_MODE, pynative_synchronize=True)
    yield
    ms.set_context(**previous)


def test_empty_coo_shape_special_case_accepts_empty_storage():
    """
    Feature: Sparse frontend and legacy reduction validation
    Description: Check empty COO shape special case accepts empty storage.
    Expectation: Empty index/value storage with a valid dense shape is accepted and returns None.
    """
    assert _checkparam.check_coo_tensor_shape((0,), (0,), (2, 2)) is None


@pytest.mark.parametrize('value', [np.int64(3), np.float32(2), complex(1, 2)])
def test_native_scalar_conversion_rejects_unsupported_scalar_types(value):
    """
    Feature: Sparse frontend and legacy reduction validation
    Description: Check native scalar conversion rejects unsupported scalar types.
    Expectation: NumPy scalar types and complex numbers raise TypeError during scalar-to-Tensor conversion.
    """
    with pytest.raises(TypeError):
        scalar_to_tensor(value)


def test_default_linalg_norm_order_forwards_complete_native_vector_protocol(monkeypatch):
    """
    Feature: Sparse frontend and legacy reduction validation
    Description: Protocol only: the valid default-order branch uses an unavailable CPU vector-norm kernel.
    Expectation: Default order dispatches exactly once with the original float32 vector and arguments (2.0, (0,),
        False, None).
    """
    data, output, calls = ms.Tensor([3., 4.]), object(), []
    def native(*args):
        assert len(args) == 5 and args[0] is data
        assert args[1:] == (2.0, (0,), False, None)
        assert data.dtype == ms.float32 and data.shape == (2,)
        np.testing.assert_array_equal(data.asnumpy(), [3, 4])
        calls.append(args)
        return output
    monkeypatch.setattr(math_func, 'linalg_vector_norm_op', native)
    assert math_func.linalg_norm(data, ord=None, dim=0) is output
    assert len(calls) == 1


@pytest.mark.parametrize('kind', ['primitive', 'non_callable'])
def test_attribute_decorator_rejects_unsupported_function_kind(kind):
    """
    Feature: Sparse frontend and legacy reduction validation
    Description: Check attribute decorator rejects unsupported function kind.
    Expectation: Primitive instances and non-callable values raise TypeError for unsupported decorator inputs.
    """
    value = Abs() if kind == 'primitive' else 123
    with pytest.raises(TypeError, match='callable type except Primitive'):
        AddAttr()(value)


def legacy(name):
    pytest.importorskip('te', reason='Legacy validation requires the optional CANN TE SDK')
    return importlib.import_module('mindspore.ops._op_impl._custom_op.' + name)


def desc(shape, dtype='float32', storage='NCHW'):
    return {'shape': shape, 'ori_shape': shape, 'dtype': dtype, 'format': storage, 'ori_format': 'NCHW'}


def test_fold_gradient_reduce_rejects_integer_input_before_placeholder_creation():
    """
    Feature: Sparse frontend and legacy reduction validation
    Description: Check fold gradient reduce rejects integer input before placeholder creation.
    Expectation: Integer input metadata raises the dtype RuntimeError before graph placeholders are created.
    """
    invalid = desc((2, 2, 2, 2), 'int32')
    with pytest.raises(RuntimeError, match='Dtype of input'):
        legacy('batchnorm_fold2_grad_reduce').batchnorm_fold2_grad_reduce(invalid, invalid, {}, {})


@pytest.mark.parametrize('reduce', [False, True])
def test_correction_gradient_rejects_noncanonical_storage_format(reduce):
    """
    Feature: Sparse frontend and legacy reduction validation
    Description: Check correction gradient rejects noncanonical storage format.
    Expectation: Both correction gradient paths reject lowercase nchw metadata with the format diagnostic.
    """
    data, statistic = desc((2, 2, 2, 2), storage='nchw'), desc((2,))
    module = legacy('correction_mul_grad')
    with pytest.raises(RuntimeError, match=r'data_format\(NCHW\) must same as ori_format'):
        if reduce:
            module.correction_mul_grad_reduce(data, statistic, 1)
        else:
            module.correction_mul_grad(data, data, statistic, statistic, data, data, 1)


def test_fold_reduction_preserves_singleton_channel_alignment_in_real_llvm_graph():
    """
    Feature: Sparse frontend and legacy reduction validation
    Description: Check fold reduction preserves singleton channel alignment in real llvm graph.
    Expectation: LLVM execution retains (1, 2, 1) float32 outputs and returns channel sums [6, 15] and [8, 62].
    """
    module = legacy('batchnorm_fold2_grad_reduce')
    # The optional SDK is probed by legacy() before importing its graph API.
    from te import tvm  # pylint: disable=import-outside-toplevel
    if not tvm.runtime.enabled('llvm'):
        pytest.skip('CANN TE SDK has no LLVM host backend')
    shape = (1, 2, 1, 3)
    inputs = [tvm.placeholder(shape, name=name, dtype='float32') for name in ('sensitivity', 'input')]
    outputs = module.batchnorm_fold2_grad_reduce_compute(*inputs, desc(shape))
    assert all(tuple(int(dim) for dim in output.shape) == (1, 2, 1) and str(output.dtype) == 'float32'
               for output in outputs)
    native = tvm.build(tvm.create_schedule([output.op for output in outputs]), inputs + outputs,
                       target='llvm', name='coverage_singleton_channel_reduce')
    arguments = [tvm.nd.array(np.arange(1, 7, dtype=np.float32).reshape(shape)),
                 tvm.nd.array(np.arange(6, dtype=np.float32).reshape(shape))]
    results = [tvm.nd.empty((1, 2, 1), 'float32') for _ in outputs]
    native(*(arguments + results))
    np.testing.assert_array_equal(results[0].numpy(), [[[6], [15]]])
    np.testing.assert_array_equal(results[1].numpy(), [[[8], [62]]])


@pytest.mark.xfail(strict=True, raises=ModuleNotFoundError,
                   reason='CANN 9.1 moved the ND scheduler; legacy impl.bn_training_reduce import is absent')
def test_legacy_nd_reduction_entry_resolves_the_real_sdk_scheduler(monkeypatch):
    """
    Feature: Sparse frontend and legacy reduction validation
    Description: Check legacy nd reduction entry resolves the real sdk scheduler.
    Expectation: The entry should reach one intercepted cce build; strict xfail identifies the missing legacy ND
        scheduler import.
    """
    module = legacy('batchnorm_fold2_grad_reduce')
    calls = []
    def binary_emission(schedule, tensors, target, name):
        assert schedule is not None and target == 'cce' and name == 'coverage_legacy_nd'
        assert len(tensors) == 3
        calls.append((schedule, tensors, target, name))
    # A future import repair must still keep this test from emitting native accelerator binaries.
    monkeypatch.setattr(module.tvm, 'build', binary_emission)
    description = desc((2, 2, 2, 2))
    try:
        module.batchnorm_fold2_grad_reduce(description, description, {}, {}, kernel_name='coverage_legacy_nd')
    except ModuleNotFoundError as error:
        assert 'impl.bn_training_reduce' in str(error)
        raise
    assert len(calls) == 1
