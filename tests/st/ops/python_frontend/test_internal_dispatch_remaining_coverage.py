# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Final internal dispatch audit: real CPU graphs and bounded external failure/codegen protocols."""
# Direct private-helper calls intentionally cover the internal rules under test.
# pylint: disable=protected-access
import copy
import importlib
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import mindspore as ms
from mindspore import dataset
from mindspore import _c_expression as native
from mindspore.experimental import MapParameter
from mindspore.ops import CustomRegOp, DataType
from mindspore.ops._utils import utils
from mindspore.ops.composite.multitype_ops import getitem_impl as getitem
from mindspore.ops.function import nn_func as nn
from mindspore.ops.function import random_func as random_functions
from mindspore.ops.operations import custom_ops as custom
from tests.mark_utils import arg_mark


@pytest.fixture(scope='module', autouse=True)
def cpu_context():
    previous = {key: ms.get_context(key) for key in ('mode', 'device_target', 'pynative_synchronize')}
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target='CPU', pynative_synchronize=True)
    yield
    ms.set_context(**previous)


@pytest.fixture(autouse=True)
def preserve_rng():
    """Restore all exposed RNG states plus graph per-op seed counters, including the rrelu graph."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    dataset_seed = dataset.config.get_seed()
    state = ms.get_rng_state()
    seed = ms.get_seed()
    manual_seed = random_functions.default_generator._generator.attrs['manual_seed']
    seed_module = importlib.import_module('mindspore.common.seed')
    op_seeds = seed_module._KERNEL_SEED.copy()
    mode = ms.get_context('mode')
    yield
    ms.set_context(mode=ms.PYNATIVE_MODE)
    ms.set_rng_state(state)
    random_functions.default_generator._generator.add_prim_attr('manual_seed', manual_seed)
    dataset.config.set_seed(dataset_seed)
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    seed_module._GLOBAL_SEED = seed
    seed_module._KERNEL_SEED.clear()
    seed_module._KERNEL_SEED.update(op_seeds)
    ms.set_context(mode=mode)


def t(value, dtype=ms.float32):
    return ms.Tensor(value, dtype)


def check(value, expected, dtype=ms.float32):
    assert isinstance(value, ms.Tensor) and value.dtype == dtype
    assert value.shape == np.asarray(expected).shape
    np.testing.assert_allclose(value.asnumpy(), expected, rtol=1e-6, atol=1e-6)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('error_class', [OSError, TypeError])
def test_url_parser_maps_external_regex_failure_without_network(monkeypatch, error_class):
    """Feature: Internal frontend dispatch and external boundary contracts.
    Description: Url parser maps external regex failure without network.
    Expectation: The regex boundary is called once with the expected IPv4 pattern, and RuntimeError preserves
        the original exception as its cause.
    """
    url = 'tcp://127.0.0.1:4321'
    failure = error_class('isolated regex engine failure')
    calls = []
    expected_pattern = r'^((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'

    def parse_boundary(pattern, ip):
        assert pattern == expected_pattern and ip == '127.0.0.1'
        calls.append((pattern, ip))
        raise failure

    # Module-local regex dependency, never a URL function replacement or socket operation.
    monkeypatch.setattr(utils, 're', SimpleNamespace(match=parse_boundary))
    with pytest.raises(RuntimeError, match='Unexpected error parsing TCP URL: isolated regex engine failure') as result:
        utils._parse_tcp_url_for_ipv4(url)
    assert result.value.__cause__ is failure
    assert calls == [(expected_pattern, '127.0.0.1')]


@ms.jit
def compiled_tuple_selection(first, second, index):
    return getitem._tuple_getitem_by_tensor((first, second), index)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('index,index_dtype,expected', [(0, ms.int32, [2.0, 3.0]), (-1, ms.int64, [5.0, 7.0])])
def test_tuple_tensor_dispatch_real_cpu_compiled_graph(index, index_dtype, expected):
    """Feature: Internal frontend dispatch and external boundary contracts.
    Description: Tuple tensor dispatch real CPU compiled graph.
    Expectation: The compiled tensor index selects the expected tuple element for both supported index dtypes.
    """
    check(compiled_tuple_selection(t([2.0, 3.0]), t([5.0, 7.0]), t(index, index_dtype)), expected)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_map_tensor_get_real_storage_native_protocol(monkeypatch):
    """Feature: Internal frontend dispatch and external boundary contracts.
    Description: Protocol only after the CPU probe established that native MapTensorGet is unavailable.
    Expectation: MapTensorGet receives the genuine map handle, lookup keys and insert-default attribute once and
        returns the supplied lookup tensor unchanged.
    """
    key_storage = t([1], ms.int32)
    value_storage = t([[2.0, 3.0]])
    parameter = MapParameter(key_tensor=key_storage, value_tensor=value_storage, default_value='zeros')
    mapping = parameter._map_tensor
    assert isinstance(mapping, native.MapTensor_) and parameter.key_dtype == ms.int32
    assert parameter.value_dtype == ms.float32 and parameter.default_value == 'zeros'
    keys = t([1, 3], ms.int32)
    provided = t([[2.0, 3.0], [0.0, 0.0]])
    calls = []

    def native_lookup(primitive, map_arg, key_arg):
        assert primitive.name == 'MapTensorGet' and primitive.attrs['insert_default_value'] is True
        assert map_arg is mapping and key_arg is keys
        check(key_arg, [1, 3], ms.int32)
        calls.append((map_arg, key_arg))
        return provided

    monkeypatch.setattr(getitem._map_tensor_ops.MapTensorGet, '__call__', native_lookup)
    assert getitem._map_tensor_getitem(mapping, keys) is provided
    assert len(calls) == 1
    # This checks the provided result's declared contract, not native lookup/insertion numerics.
    check(provided, [[2.0, 3.0], [0.0, 0.0]])


class DynamicRrelu(ms.nn.Cell):
    def construct(self, value):
        return nn.rrelu(value, 0.25, 0.25)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_rrelu_natural_dynamic_shape_real_cpu_graph():
    """Feature: Internal frontend dispatch and external boundary contracts.
    Description: RReLU natural dynamic shape real CPU graph.
    Expectation: Dynamic graph execution scales negative values by 0.25 and preserves zero and positive values.
    """
    network = DynamicRrelu()
    network.set_inputs(ms.Tensor(shape=[None, None], dtype=ms.float32))
    ms.set_context(mode=ms.GRAPH_MODE)
    check(network(t([[-2.0, 0.0, 3.0], [4.0, -1.0, 2.0]])), [[-0.5, 0.0, 3.0], [4.0, -0.25, 2.0]])


def same_shape(shape):
    return shape


def same_dtype(dtype):
    return dtype


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('api_type_form', ['list', 'invalid'])
def test_custom_workspace_real_constructor_codegen_external_protocol(monkeypatch, tmp_path, api_type_form):
    """Feature: Internal frontend dispatch and external boundary contracts.
    Description: Real Custom AOT construction/code generation; no compilation, loading or Ascend execution.
    Expectation: Real Custom construction emits the expected callback source and registration metadata through
        one supplied compiler request without producing a shared object.
    """
    api_name = 'aclnnDispatchAuditIdentity'
    api_types = ['aclTensor*', 'aclTensor*', 'uint64_t*', 'aclOpExecutor**']
    assert ms.get_context('device_target') == 'CPU' and custom.GlobalComm.INITED is False
    for name in (
        'registered_func',
        'attr_dict',
        'compiled_bin',
        'tbe_path_checked',
        'tbe_path_failed',
        'op_path_in_cache',
    ):
        monkeypatch.setattr(custom.Custom, name, copy.deepcopy(getattr(custom.Custom, name)))
    monkeypatch.setattr(custom.Custom, 'custom_aot_warning', custom.Custom.custom_aot_warning)
    monkeypatch.setenv('MS_COMPILER_CACHE_PATH', str(tmp_path))
    queries = []
    registration = []
    schema = []
    compilations = []

    def frontend_target(key):
        assert key == 'device_target'
        queries.append(key)
        return 'Ascend'

    monkeypatch.setattr(custom, 'context', SimpleNamespace(get_context=frontend_target))
    expected_registration = {
        'op_name': 'Custom_DispatchAuditIdentity_' + api_name,
        'inputs': [{'index': 0, 'name': 'x', 'paramType': 'required'}],
        'outputs': [{'index': 0, 'name': 'y', 'paramType': 'required'}],
        'attr': [],
        'fusion_type': 'OPAQUE',
        'dtype_format': [[['float32', 'DefaultFormat'], ['float32', 'DefaultFormat']]],
        'target': 'Ascend',
        'imply_type': 'BiSheng',
    }

    class RegistryBoundary:
        def reg_op(self, payload, path):
            info = json.loads(payload)
            assert info == expected_registration and path == ''
            registration.append((info, path))
            return True

    def header_schema(provider):
        assert provider.ori_op_name == api_name and provider.prefix_op_name == api_name
        schema.append(provider.ori_op_name)
        return api_types.copy()

    expected_code = '''
#include <vector>
#include "acl/acl_base.h"
typedef struct aclOpExecutor aclOpExecutor;
typedef struct aclTensor aclTensor;
typedef struct aclScalar aclScalar;
typedef struct aclIntArray aclIntArray;
typedef struct aclFloatArray aclFloatArray;
typedef struct aclBoolArray aclBoolArray;
typedef struct aclTensorList aclTensorList;
typedef struct aclScalarList aclScalarList;
extern "C" int aclnnDispatchAuditIdentityGetWorkSpaceSize(void *func_ptr, std::vector<void *> inputs, std::vector<void *> outputs,
                           uint64_t *workspace_size, aclOpExecutor **executor) {
  using FuncType = int (*)(aclTensor*, aclTensor*, uint64_t *, aclOpExecutor **);
  auto func = reinterpret_cast<FuncType>(func_ptr);
  aclTensor* input0 = static_cast<aclTensor*>(inputs[0]);
  aclTensor* output0 = static_cast<aclTensor*>(outputs[0]);
  return func(input0, output0, workspace_size, executor);
}'''
    callback = tmp_path / api_name / (api_name + '_callback.cc')

    def external_compiler(path):
        assert Path(path) == callback and callback.is_file()
        contents = callback.read_text()
        assert ''.join(contents.split()) == ''.join(expected_code.split())
        assert ms.get_context('device_target') == 'CPU'
        compilations.append((path, contents))
        return str(callback.with_suffix('.so'))

    monkeypatch.setattr(custom, 'Oplib', RegistryBoundary)
    monkeypatch.setattr(custom.CustomInfoGenerator, 'get_aclnn_api_types', header_schema)
    monkeypatch.setattr(custom, '_compile_aot', external_compiler)
    info = (
        CustomRegOp('DispatchAuditIdentity')
        .input(0, 'x')
        .output(0, 'y')
        .dtype_format(DataType.F32_Default, DataType.F32_Default)
        .target('Ascend')
        .get_op_info()
    )
    operation = custom.Custom(api_name, same_shape, same_dtype, 'aot', reg_info=info)
    assert operation.is_ascend_c is True
    assert isinstance(operation.custom_pyboost, custom._CustomExt)
    assert operation.custom_pyboost.func == api_name
    assert operation.custom_pyboost.out_shape is same_shape and operation.custom_pyboost.out_dtype is same_dtype
    assert queries == ['device_target'] and len(registration) == 1 and schema == [api_name]
    assert len(compilations) == 1
    expected_callback = str(callback.with_suffix('.so')) + ':' + api_name
    assert operation.attrs['custom_callback_func'] == expected_callback
    assert operation.attrs['custom_inputs_type'] == api_types[:-2]
    if api_type_form == 'list':
        operation._generate_get_worspace_size_func_by_types(api_types)
        assert len(compilations) == 2 and compilations[0] == compilations[1]
        assert operation.attrs['custom_callback_func'] == expected_callback
    else:
        with pytest.raises(RuntimeError, match="Unsupported type: <class 'int'>, support type is list or string"):
            operation._generate_get_worspace_size_func_by_types(3)
        assert len(compilations) == 1
    assert not callback.with_suffix('.so').exists()
    assert ms.get_context('device_target') == 'CPU'
