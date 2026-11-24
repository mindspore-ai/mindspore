# Copyright 2026 Huawei Technologies Co., Ltd
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
"""Validate operator registration payloads and native registry boundaries."""
import importlib
import json
from pathlib import Path

import pytest

# The package also exports the same-named decorator; resolve the actual module.
registration = importlib.import_module('mindspore.ops.op_info_register')


@pytest.mark.parametrize('name', ['', '   ', None, 123])
def test_registration_requires_nonempty_operator_name(name):
    """
    Feature: Registration requires nonempty operator name.
    Description: Invalid operator names cannot enter the registry namespace.
    Expectation: Invalid operator names cannot enter the registry namespace.
    """
    with pytest.raises(ValueError, match='op name'):
        registration.RegOp(name)


def test_tbe_registration_serializes_dynamic_capabilities():
    """
    Feature: Tbe registration serializes dynamic capabilities.
    Description: Dynamic capabilities and frontend mappings must survive schema serialization.
    Expectation: Dynamic capabilities and frontend mappings must survive schema serialization.
    """
    info = (registration.TBERegOp('CoverageDynamic')
            .input(0, 'x').output(0, 'y')
            .unknown_shape_formats(['ND', 'NCHW'])
            .dynamic_rank_support(True).real_input_index([1, 0]).input_to_attr_index([2])
            .reshape_type('CH').dynamic_shape(True).dynamic_compile_static(True)
            .need_check_supported(True).is_dynamic_format(True).op_pattern('broadcast')
            .dtype_format(('float32', 'DefaultFormat', 'Tensor'), ('float32', 'DefaultFormat', 'Tensor'))
            .get_op_info())
    assert info['op_name'] == 'CoverageDynamic'
    assert info['unknown_shape_formats'] == [['ND', 'NCHW']]
    assert info['dynamicRankSupport'] is True
    assert info['dynamicShapeSupport'] is True
    assert info['dynamicCompileStatic'] is True
    assert info['needCheckSupport'] is True
    assert info['dynamicFormat'] is True
    assert info['real_input_index'] == [1, 0]
    assert info['input_to_attr_index'] == [2]
    assert info['reshape_type'] == 'CH'
    assert info['op_pattern'] == 'broadcast'
    assert info['dtype_format'] == [(('float32', 'DefaultFormat', 'Tensor'),
                                     ('float32', 'DefaultFormat', 'Tensor'))]


@pytest.mark.parametrize('method,value', [
    ('unknown_shape_formats', 'ND'), ('real_input_index', (0,)), ('input_to_attr_index', (1,)),
    ('dynamic_rank_support', 1), ('reshape_type', 1), ('dynamic_shape', 'yes'),
    ('dynamic_compile_static', 1), ('need_check_supported', 1), ('is_dynamic_format', 1),
    ('op_pattern', 1), ('kernel_name', 1), ('compute_cost', 'fast'),
])
def test_registration_rejects_invalid_metadata_types(method, value):
    """
    Feature: Registration rejects invalid metadata types.
    Description: Metadata setters must reject values that violate the backend JSON schema.
    Expectation: Metadata setters must reject values that violate the backend JSON schema.
    """
    with pytest.raises(TypeError):
        getattr(registration.TBERegOp('CoverageInvalid'), method)(value)


@pytest.mark.parametrize('formats,error', [
    ((('float32', 'DefaultFormat'),), ValueError),
    ((['float32', 'DefaultFormat'], ('float32', 'DefaultFormat')), ValueError),
    ((('float32',), ('float32', 'DefaultFormat')), ValueError),
    ((('float32', 'DefaultFormat', 3), ('float32', 'DefaultFormat')), TypeError),
])
def test_registration_dtype_format_validates_arity_and_element_types(formats, error):
    """
    Feature: Registration dtype format validates arity and element types.
    Description: Each input/output needs one complete dtype-format description.
    Expectation: Each input/output needs one complete dtype-format description.
    """
    info = registration.RegOp('CoverageFormats')
    info.inputs, info.outputs = [{'index': 0}], [{'index': 0}]
    with pytest.raises(error):
        info.dtype_format(*formats)


@pytest.mark.parametrize('as_json', [False, True])
def test_registration_decorator_preserves_payload_and_function_arguments(monkeypatch, as_json):
    """
    Feature: Registration decorator preserves payload and function arguments.
    Description: Only the native registry is replaced; encoding, paths and forwarding stay real.
    Expectation: Only the native registry is replaced; encoding, paths and forwarding stay real.
    """
    calls = []
    class NativeRegistryBoundary:
        def reg_op(self, payload, path):
            calls.append((json.loads(payload), path))
            return True
    monkeypatch.setattr(registration, 'Oplib', NativeRegistryBoundary)
    payload = {'op_name': 'CoverageAffine', 'inputs': [{'index': 0, 'name': 'x'}], 'outputs': []}
    @registration.op_info_register(json.dumps(payload) if as_json else payload)
    def affine(x, *, scale=1):
        return x * scale + 1
    assert affine(2, scale=3) == 7
    assert calls == [(payload, str(Path(__file__).resolve()))]


def test_registration_decorator_propagates_native_rejection(monkeypatch):
    """
    Feature: Registration decorator propagates native rejection.
    Description: A rejected registry payload must fail rather than producing a callable wrapper.
    Expectation: A rejected registry payload must fail rather than producing a callable wrapper.
    """
    class RejectingRegistryBoundary:
        def reg_op(self, payload, path):
            assert json.loads(payload)['op_name'] == 'CoverageRejected'
            assert path == str(Path(__file__).resolve())
            return False
    monkeypatch.setattr(registration, 'Oplib', RejectingRegistryBoundary)
    with pytest.raises(ValueError, match='Invalid op info'):
        registration.op_info_register({'op_name': 'CoverageRejected'})(lambda value: value)


def test_custom_registration_binds_metadata_without_changing_call_contract():
    """
    Feature: Custom registration binds metadata without changing call contract.
    Description: Custom registration retains metadata and positional/keyword function behavior.
    Expectation: Custom registration retains metadata and positional/keyword function behavior.
    """
    metadata = {'op_name': 'CoverageCustom', 'target': 'CPU'}
    @registration.custom_info_register(metadata)
    def affine(value, *, offset):
        return value + offset
    assert affine(3, offset=4) == 7
    assert affine.__name__ == 'affine'
    assert affine.reg_info == (metadata,)
