# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
"""test custom pass registration"""

import os
import tempfile
import pytest
from unittest.mock import patch

def test_custom_pass_registration_basic():
    """
    Feature: Custom pass registration
    Description: Test basic custom pass registration functionality
    Expectation: Pass registration succeeds with correct parameters
    """
    try:
        with patch('mindspore._c_expression.register_custom_pass') as mock_register:
            mock_register.return_value = True

            from mindspore import graph

            result = graph.register_custom_pass(
                pass_name="TestPass",
                plugin_so_path="/fake/plugin.so",
                device="cpu",
                pass_type=graph.CustomPassType.PATTERN
            )

            assert result is True
            mock_register.assert_called_once_with("TestPass", "/fake/plugin.so", "cpu", graph.CustomPassType.PATTERN)
    except ImportError:
        pytest.skip("MindSpore not available in UT environment")

def test_custom_pass_registration_parameter_validation():
    """
    Feature: Custom pass registration parameter validation
    Description: Test parameter validation for different device types
    Expectation: All valid device types should be accepted
    """
    try:
        with patch('mindspore._c_expression.register_custom_pass') as mock_register:
            mock_register.return_value = True

            from mindspore import graph

            devices = ["cpu", "gpu", "ascend", "all"]
            for device in devices:
                result = graph.register_custom_pass(
                    pass_name="TestPass",
                    plugin_so_path="/fake/plugin.so",
                    device=device,
                    pass_type=graph.CustomPassType.PATTERN
                )
                assert result is True

            assert mock_register.call_count == len(devices)
    except ImportError:
        pytest.skip("MindSpore not available in UT environment")

def test_custom_pass_registration_multiple_calls():
    """
    Feature: Custom pass registration multiple calls
    Description: Test A-B-A pattern where same pass can be registered multiple times
    Expectation: All registrations succeed and maintain correct order
    """
    try:
        with patch('mindspore._c_expression.register_custom_pass') as mock_register:
            mock_register.return_value = True

            from mindspore import graph

            result1 = graph.register_custom_pass("PassA", "/fake/plugin.so", "cpu", graph.CustomPassType.PATTERN)
            assert result1 is True

            result2 = graph.register_custom_pass("PassB", "/fake/plugin.so", "cpu", graph.CustomPassType.PATTERN)
            assert result2 is True

            result3 = graph.register_custom_pass("PassA", "/fake/plugin.so", "cpu", graph.CustomPassType.FULL_GRAPH)
            assert result3 is True

            assert mock_register.call_count == 3

            calls = mock_register.call_args_list
            assert calls[0][0] == ("PassA", "/fake/plugin.so", "cpu", graph.CustomPassType.PATTERN)
            assert calls[1][0] == ("PassB", "/fake/plugin.so", "cpu", graph.CustomPassType.PATTERN)
            assert calls[2][0] == ("PassA", "/fake/plugin.so", "cpu", graph.CustomPassType.FULL_GRAPH)
    except ImportError:
        pytest.skip("MindSpore not available in UT environment")

def test_custom_pass_registration_error_handling():
    """
    Feature: Custom pass registration error handling
    Description: Test error handling when registration fails
    Expectation: Registration failure should be properly handled
    """
    try:
        with patch('mindspore._c_expression.register_custom_pass') as mock_register:
            mock_register.return_value = False

            from mindspore import graph

            result = graph.register_custom_pass("TestPass", "/nonexistent/plugin.so")
            assert result is False

            mock_register.assert_called_once()
    except ImportError:
        pytest.skip("MindSpore not available in UT environment")

def test_custom_pass_registration_invalid_pass_type_parameter():
    """
    Feature: Custom pass registration invalid pass_type parameter
    Description: Test that TypeError is raised when pass_type is not CustomPassType enum
    Expectation: TypeError should be raised for invalid pass_type values
    """
    try:
        from mindspore import graph
        
        invalid_values = ["fullgraph", 0, None]
        
        for invalid_value in invalid_values:
            with pytest.raises(TypeError) as exc_info:
                graph.register_custom_pass(
                    pass_name="TestPass",
                    plugin_so_path="/fake/plugin.so",
                    pass_type=invalid_value
                )
            assert "pass_type must be of type CustomPassType" in str(exc_info.value)
            assert type(invalid_value).__name__ in str(exc_info.value)

    except ImportError:
        pytest.skip("MindSpore not available in UT environment")

def test_custom_pass_registration_pass_type_parameter():
    """
    Feature: Custom pass registration pass_type parameter
    Description: Test pass_type parameter passing through interface
    Expectation: pass_type parameter should be correctly passed to C++ layer
    """
    try:
        with patch('mindspore._c_expression.register_custom_pass') as mock_register:
            mock_register.return_value = True

            from mindspore import graph

            result = graph.register_custom_pass(
                pass_name="TestPass",
                plugin_so_path="/fake/plugin.so",
                device="ascend",
                pass_type=graph.CustomPassType.PATTERN
            )

            assert result is True
            mock_register.assert_called_once_with(
                "TestPass", "/fake/plugin.so", "ascend", graph.CustomPassType.PATTERN
            )
    except ImportError:
        pytest.skip("MindSpore not available in UT environment")

def test_custom_pass_registration_default_parameters():
    """
    Feature: Custom pass registration default parameters
    Description: Test default parameter values
    Expectation: Default values should be correctly applied
    """
    try:
        with patch('mindspore._c_expression.register_custom_pass') as mock_register:
            mock_register.return_value = True

            from mindspore import graph

            result = graph.register_custom_pass("TestPass", "/fake/plugin.so")

            assert result is True
            mock_register.assert_called_once_with("TestPass", "/fake/plugin.so", "all", graph.CustomPassType.PATTERN)
    except ImportError:
        pytest.skip("MindSpore not available in UT environment")

def test_custom_pass_registration_with_temp_file():
    """
    Feature: Custom pass registration with temporary file
    Description: Test registration with actual file path
    Expectation: Registration should work with real file paths
    """
    try:
        with patch('mindspore._c_expression.register_custom_pass') as mock_register:
            mock_register.return_value = True

            from mindspore import graph

            with tempfile.NamedTemporaryFile(suffix='.so', delete=False) as temp_file:
                temp_path = temp_file.name

            try:
                result = graph.register_custom_pass("TestPass", temp_path)
                assert result is True
                mock_register.assert_called_once_with("TestPass", temp_path, "all", graph.CustomPassType.PATTERN)
            finally:
                os.unlink(temp_path)
    except ImportError:
        pytest.skip("MindSpore not available in UT environment")

def test_custom_pass_api_exists():
    """
    Feature: Custom pass API existence
    Description: Test that custom pass API exists in mindspore.graph module
    Expectation: API should exist and have correct signature
    """
    try:
        from mindspore import graph
        import inspect

        assert hasattr(graph, 'register_custom_pass')
        assert callable(graph.register_custom_pass)

        sig = inspect.signature(graph.register_custom_pass)
        param_names = list(sig.parameters.keys())

        expected_params = ['pass_name', 'plugin_so_path', 'device', 'pass_type']
        for param in expected_params:
            assert param in param_names, f"Missing parameter: {param}"
    except ImportError:
        pytest.skip("MindSpore not available in UT environment")
