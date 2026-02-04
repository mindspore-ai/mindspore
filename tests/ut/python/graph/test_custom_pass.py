# Copyright 2025 Huawei Technologies Co., Ltd
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

            import mindspore.graph as graph

            result = graph.register_custom_pass(
                pass_name="TestPass",
                plugin_so_path="/fake/plugin.so",
                device="cpu",
                stage="stage1"
            )

            assert result is True
            mock_register.assert_called_once_with("TestPass", "/fake/plugin.so", "cpu", "stage1")
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

            import mindspore.graph as graph

            devices = ["cpu", "gpu", "ascend", "all"]
            for device in devices:
                result = graph.register_custom_pass(
                    pass_name="TestPass",
                    plugin_so_path="/fake/plugin.so",
                    device=device,
                    stage="stage1"
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

            import mindspore.graph as graph

            result1 = graph.register_custom_pass("PassA", "/fake/plugin.so", "cpu", "stage1")
            assert result1 is True

            result2 = graph.register_custom_pass("PassB", "/fake/plugin.so", "cpu", "stage1")
            assert result2 is True

            result3 = graph.register_custom_pass("PassA", "/fake/plugin.so", "cpu", "stage2")
            assert result3 is True

            assert mock_register.call_count == 3

            calls = mock_register.call_args_list
            assert calls[0][0] == ("PassA", "/fake/plugin.so", "cpu", "stage1")
            assert calls[1][0] == ("PassB", "/fake/plugin.so", "cpu", "stage1")
            assert calls[2][0] == ("PassA", "/fake/plugin.so", "cpu", "stage2")
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

            import mindspore.graph as graph

            result = graph.register_custom_pass("TestPass", "/nonexistent/plugin.so")
            assert result is False

            mock_register.assert_called_once()
    except ImportError:
        pytest.skip("MindSpore not available in UT environment")

def test_custom_pass_registration_stage_parameter():
    """
    Feature: Custom pass registration stage parameter
    Description: Test stage parameter passing through interface
    Expectation: Stage parameter should be correctly passed to C++ layer
    """
    try:
        with patch('mindspore._c_expression.register_custom_pass') as mock_register:
            mock_register.return_value = True

            import mindspore.graph as graph

            result = graph.register_custom_pass(
                pass_name="TestPass",
                plugin_so_path="/fake/plugin.so",
                device="ascend",
                stage="optimization_stage"
            )

            assert result is True
            mock_register.assert_called_once_with(
                "TestPass", "/fake/plugin.so", "ascend", "optimization_stage"
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

            import mindspore.graph as graph

            result = graph.register_custom_pass("TestPass", "/fake/plugin.so")

            assert result is True
            mock_register.assert_called_once_with("TestPass", "/fake/plugin.so", "all", "")
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

            import mindspore.graph as graph

            with tempfile.NamedTemporaryFile(suffix='.so', delete=False) as temp_file:
                temp_path = temp_file.name

            try:
                result = graph.register_custom_pass("TestPass", temp_path)
                assert result is True
                mock_register.assert_called_once_with("TestPass", temp_path, "all", "")
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
        import mindspore.graph as graph
        import inspect

        assert hasattr(graph, 'register_custom_pass')
        assert callable(graph.register_custom_pass)

        sig = inspect.signature(graph.register_custom_pass)
        param_names = list(sig.parameters.keys())

        expected_params = ['pass_name', 'plugin_so_path', 'device', 'stage']
        for param in expected_params:
            assert param in param_names, f"Missing parameter: {param}"
    except ImportError:
        pytest.skip("MindSpore not available in UT environment")
