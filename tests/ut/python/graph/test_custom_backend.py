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
"""test custom backend registration"""

import pytest
import tempfile
import os
from unittest.mock import patch

def test_custom_backend_registration_basic():
    """
    Feature: Custom backend registration
    Description: Test basic custom backend registration functionality
    Expectation: Pass registration succeeds with correct parameters
    """
    with pytest.raises(ValueError):
        from mindspore import graph

        graph.register_custom_backend(
            backend_name="TestBackend",
            backend_path="/fake/plugin.so",
        )


def test_custom_backend_registration_multiple_calls():
    """
    Feature: custom backend registration multiple calls
    Description: Test same backend can be registered multiple times
    Expectation: All registrations succeed and maintain correct order
    """
    try:
        with patch('mindspore._c_expression.register_custom_backend') as mock_register:
            mock_register.return_value = True

            from mindspore import graph
            with tempfile.NamedTemporaryFile(suffix='.so', delete=False) as temp_file:
                temp_path = temp_file.name
            try:
                result1 = graph.register_custom_backend("backend1", temp_path)
                assert result1 is True

                result2 = graph.register_custom_backend("backend2", temp_path)
                assert result2 is True

                result3 = graph.register_custom_backend("backend3", temp_path)
                assert result3 is True

                assert mock_register.call_count == 3

                calls = mock_register.call_args_list
                assert calls[0][0] == ("backend1", temp_path)
                assert calls[1][0] == ("backend2", temp_path)
                assert calls[2][0] == ("backend3", temp_path)
            finally:
                os.unlink(temp_path)
    except ImportError:
        pytest.skip("MindSpore not available in UT environment")

def test_custom_backend_registration_error_handling():
    """
    Feature: custom backend registration error handling
    Description: Test error handling when registration fails
    Expectation: Registration failure should be properly handled
    """
    try:
        with patch('mindspore._c_expression.register_custom_backend') as mock_register:
            mock_register.return_value = False

            from mindspore import graph
            with tempfile.NamedTemporaryFile(suffix='.so', delete=False) as temp_file:
                temp_path = temp_file.name
            result = graph.register_custom_backend("TestBackend", temp_path)
            assert result is False
            mock_register.assert_called_once()
            os.unlink(temp_path)
    except ImportError:
        pytest.skip("MindSpore not available in UT environment")

def test_custom_backend_registration_success_hanldling():
    """
    Feature: custom backend registration success handling
    Description: Test stage parameter passing through interface
    Expectation: Registration success should be properly handled
    """
    try:
        with patch('mindspore._c_expression.register_custom_backend') as mock_register:
            mock_register.return_value = True

            from mindspore import graph
            with tempfile.NamedTemporaryFile(suffix='.so', delete=False) as temp_file:
                temp_path = temp_file.name
            result = graph.register_custom_backend(
                backend_name="TestBackend",
                backend_path=temp_path,
            )

            assert result is True
            mock_register.assert_called_once_with(
                "TestBackend", temp_path
            )
            os.unlink(temp_path)
    except ImportError:
        pytest.skip("MindSpore not available in UT environment")


def test_custom_backend_api_exists():
    """
    Feature: custom backend API existence
    Description: Test that custom backend API exists in mindspore.graph module
    Expectation: API should exist and have correct signature
    """
    try:
        from mindspore import graph
        import inspect

        assert hasattr(graph, 'register_custom_backend')
        assert callable(graph.register_custom_backend)

        sig = inspect.signature(graph.register_custom_backend)
        param_names = list(sig.parameters.keys())

        expected_params = ['backend_name', 'backend_path']
        for param in expected_params:
            assert param in param_names, f"Missing parameter: {param}"
    except ImportError:
        pytest.skip("MindSpore not available in UT environment")
