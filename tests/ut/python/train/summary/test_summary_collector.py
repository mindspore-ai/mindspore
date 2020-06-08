# Copyright 2020 Huawei Technologies Co., Ltd
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
"""Test the exception parameter scenario for summary collector."""
import os
import tempfile
import shutil
import pytest

from mindspore.train.callback import SummaryCollector


class TestSummaryCollector:
    """Test the exception parameter for summary collector."""
    base_summary_dir = ''

    def setup_class(self):
        """Run before test this class."""
        self.base_summary_dir = tempfile.mkdtemp(suffix='summary')

    def teardown_class(self):
        """Run after test this class."""
        if os.path.exists(self.base_summary_dir):
            shutil.rmtree(self.base_summary_dir)

    @pytest.mark.parametrize("summary_dir", [1234, None, True, ''])
    def test_params_with_summary_dir_value_error(self, summary_dir):
        """Test the exception scenario for summary dir."""
        if isinstance(summary_dir, str):
            with pytest.raises(ValueError) as exc:
                SummaryCollector(summary_dir=summary_dir)
            assert str(exc.value) == 'For `summary_dir` the value should be a valid string of path, ' \
                                     'but got empty string.'
        else:
            with pytest.raises(TypeError) as exc:
                SummaryCollector(summary_dir=summary_dir)
            assert 'For `summary_dir` the type should be a valid type' in str(exc.value)

    def test_params_with_summary_dir_not_dir(self):
        """Test the given summary dir parameter is not a directory."""
        summary_dir = tempfile.mkdtemp(dir=self.base_summary_dir)
        summary_file = os.path.join(summary_dir, 'temp_file.txt')
        with open(summary_file, 'w') as file_handle:
            file_handle.write('temp')
        print(os.path.isfile(summary_file))
        with pytest.raises(NotADirectoryError):
            SummaryCollector(summary_dir=summary_file)

    @pytest.mark.parametrize("collect_freq", [None, 0, 0.01])
    def test_params_with_collect_freq_exception(self, collect_freq):
        """Test the exception scenario for collect freq."""
        summary_dir = tempfile.mkdtemp(dir=self.base_summary_dir)
        if isinstance(collect_freq, int):
            with pytest.raises(ValueError) as exc:
                SummaryCollector(summary_dir=summary_dir, collect_freq=collect_freq)
            expected_msg = f'For `collect_freq` the value should be greater than 0, but got `{collect_freq}`.'
            assert expected_msg == str(exc.value)
        else:
            with pytest.raises(TypeError) as exc:
                SummaryCollector(summary_dir=summary_dir, collect_freq=collect_freq)
            expected_msg = f"For `collect_freq` the type should be a valid type of ['int'], " \
                           f'bug got {type(collect_freq).__name__}.'
            assert expected_msg == str(exc.value)

    @pytest.mark.parametrize("action", [None, 123, '', '123'])
    def test_params_with_action_exception(self, action):
        """Test the exception scenario for action."""
        summary_dir = tempfile.mkdtemp(dir=self.base_summary_dir)
        with pytest.raises(TypeError) as exc:
            SummaryCollector(summary_dir=summary_dir, keep_default_action=action)
        expected_msg = f"For `keep_default_action` the type should be a valid type of ['bool'], " \
                       f"bug got {type(action).__name__}."
        assert expected_msg == str(exc.value)

    @pytest.mark.parametrize("collect_specified_data", [123])
    def test_params_with_collect_specified_data_type_error(self, collect_specified_data):
        """Test type error scenario for collect specified data param."""
        summary_dir = tempfile.mkdtemp(dir=self.base_summary_dir)
        with pytest.raises(TypeError) as exc:
            SummaryCollector(summary_dir, collect_specified_data=collect_specified_data)

        expected_msg = f"For `collect_specified_data` the type should be a valid type of ['dict', 'NoneType'], " \
                       f"bug got {type(collect_specified_data).__name__}."

        assert expected_msg == str(exc.value)

    @pytest.mark.parametrize("collect_specified_data", [
        {
            123: 123
        },
        {
            None: True
        }
    ])
    def test_params_with_collect_specified_data_key_type_error(self, collect_specified_data):
        """Test the key of collect specified data param."""
        summary_dir = tempfile.mkdtemp(dir=self.base_summary_dir)
        with pytest.raises(TypeError) as exc:
            SummaryCollector(summary_dir, collect_specified_data=collect_specified_data)

        param_name = list(collect_specified_data)[0]
        expected_msg = f"For `{param_name}` the type should be a valid type of ['str'], " \
                       f"bug got {type(param_name).__name__}."
        assert expected_msg == str(exc.value)

    @pytest.mark.parametrize("collect_specified_data", [
        {
            'collect_metric': None
        },
        {
            'collect_graph': 123
        },
        {
            'histogram_regular': 123
        },
    ])
    def test_params_with_collect_specified_data_value_type_error(self, collect_specified_data):
        """Test the value of collect specified data param."""
        summary_dir = tempfile.mkdtemp(dir=self.base_summary_dir)
        with pytest.raises(TypeError) as exc:
            SummaryCollector(summary_dir, collect_specified_data=collect_specified_data)

        param_name = list(collect_specified_data)[0]
        param_value = collect_specified_data[param_name]
        expected_type = "['bool']" if param_name != 'histogram_regular' else "['str', 'NoneType']"
        expected_msg = f'For `{param_name}` the type should be a valid type of {expected_type}, ' \
                       f'bug got {type(param_value).__name__}.'

        assert expected_msg == str(exc.value)

    def test_params_with_collect_specified_data_unexpected_key(self):
        """Test the collect_specified_data parameter with unexpected key."""
        summary_dir = tempfile.mkdtemp(dir=self.base_summary_dir)
        data = {'unexpected_key': True}
        with pytest.raises(ValueError) as exc:
            SummaryCollector(summary_dir, collect_specified_data=data)
        expected_msg = f"For `collect_specified_data` the keys {set(data)} are unsupported."
        assert expected_msg == str(exc.value)

    @pytest.mark.parametrize("custom_lineage_data", [
        123,
        {
            'custom': {}
        },
        {
            'custom': None
        },
        {
            123: 'custom'
        }
    ])
    def test_params_with_custom_lineage_data_type_error(self, custom_lineage_data):
        """Test the custom lineage data parameter type error."""
        summary_dir = tempfile.mkdtemp(dir=self.base_summary_dir)
        with pytest.raises(TypeError) as exc:
            SummaryCollector(summary_dir, custom_lineage_data=custom_lineage_data)

        if not isinstance(custom_lineage_data, dict):
            expected_msg = f"For `custom_lineage_data` the type should be a valid type of ['dict', 'NoneType'], " \
                           f"bug got {type(custom_lineage_data).__name__}."
        else:
            param_name = list(custom_lineage_data)[0]
            param_value = custom_lineage_data[param_name]
            if not isinstance(param_name, str):
                arg_name = f'custom_lineage_data -> {param_name}'
                expected_msg = f"For `{arg_name}` the type should be a valid type of ['str'], " \
                               f'bug got {type(param_name).__name__}.'
            else:
                arg_name = f'the value of custom_lineage_data -> {param_name}'
                expected_msg = f"For `{arg_name}` the type should be a valid type of ['int', 'str', 'float'], " \
                               f'bug got {type(param_value).__name__}.'

        assert expected_msg == str(exc.value)
