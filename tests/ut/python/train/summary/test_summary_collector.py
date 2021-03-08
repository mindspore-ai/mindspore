# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
from importlib import import_module
from unittest import mock

import numpy as np
import pytest

from mindspore import Tensor
from mindspore import Parameter
from mindspore.train.callback import SummaryCollector
from mindspore.train.callback import _InternalCallbackParam
from mindspore.train.summary.enums import ModeEnum, PluginEnum
from mindspore.train.summary import SummaryRecord
from mindspore.train.summary.summary_record import _DEFAULT_EXPORT_OPTIONS
from mindspore.nn import Cell
from mindspore.nn.optim.optimizer import Optimizer
from mindspore.ops.operations import Add



_VALUE_CACHE = list()


def add_value(plugin, name, value):
    """This function is mock the function in SummaryRecord."""
    global _VALUE_CACHE
    _VALUE_CACHE.append((plugin, name, value))


def get_value():
    """Get the value which is added by add_value function."""
    global _VALUE_CACHE

    value = _VALUE_CACHE
    _VALUE_CACHE = list()
    return value

_SPECIFIED_DATA = SummaryCollector._DEFAULT_SPECIFIED_DATA
_SPECIFIED_DATA['collect_metric'] = False


class CustomNet(Cell):
    """Define custom network."""
    def __init__(self):
        super(CustomNet, self).__init__()
        self.add = Add
        self.optimizer = Optimizer(learning_rate=1, parameters=[Parameter(Tensor(1), 'weight')])

    def construct(self, data):
        return data


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

    def teardown_method(self):
        """Run after each test function."""
        get_value()

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
                           f'but got {type(collect_freq).__name__}.'
            assert expected_msg == str(exc.value)

    @pytest.mark.parametrize("action", [None, 123, '', '123'])
    def test_params_with_action_exception(self, action):
        """Test the exception scenario for action."""
        summary_dir = tempfile.mkdtemp(dir=self.base_summary_dir)
        with pytest.raises(TypeError) as exc:
            SummaryCollector(summary_dir=summary_dir, keep_default_action=action)
        expected_msg = f"For `keep_default_action` the type should be a valid type of ['bool'], " \
                       f"but got {type(action).__name__}."
        assert expected_msg == str(exc.value)

    @pytest.mark.parametrize("collect_specified_data", [123])
    def test_params_with_collect_specified_data_type_error(self, collect_specified_data):
        """Test type error scenario for collect specified data param."""
        summary_dir = tempfile.mkdtemp(dir=self.base_summary_dir)
        with pytest.raises(TypeError) as exc:
            SummaryCollector(summary_dir, collect_specified_data=collect_specified_data)

        expected_msg = f"For `collect_specified_data` the type should be a valid type of ['dict', 'NoneType'], " \
                       f"but got {type(collect_specified_data).__name__}."

        assert expected_msg == str(exc.value)

    @pytest.mark.parametrize("export_options", [
        {
            "tensor_format": "npz"
        }
    ])
    def test_params_with_tensor_format_type_error(self, export_options):
        """Test type error scenario for collect specified data param."""
        summary_dir = tempfile.mkdtemp(dir=self.base_summary_dir)
        with pytest.raises(ValueError) as exc:
            SummaryCollector(summary_dir, export_options=export_options)

        unexpected_format = {export_options.get("tensor_format")}
        expected_msg = f'For `export_options`, the export_format {unexpected_format} are ' \
                       f'unsupported for tensor_format, expect the follow values: ' \
                       f'{list(_DEFAULT_EXPORT_OPTIONS.get("tensor_format"))}'

        assert expected_msg == str(exc.value)

    @pytest.mark.parametrize("export_options", [123])
    def test_params_with_export_options_type_error(self, export_options):
        """Test type error scenario for collect specified data param."""
        summary_dir = tempfile.mkdtemp(dir=self.base_summary_dir)
        with pytest.raises(TypeError) as exc:
            SummaryCollector(summary_dir, export_options=export_options)

        expected_msg = f"For `export_options` the type should be a valid type of ['dict', 'NoneType'], " \
                       f"but got {type(export_options).__name__}."

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
                       f"but got {type(param_name).__name__}."
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
                       f'but got {type(param_value).__name__}.'

        assert expected_msg == str(exc.value)

    def test_params_with_histogram_regular_value_error(self):
        """Test histogram regular."""
        summary_dir = tempfile.mkdtemp(dir=self.base_summary_dir)
        with pytest.raises(ValueError) as exc:
            SummaryCollector(summary_dir, collect_specified_data={'histogram_regular': '*'})

        assert 'For `collect_specified_data`, the value of `histogram_regular`' in str(exc.value)

    def test_params_with_collect_specified_data_unexpected_key(self):
        """Test the collect_specified_data parameter with unexpected key."""
        summary_dir = tempfile.mkdtemp(dir=self.base_summary_dir)
        data = {'unexpected_key': True}
        with pytest.raises(ValueError) as exc:
            SummaryCollector(summary_dir, collect_specified_data=data)
        expected_msg = f"For `collect_specified_data` the keys {set(data)} are unsupported"
        assert expected_msg in str(exc.value)

    def test_params_with_export_options_unexpected_key(self):
        """Test the export_options parameter with unexpected key."""
        summary_dir = tempfile.mkdtemp(dir=self.base_summary_dir)
        data = {'unexpected_key': "value"}
        with pytest.raises(ValueError) as exc:
            SummaryCollector(summary_dir, export_options=data)
        expected_msg = f"For `export_options` the keys {set(data)} are unsupported"
        assert expected_msg in str(exc.value)

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
                           f"but got {type(custom_lineage_data).__name__}."
        else:
            param_name = list(custom_lineage_data)[0]
            param_value = custom_lineage_data[param_name]
            if not isinstance(param_name, str):
                arg_name = f'custom_lineage_data -> {param_name}'
                expected_msg = f"For `{arg_name}` the type should be a valid type of ['str'], " \
                               f'but got {type(param_name).__name__}.'
            else:
                arg_name = f'the value of custom_lineage_data -> {param_name}'
                expected_msg = f"For `{arg_name}` the type should be a valid type of ['int', 'str', 'float'], " \
                               f'but got {type(param_value).__name__}.'

        assert expected_msg == str(exc.value)

    def test_check_callback_with_multi_instances(self):
        """Use multi SummaryCollector instances to test check_callback function."""
        cb_params = _InternalCallbackParam()
        cb_params.list_callback = [
            SummaryCollector(tempfile.mkdtemp(dir=self.base_summary_dir)),
            SummaryCollector(tempfile.mkdtemp(dir=self.base_summary_dir))
        ]
        with pytest.raises(ValueError) as exc:
            SummaryCollector((tempfile.mkdtemp(dir=self.base_summary_dir)))._check_callbacks(cb_params)
        assert f"more than one SummaryCollector instance in callback list" in str(exc.value)

    def test_collect_input_data_with_train_dataset_element_invalid(self):
        """Test the param 'train_dataset_element' in cb_params is invalid."""
        cb_params = _InternalCallbackParam()
        for invalid in (), [], None:
            cb_params.train_dataset_element = invalid
            summary_collector = SummaryCollector(tempfile.mkdtemp(dir=self.base_summary_dir))
            summary_collector._collect_input_data(cb_params)
            assert not summary_collector._collect_specified_data['collect_input_data']

    @mock.patch.object(SummaryRecord, 'add_value')
    def test_collect_input_data_success(self, mock_add_value):
        """Mock a image data, and collect image data success."""
        mock_add_value.side_effect = add_value
        cb_params = _InternalCallbackParam()
        image_data = Tensor(np.random.randint(0, 255, size=(1, 1, 1, 1)).astype(np.uint8))
        cb_params.train_dataset_element = image_data
        with SummaryCollector((tempfile.mkdtemp(dir=self.base_summary_dir))) as summary_collector:
            summary_collector._collect_input_data(cb_params)
            # Note Here need to assert the result and expected data

    @mock.patch.object(SummaryRecord, 'add_value')
    def test_collect_dataset_graph_success(self, mock_add_value):
        """Test collect dataset graph."""
        dataset = import_module('mindspore.dataset')
        mock_add_value.side_effect = add_value
        cb_params = _InternalCallbackParam()
        cb_params.train_dataset = dataset.MnistDataset(dataset_dir=tempfile.mkdtemp(dir=self.base_summary_dir))
        cb_params.mode = ModeEnum.TRAIN.value
        with SummaryCollector((tempfile.mkdtemp(dir=self.base_summary_dir))) as summary_collector:
            summary_collector._collect_dataset_graph(cb_params)
            plugin, name, _ = get_value()[0]
        assert plugin == 'dataset_graph'
        assert name == 'train_dataset'

    @pytest.mark.parametrize("net_output, expected_loss", [
        (None, None),
        (1, Tensor(1)),
        (1.5, Tensor(1.5)),
        (Tensor(1), Tensor(1)),
        ([1], Tensor(1)),
        ([Tensor(1)], Tensor(1)),
        ({}, None),
        (Tensor([[1, 2], [3, 4]]), Tensor(2.5)),
        ([Tensor([[3, 4, 3]]), Tensor([3, 4])], Tensor(3.33333)),
        (tuple([1]), Tensor(1)),
    ])
    def test_get_loss(self, net_output, expected_loss):
        """Test get loss success and failed."""
        cb_params = _InternalCallbackParam()
        cb_params.net_outputs = net_output
        summary_collector = SummaryCollector((tempfile.mkdtemp(dir=self.base_summary_dir)))
        summary_collector._get_loss(cb_params)

        if expected_loss is None:
            assert not summary_collector._is_parse_loss_success
        else:
            assert summary_collector._is_parse_loss_success

    def test_get_optimizer_from_cb_params_success(self):
        """Test get optimizer success from cb params."""
        cb_params = _InternalCallbackParam()
        cb_params.optimizer = Optimizer(learning_rate=0.1, parameters=[Parameter(Tensor(1), 'weight')])
        summary_collector = SummaryCollector((tempfile.mkdtemp(dir=self.base_summary_dir)))
        optimizer = summary_collector._get_optimizer(cb_params)
        assert optimizer == cb_params.optimizer

        # Test get optimizer again
        assert summary_collector._get_optimizer(cb_params) == cb_params.optimizer

    @pytest.mark.parametrize('mode', [ModeEnum.TRAIN.value, ModeEnum.EVAL.value])
    def test_get_optimizer_from_network(self, mode):
        """Get optimizer from train network"""
        cb_params = _InternalCallbackParam()
        cb_params.optimizer = None
        cb_params.mode = mode
        if mode == ModeEnum.TRAIN.value:
            cb_params.train_network = CustomNet()
        else:
            cb_params.eval_network = CustomNet()
        summary_collector = SummaryCollector((tempfile.mkdtemp(dir=self.base_summary_dir)))
        optimizer = summary_collector._get_optimizer(cb_params)
        assert isinstance(optimizer, Optimizer)

    def test_get_optimizer_failed(self):
        """Test get optimizer failed."""
        class Net(Cell):
            """Define net."""
            def __init__(self):
                super(Net, self).__init__()
                self.add = Add()

            def construct(self, data):
                return data

        cb_params = _InternalCallbackParam()
        cb_params.optimizer = None
        cb_params.train_network = Net()
        cb_params.mode = ModeEnum.TRAIN.value
        summary_collector = SummaryCollector((tempfile.mkdtemp(dir=self.base_summary_dir)))
        optimizer = summary_collector._get_optimizer(cb_params)
        assert optimizer is None
        assert summary_collector._temp_optimizer == 'Failed'

        # Test get optimizer again
        optimizer = summary_collector._get_optimizer(cb_params)
        assert optimizer is None
        assert summary_collector._temp_optimizer == 'Failed'

    @pytest.mark.parametrize("histogram_regular, expected_names", [
        (
            'conv1|conv2',
            ['conv1.weight1/auto', 'conv2.weight2/auto', 'conv1.bias1/auto']
        ),
        (
            None,
            ['conv1.weight1/auto', 'conv2.weight2/auto', 'conv1.bias1/auto', 'conv3.bias/auto', 'conv5.bias/auto']
        )
    ])
    @mock.patch.object(SummaryRecord, 'add_value')
    def test_collect_histogram_from_regular(self, mock_add_value, histogram_regular, expected_names):
        """Test collect histogram from regular success."""
        mock_add_value.side_effect = add_value
        cb_params = _InternalCallbackParam()
        parameters = [
            Parameter(Tensor(1), 'conv1.weight1'),
            Parameter(Tensor(2), 'conv2.weight2'),
            Parameter(Tensor(3), 'conv1.bias1'),
            Parameter(Tensor(4), 'conv3.bias'),
            Parameter(Tensor(5), 'conv5.bias'),
            Parameter(Tensor(6), 'conv6.bias'),
        ]
        cb_params.optimizer = Optimizer(learning_rate=0.1, parameters=parameters)
        with SummaryCollector((tempfile.mkdtemp(dir=self.base_summary_dir))) as summary_collector:
            summary_collector._collect_specified_data['histogram_regular'] = histogram_regular
            summary_collector._collect_histogram(cb_params)
        result = get_value()
        assert PluginEnum.HISTOGRAM.value == result[0][0]
        assert expected_names == [data[1] for data in result]

    @pytest.mark.parametrize("specified_data, action, expected_result", [
        (None, True, SummaryCollector._DEFAULT_SPECIFIED_DATA),
        (None, False, {}),
        ({}, True, SummaryCollector._DEFAULT_SPECIFIED_DATA),
        ({}, False, {}),
        ({'collect_metric': False}, True, _SPECIFIED_DATA),
        ({'collect_metric': True}, False, {'collect_metric': True})
    ])
    def test_process_specified_data(self, specified_data, action, expected_result):
        """Test process specified data."""
        summary_dir = tempfile.mkdtemp(dir=self.base_summary_dir)
        summary_collector = SummaryCollector(summary_dir,
                                             collect_specified_data=specified_data,
                                             keep_default_action=action)

        assert summary_collector._collect_specified_data == expected_result
