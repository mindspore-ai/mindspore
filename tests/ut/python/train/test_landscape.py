# Copyright 2021 Huawei Technologies Co., Ltd
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
"""test create landscape."""
import os
import shutil
import tempfile
import pytest

from mindspore.common import set_seed
from mindspore import nn, SummaryLandscape
from mindspore.train import Loss
from mindspore.train import Model
from tests.security_utils import security_off_wrap
from tests.ut.python.train.dataset import create_mnist_dataset, LeNet5

set_seed(1)

_VALUE_CACHE = list()

def get_value():
    """Get the value which is added by add_value function."""
    global _VALUE_CACHE

    value = _VALUE_CACHE
    _VALUE_CACHE = list()
    return value


def callback_fn():
    """A python function job"""
    network = LeNet5()
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    metrics = {"Loss": Loss()}
    model = Model(network, loss, metrics=metrics)
    ds_train = create_mnist_dataset("train")
    return model, network, ds_train, metrics


class TestLandscape:
    """Test the exception parameter for landscape."""
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

    @security_off_wrap
    @pytest.mark.parametrize("collect_landscape", [
        {
            'landscape_size': None
        },
        {
            'create_landscape': None
        },
        {
            'num_samples': None
        },
        {
            'intervals': None
        },
    ])
    def test_params_gen_landscape_with_multi_process_value_type_error(self, collect_landscape):
        """Test the value of gen_landscape_with_multi_process param."""
        device_id = int(os.getenv('DEVICE_ID')) if os.getenv('DEVICE_ID') else 0
        summary_dir = tempfile.mkdtemp(dir=self.base_summary_dir)
        summary_landscape = SummaryLandscape(summary_dir)
        with pytest.raises(TypeError) as exc:
            summary_landscape.gen_landscapes_with_multi_process(
                callback_fn,
                collect_landscape=collect_landscape,
                device_ids=[device_id]
            )
        param_name = list(collect_landscape)[0]
        param_value = collect_landscape[param_name]
        if param_name in ['landscape_size', 'num_samples']:
            expected_type = "['int']"
        elif param_name == 'unit':
            expected_type = "['str']"
        elif param_name == 'create_landscape':
            expected_type = "['dict']"
        else:
            expected_type = "['list']"
        expected_msg = f'For `{param_name}` the type should be a valid type of {expected_type}, ' \
                       f'but got {type(param_value).__name__}.'
        assert expected_msg == str(exc.value)
