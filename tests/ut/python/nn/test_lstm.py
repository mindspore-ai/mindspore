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
""" test lstm """
import pytest

import mindspore.context as context
from mindspore import nn
from ..ut_filter import run_on_gpu
from ....ops_common import convert


class LstmTestNet(nn.Cell):
    """ LstmTestNet definition """

    def __init__(self, input_size, hidden_size, num_layers, has_bias, batch_first, bidirectional):
        super(LstmTestNet, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            has_bias=has_bias,
                            batch_first=batch_first,
                            bidirectional=bidirectional,
                            dropout=0.0)

    def construct(self, inp, h0, c0):
        return self.lstm(inp, (h0, c0))


test_case_cell_ops = [
    ('lstm1_with_bias', {
        'cell': LstmTestNet(10, 12, 2, has_bias=True, batch_first=False, bidirectional=False),
        'input_shape': [[5, 3, 10], [2, 3, 12], [2, 3, 12]],
        'output_shape': [[5, 3, 12], [2, 3, 12], [2, 3, 12]]}),
    ('lstm2_without_bias', {
        'cell': LstmTestNet(10, 12, 2, has_bias=False, batch_first=False, bidirectional=False),
        'input_shape': [[5, 3, 10], [2, 3, 12], [2, 3, 12]],
        'output_shape': [[5, 3, 12], [2, 3, 12], [2, 3, 12]]}),
    ('lstm3_with_bias_bidirectional', {
        'cell': LstmTestNet(10, 12, 2, has_bias=True, batch_first=False, bidirectional=True),
        'input_shape': [[5, 3, 10], [4, 3, 12], [4, 3, 12]],
        'output_shape': [[5, 3, 24], [4, 3, 12], [4, 3, 12]]}),
    ('lstm4_without_bias_bidirectional', {
        'cell': LstmTestNet(10, 12, 2, has_bias=False, batch_first=False, bidirectional=True),
        'input_shape': [[5, 3, 10], [4, 3, 12], [4, 3, 12]],
        'output_shape': [[5, 3, 24], [4, 3, 12], [4, 3, 12]]}),
    ('lstm5_with_bias_batch_first', {
        'cell': LstmTestNet(10, 12, 2, has_bias=True, batch_first=True, bidirectional=False),
        'input_shape': [[3, 5, 10], [2, 3, 12], [2, 3, 12]],
        'output_shape': [[3, 5, 12], [2, 3, 12], [2, 3, 12]]}),
    ('lstm6_without_bias_batch_first', {
        'cell': LstmTestNet(10, 12, 2, has_bias=False, batch_first=True, bidirectional=False),
        'input_shape': [[3, 5, 10], [2, 3, 12], [2, 3, 12]],
        'output_shape': [[3, 5, 12], [2, 3, 12], [2, 3, 12]]}),
    ('lstm7_with_bias_bidirectional_batch_first', {
        'cell': LstmTestNet(10, 12, 2, has_bias=True, batch_first=True, bidirectional=True),
        'input_shape': [[3, 5, 10], [4, 3, 12], [4, 3, 12]],
        'output_shape': [[3, 5, 24], [4, 3, 12], [4, 3, 12]]}),
    ('lstm8_without_bias_bidirectional_batch_first', {
        'cell': LstmTestNet(10, 12, 2, has_bias=False, batch_first=True, bidirectional=True),
        'input_shape': [[3, 5, 10], [4, 3, 12], [4, 3, 12]],
        'output_shape': [[3, 5, 24], [4, 3, 12], [4, 3, 12]]}),
]


# use -k to select certain testcast
# pytest  tests/python/ops/test_lstm.py::test_compile -k lstm_with_bias

@pytest.mark.parametrize('args', test_case_cell_ops, ids=lambda x: x[0])
def test_compile(args):
    config = args[1]
    shapes = config['input_shape']
    net = config['cell']
    net.set_train()
    inputs = [convert(shp) for shp in shapes]
    out = net(*inputs)
    print(f"out: {out}")


@run_on_gpu
@pytest.mark.parametrize('args', test_case_cell_ops, ids=lambda x: x[0])
def test_execute(args):
    """ test_execute """
    config = args[1]
    shapes = config['input_shape']
    net = config['cell']
    net.set_train()
    inputs = [convert(shp) for shp in shapes]
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    # pylint: disable=unused-variable
    ret, (hn, cn) = net(*inputs)
    print(f'result: {shapes[0]} --> {ret.asnumpy().shape}, expected: {config["output_shape"][0]}')
