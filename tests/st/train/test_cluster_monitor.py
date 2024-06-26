# Copyright 2024 Huawei Technologies Co., Ltd
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
import pytest

import mindspore as ms
from mindspore.train.callback._cluster_monitor import _get_dp_tp_from_layout
from tests.mark_utils import arg_mark

parameter_layout_dict = {
    'accu_grads.backbone.embedding.word_embedding.embedding_table':
        ([4, 4], [0, -1], [10000, 2560], 0, True, ''),
    'accu_grads.backbone.blocks.16.attention.projection.weight':
        ([4, 4], [0, -1], [640, 2560], 0, True, '4-11650191013956257822'),
    'accu_grads.backbone.blocks.16.attention.dense1.weight':
        ([4, 4], [0, -1], [640, 2560], 0, True, '4-11650191013956257822'),
    'accu_grads.backbone.blocks.16.attention.dense2.weight':
        ([4, 4], [0, -1], [640, 2560], 0, True, '4-11650191013956257822'),
    'accu_grads.backbone.blocks.16.attention.dense3.weight':
        ([4, 4], [0, -1], [640, 2560], 0, True, '4-11650191013956257822'),
    'accu_grads.backbone.blocks.16.output.mapping.weight':
        ([4, 4], [-1, 0], [2560, 2560], 0, True, '4-11650191013956257822'),
    'accu_grads.backbone.blocks.16.layernorm1.gamma':
        ([4, 4], [-1], [2560], 0, True, ''),
    'accu_grads.backbone.blocks.16.layernorm1.beta':
        ([4, 4], [-1], [2560], 0, True, ''),
    'accu_grads.backbone.blocks.16.layernorm2.gamma':
        ([4, 4], [-1], [2560], 0, True, ''),
    'accu_grads.backbone.blocks.16.attention.dense1.bias':
        ([4, 4], [0], [640], 0, True, ''),
}


@arg_mark(
    plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend', 'platform_ascend910b'],
    level_mark='level0',
    card_mark='onecard',
    essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_functional_amp_overflow(mode):
    """
    Feature: mindspore.amp.overflow
    Description: test amp overflow
    Expectation: Success.
    """
    ms.set_context(mode=mode)
    dp, tp = _get_dp_tp_from_layout(parameter_layout_dict)
    assert tp == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]
    assert dp == [[0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15]]
