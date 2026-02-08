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

"""passes optimize."""

import pytest
import numpy as np
import mindspore.context as context
from mindspore import Tensor
from mindspore.graph.api import _cell_graph_executor
from .resnet_example import resnet50
from ..train_step_wrap import train_step_with_loss_warp
from .ms_opt import set_optimize_config, clear_optimize_config
from mindspore.graph._parse import compile_config
from mindspore._c_expression import GraphExecutor_ as graph_exec

context.set_context(mode=context.GRAPH_MODE)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_ascend_training
def test_set_optimize_config():
    """
    Feature: custom define the optimize config.
    Description: use the custom optimize config to compile.
    Expectation: Null.
    """
    set_optimize_config()
    net = train_step_with_loss_warp(resnet50())
    net.set_train()
    inp = Tensor(np.ones([1, 3, 224, 224], np.float32))
    label = Tensor(np.zeros([1, 10], np.float32))
    _cell_graph_executor.compile(net, inp, label)
    clear_optimize_config()


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_ascend_training
def test_config_passes():
    """
    Feature: optimize the passes.
    Description: only apply the config passes to optimize the graph.
    Expectation: Null.
    """
    config_passes = ['opt_a.r1.a_1', \
                     'opt_a.r1.a_1.arithmetic_simplify', 'opt_a.r1.a_1.cast_eliminate', 'opt_a.r1.a_1.inline',
                     'opt_a.r1.a_1.partial_eliminate', \
                     'opt_a.r1.a_1.tuple_list_get_item_depend_reorder', 'opt_a.r1.a_1.tuple_list_get_item_eliminator', \
                     'opt_a.r1.a_2', 'opt_a.r1.a_2.depend_value_elim', \
                     'opt_a.r1.a_3', 'opt_a.r1.a_3.replace_applicator', \
                     'opt_a.r1.meta_fg_expand', 'opt_a.r1.real_op_eliminate',
                     'opt_a.r1.real_op_eliminate.real_op_eliminate', 'opt_a.r1.renormalize', \
                     'opt_a.r1.switch_simplify', 'opt_a.r1.switch_simplify.switch_simplify', \
                     'opt_a.r1.updatestate_depend_eliminate', 'opt_a.r1.updatestate_loads_eliminate', \
                     'opt_a.r2.a_1', \
                     'opt_a.r2.a_1.environ_get_eliminate', 'opt_a.r2.a_1.environ_get_set_eliminate',
                     'opt_a.r2.a_1.inline', \
                     'opt_a.r2.a_1.tuple_list_get_item_eliminator', 'opt_a.r2.a_1.tuple_list_get_set_item_eliminator', \
                     'opt_a.r2.a_1.updatestate_useless_node_eliminater', \
                     'opt_a.r2.renormalize', 'opt_a.r2.updatestate_depend_eliminate',
                     'opt_a.r2.updatestate_loads_eliminate', \
                     'opt_b.r1.b_1', 'opt_b.r1.b_1.zero_like_fill_zero', 'opt_b.r1.renormalize', \
                     'opt_resolve.r1.resolve', 'opt_resolve.r1.resolve.getattr_setattr_resolve']

    graph_exec.get_instance().set_config_passes(config_passes)
    net = train_step_with_loss_warp(resnet50())
    net.set_train()
    inp = Tensor(np.ones([1, 3, 224, 224], np.float32))
    label = Tensor(np.zeros([1, 10], np.float32))
    _cell_graph_executor.compile(net, inp, label)
    # reset to original
    graph_exec.get_instance().set_config_passes([])


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_ascend_training
def test_auto_pass_optimize():
    """
    Feature: optimize the passes.
    Description: only apply the config passes to optimize the graph.
    Expectation: Null.
    """
    compile_config.AUTO_PASSES_OPTIMIZE_PATH = "."
    net = train_step_with_loss_warp(resnet50())
    net.set_train()
    inp = Tensor(np.ones([1, 3, 224, 224], np.float32))
    label = Tensor(np.zeros([1, 10], np.float32))
    _cell_graph_executor.compile(net, inp, label)
    compile_config.AUTO_PASSES_OPTIMIZE_PATH = ""
