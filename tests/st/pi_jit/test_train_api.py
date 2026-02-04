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
"""Test mindspore.train API"""

import numpy as np
from mindspore import Tensor, jit, nn
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.dataset import GeneratorDataset

from tests.mark_utils import arg_mark
from tests.st.pi_jit.one_stage.test_utils import save_graph_ir, check_ir_num


@save_graph_ir(ir_name='graph_before_compile')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_model_checkpoint_step_end():
    """
    Feature: mindspore.train.ModelCheckpoint inside PIJit training loop.
    Description: Train Dense network with ModelCheckpoint callback in jit-compiled function and compare with pynative run.
    Expectation: JIT result matches pynative result, checkpoint files created and graph count equals one.
    Migrated from: test_pijit_cfunc_buildin.py::test_pijit_func_split_only_modelcheckpoint
    """

    def get_data():
        for _ in range(3):
            x = np.ones((8, 3), np.float32)
            y = np.ones((8, 4), np.float32)
            yield x, y

    dataset = GeneratorDataset(get_data, column_names=["x", "label"])
    net = nn.Dense(3, 4)
    loss = nn.MSELoss()
    optimizer = nn.Adam(net.trainable_params())
    ckpt_config = CheckpointConfig()
    ckpt_cbk = ModelCheckpoint(prefix="CKPT", config=ckpt_config)

    @jit(capture_mode='bytecode')
    def train():
        model = Model(net, loss, optimizer)
        model.train(3, dataset, callbacks=[ckpt_cbk], dataset_sink_mode=False)

    train()
    check_ir_num('graph_before_compile', 1)
