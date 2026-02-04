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
"""
Test cases of parallel_comm_switch_nic
"""
import os
import numpy as np
import mindspore as ms
from mindspore import Tensor
from mindspore import log as logger
from mindspore import Parameter
from mindspore.nn import Cell
from mindspore.communication import init
import mindspore.ops.operations as P
from mindspore.parallel.auto_parallel import AutoParallel
from mindspore.parallel.shard import Layout
from mindspore.communication.management import _comm_switch_nic, get_rank
from mindspore.train import Callback
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint
from mindspore.train.serialization import load_checkpoint
from ..utils.dataset import FakeData
from ..utils.context_base import contextbase
from ..utils.compare_base import comparebase
from ..utils.modeltrain_base import modeltrainbase
from ..utils._utils import clean_all_ckpt_files, find_newest_ckpt_file_by_name


def setup_function():
    from mindspore import set_device
    rank_id = get_rank() if 'RANK_SIZE' in os.environ and int(os.environ['RANK_SIZE']) > 1 else 0
    set_device("Ascend", rank_id)
    contextbase.case_prepare()


def teardown_function():
    contextbase.case_cleanup()


class SimpleNet(Cell):
    def __init__(self, mul_size, in_strategy1=None, in_strategy2=None, dtype=np.float32):
        super().__init__()
        np.random.seed(666)
        mul_t = Tensor(np.random.random(size=mul_size).astype(dtype))
        self.mul_param = Parameter(mul_t, name="mul_weight")
        self.add = P.Add()
        self.mul = P.Mul()
        if in_strategy1 is not None:
            self.add.shard(in_strategy1)
        if in_strategy2 is not None:
            self.mul.shard(in_strategy2)

    def construct(self, inputs, label):
        out = self.add(inputs, inputs)
        out = self.mul(out, self.mul_param)
        return out


class CommSwitchNic(Callback):
    def __init__(self, switch_ranks, switch_status, switch_steps):
        super().__init__()
        self.count = 0
        self.switch_ranks = switch_ranks
        self.switch_status = switch_status
        self.switch_steps = switch_steps

    def on_train_step_end(self, run_context):
        cb_params = run_context.original_args()
        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1
        loss = float(np.mean(cb_params.net_outputs.asnumpy()))
        cur_epoch_num = cb_params.get("cur_epoch_num", 1)
        logger.info(f"epoch: {cur_epoch_num} step: {cur_step_in_epoch}, loss is {loss}")
        if cur_step_in_epoch in self.switch_steps and self.count == 0:
            try:
                _comm_switch_nic(self.switch_ranks, self.switch_status)
            except Exception as e:  # pylint: disable=W0718
                logger.error(f"Failed to switch NIC: {e}")
            finally:
                self.count += 1


def load_newest_ckpt_from_model_train(model, epoch, dataset, callback=None,
                                      dataset_sink_mode=True,
                                      ckpt_path="./", ckpt_prefix="ckpt_ms", async_save=False,
                                      save_checkpoint_steps=1, sink_size=-1,
                                      integrated_save=True):
    logger.info("MindSporeTest::configure Config to save Checkpoint")
    ckpt_config = CheckpointConfig(
        keep_checkpoint_max=5,
        integrated_save=integrated_save,
        save_checkpoint_steps=save_checkpoint_steps,
        sync_save=async_save)
    ckpt_callback = ModelCheckpoint(prefix=ckpt_prefix, directory=ckpt_path, config=ckpt_config)
    logger.info(f"MindSporeTest::clean all Checkpoint file under {ckpt_path}")
    clean_all_ckpt_files(ckpt_path)
    callbacks = [ckpt_callback]
    if callback is not None and isinstance(callback, Callback):
        if isinstance(callback, list):
            callbacks.extend([cb for cb in callback if isinstance(cb, Callback)])
        else:
            callbacks.append(callback)
    logger.info(f"MindSporeTest::Model train and save checkpoint under {ckpt_path}")
    model.train(epoch=epoch, train_dataset=dataset, dataset_sink_mode=dataset_sink_mode,
                callbacks=callbacks, sink_size=sink_size)
    logger.info("MindSporeTest::load the newest checkpoint file and return")
    newest_ckpt_file = find_newest_ckpt_file_by_name(ckpt_path)
    return load_checkpoint(newest_ckpt_file)


def test_parallel_comm_switch_nic_01():
    init()
    ms.set_seed(666)
    standalone_net = SimpleNet(mul_size=(128, 32))
    standalone_dataset = FakeData(size=1280, batch_size=128, image_size=(32,), num_classes=32)
    standalone_model = modeltrainbase.create_train_model(standalone_net, loss=None)
    standalone_ckpt = load_newest_ckpt_from_model_train(
        standalone_model, epoch=1, dataset=standalone_dataset, callback=None, dataset_sink_mode=False,
        ckpt_path="./rank_{}_ckpt".format(contextbase.get_parallel_variable_from_env("RANK_ID")),
        ckpt_prefix="ckpt_standalone")
    init()
    layout = Layout((2, 4), ("dp", "mp"))
    in_strategy1 = (layout("dp", "mp"), layout("dp", "mp"))
    in_strategy2 = (layout("mp", "dp"), layout("mp", "dp"))
    os.environ['MS_ENABLE_TFT'] = "{TSP:1}"
    callback = CommSwitchNic([0, 1, 4], [False, False, True], [5])
    parallel_net = AutoParallel(SimpleNet(mul_size=(128, 32), in_strategy1=in_strategy1, in_strategy2=in_strategy2),
                                parallel_mode="semi_auto")
    parallel_dataset = FakeData(size=1280, batch_size=16, image_size=(32,), num_classes=32, use_parallel=True)
    parallel_model = modeltrainbase.create_train_model(parallel_net, loss=None)
    parallel_ckpt = load_newest_ckpt_from_model_train(
        parallel_model, epoch=1, dataset=parallel_dataset, callback=callback, dataset_sink_mode=False,
        ckpt_path="./rank_{}_ckpt".format(contextbase.get_parallel_variable_from_env("RANK_ID")),
        ckpt_prefix="ckpt_parallel")
    net_cmp = SimpleNet(mul_size=(128, 32))
    inputs_np = Tensor(np.random.randn(128, 32).astype(np.float32))
    label = Tensor(np.random.randn(128, 32).astype(np.float32))
    comparebase.compare_checkpoint_dict(net_cmp, standalone_ckpt, parallel_ckpt, inputs_np, label)


def test_parallel_comm_switch_nic_02():
    init()
    ms.set_seed(666)
    standalone_net = SimpleNet(mul_size=(128, 32))
    standalone_dataset = FakeData(size=2560, batch_size=128, image_size=(32,), num_classes=32)
    standalone_model = modeltrainbase.create_train_model(standalone_net, loss=None)
    standalone_ckpt = load_newest_ckpt_from_model_train(
        standalone_model, epoch=1, dataset=standalone_dataset, callback=None, dataset_sink_mode=False,
        ckpt_path="./rank_{}_ckpt".format(contextbase.get_parallel_variable_from_env("RANK_ID")),
        ckpt_prefix="ckpt_standalone")
    init()
    layout = Layout((2, 4, 1), ("dp", "mp", "sp"))
    in_strategy1 = (layout("dp", "mp"), layout("dp", "mp"))
    in_strategy2 = (layout("mp", "sp"), layout("mp", "sp"))
    os.environ['MS_ENABLE_TFT'] = "{TSP:1}"
    callback = CommSwitchNic([0, 1, 2, 3, 4, 5, 6, 7],
                             [True, True, True, True, True, True, True, True], [10, 12])
    parallel_net = AutoParallel(SimpleNet(mul_size=(128, 32), in_strategy1=in_strategy1, in_strategy2=in_strategy2),
                                parallel_mode="semi_auto")
    parallel_dataset = FakeData(size=2560, batch_size=16, image_size=(32,), num_classes=32, use_parallel=True)
    parallel_model = modeltrainbase.create_train_model(parallel_net, loss=None)
    parallel_ckpt = load_newest_ckpt_from_model_train(
        parallel_model, epoch=1, dataset=parallel_dataset, callback=callback, dataset_sink_mode=False,
        ckpt_path="./rank_{}_ckpt".format(contextbase.get_parallel_variable_from_env("RANK_ID")),
        ckpt_prefix="ckpt_parallel")
    net_cmp = SimpleNet(mul_size=(128, 32))
    inputs_np = Tensor(np.random.randn(128, 32).astype(np.float32))
    label = Tensor(np.random.randn(128, 32).astype(np.float32))
    comparebase.compare_checkpoint_dict(net_cmp, standalone_ckpt, parallel_ckpt, inputs_np, label)


def test_parallel_comm_switch_nic_03():
    init()
    ms.set_seed(666)
    standalone_net = SimpleNet(mul_size=(128, 32))
    standalone_dataset = FakeData(size=2560, batch_size=128, image_size=(32,), num_classes=32)
    standalone_model = modeltrainbase.create_train_model(standalone_net, loss=None)
    standalone_ckpt = load_newest_ckpt_from_model_train(
        standalone_model, epoch=1, dataset=standalone_dataset, callback=None, dataset_sink_mode=False,
        ckpt_path="./rank_{}_ckpt".format(contextbase.get_parallel_variable_from_env("RANK_ID")),
        ckpt_prefix="ckpt_standalone")
    init()
    layout = Layout((2, 2, 2), ("dp", "mp", "sp"))
    in_strategy1 = (layout("dp", "sp"), layout("dp", "sp"))
    in_strategy2 = (layout("dp", "mp"), layout("dp", "mp"))
    os.environ['MS_ENABLE_TFT'] = "{TSP:1}"
    callback = CommSwitchNic([0, 7],
                             [True, True], [11, 12])
    parallel_net = AutoParallel(SimpleNet(mul_size=(128, 32), in_strategy1=in_strategy1, in_strategy2=in_strategy2),
                                parallel_mode="semi_auto")
    parallel_dataset = FakeData(size=2560, batch_size=16, image_size=(32,), num_classes=32, use_parallel=True)
    parallel_model = modeltrainbase.create_train_model(parallel_net, loss=None)
    parallel_ckpt = load_newest_ckpt_from_model_train(
        parallel_model, epoch=1, dataset=parallel_dataset, callback=callback, dataset_sink_mode=False,
        ckpt_path="./rank_{}_ckpt".format(contextbase.get_parallel_variable_from_env("RANK_ID")),
        ckpt_prefix="ckpt_parallel")
    net_cmp = SimpleNet(mul_size=(128, 32))
    inputs_np = Tensor(np.random.randn(128, 32).astype(np.float32))
    label = Tensor(np.random.randn(128, 32).astype(np.float32))
    comparebase.compare_checkpoint_dict(net_cmp, standalone_ckpt, parallel_ckpt, inputs_np, label)


def test_parallel_comm_switch_nic_04():
    init()
    ms.set_seed(666)
    standalone_net = SimpleNet(mul_size=(128, 32))
    standalone_dataset = FakeData(size=2560, batch_size=128, image_size=(32,), num_classes=32)
    standalone_model = modeltrainbase.create_train_model(standalone_net, loss=None)
    standalone_ckpt = load_newest_ckpt_from_model_train(
        standalone_model, epoch=1, dataset=standalone_dataset, callback=None, dataset_sink_mode=False,
        ckpt_path="./rank_{}_ckpt".format(contextbase.get_parallel_variable_from_env("RANK_ID")),
        ckpt_prefix="ckpt_standalone")
    init()
    os.environ['MS_ENABLE_TFT'] = "{TSP:1}"
    callback = CommSwitchNic([0, 6, 7],
                             [True, True, True], [11, 12])
    parallel_net = AutoParallel(SimpleNet(mul_size=(128, 32)), parallel_mode="semi_auto")
    parallel_dataset = FakeData(size=2560, batch_size=16, image_size=(32,), num_classes=32, use_parallel=True)
    parallel_model = modeltrainbase.create_train_model(parallel_net, loss=None)
    parallel_ckpt = load_newest_ckpt_from_model_train(
        parallel_model, epoch=1, dataset=parallel_dataset, callback=callback, dataset_sink_mode=False,
        ckpt_path="./rank_{}_ckpt".format(contextbase.get_parallel_variable_from_env("RANK_ID")),
        ckpt_prefix="ckpt_parallel")
    net_cmp = SimpleNet(mul_size=(128, 32))
    inputs_np = Tensor(np.random.randn(128, 32).astype(np.float32))
    label = Tensor(np.random.randn(128, 32).astype(np.float32))
    comparebase.compare_checkpoint_dict(net_cmp, standalone_ckpt, parallel_ckpt, inputs_np, label)


def test_parallel_comm_switch_nic_05():
    init()
    ms.set_seed(666)
    standalone_net = SimpleNet(mul_size=(128, 32))
    standalone_dataset = FakeData(size=2560, batch_size=128, image_size=(32,), num_classes=32)
    standalone_model = modeltrainbase.create_train_model(standalone_net, loss=None)
    standalone_ckpt = load_newest_ckpt_from_model_train(
        standalone_model, epoch=1, dataset=standalone_dataset, callback=None, dataset_sink_mode=False,
        ckpt_path="./rank_{}_ckpt".format(contextbase.get_parallel_variable_from_env("RANK_ID")),
        ckpt_prefix="ckpt_standalone")
    init()
    layout = Layout((2, 2, 2), ("dp", "mp", "sp"))
    in_strategy1 = (layout(("dp", "mp"), "sp"), layout(("dp", "mp"), "sp"))
    in_strategy2 = (layout(("dp", "sp"), "mp"), layout(("dp", "sp"), "mp"))
    os.environ['MS_ENABLE_TFT'] = "{TSP:1}"
    callback = CommSwitchNic([4], [True], [9])
    parallel_net = AutoParallel(
        SimpleNet(mul_size=(128, 32), in_strategy1=in_strategy1, in_strategy2=in_strategy2),
        parallel_mode="sharding_propagation")
    parallel_dataset = FakeData(size=2560, batch_size=16, image_size=(32,), num_classes=32, use_parallel=True)
    parallel_model = modeltrainbase.create_train_model(parallel_net, loss=None)
    parallel_ckpt = load_newest_ckpt_from_model_train(
        parallel_model, epoch=1, dataset=parallel_dataset, callback=callback, save_checkpoint_steps=10,
        dataset_sink_mode=False,
        ckpt_path="./rank_{}_ckpt".format(contextbase.get_parallel_variable_from_env("RANK_ID")),
        ckpt_prefix="ckpt_parallel")
    net_cmp = SimpleNet(mul_size=(128, 32))
    inputs_np = Tensor(np.random.randn(128, 32).astype(np.float32))
    label = Tensor(np.random.randn(128, 32).astype(np.float32))
    comparebase.compare_checkpoint_dict(net_cmp, standalone_ckpt, parallel_ckpt, inputs_np, label)
