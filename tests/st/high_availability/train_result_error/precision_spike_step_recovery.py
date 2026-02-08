
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
""" precision_spike_step_recovery """
import os
import math
from enum import Enum
import pytest
import numpy as np
from mindspore import log as logger
from mindspore import nn, load_checkpoint, load_param_into_net, context
from mindspore import Tensor
from mindspore.dataset import GeneratorDataset
from mindspore.common.initializer import HeUniform
from mindspore.train.callback import TrainFaultTolerance, CheckpointConfig, ModelCheckpoint, Callback
from mindspore.communication import init, release, get_rank, get_group_size
from mindspore.communication.comm_func import barrier
from mindspore.train import Model
from mindspore.nn import SoftmaxCrossEntropyWithLogits, Momentum



class FakeDataInitMode(Enum):
    ZEROS_INIT = 0
    ONES_INIT = 1
    UNIQUE_INIT = 2


class ContextBase:
    def __init__(self):
        self.save_graphs_flag = 0
        self.save_graphs_path = '.'

    def case_prepare(self):
        logger.info("MindSporeTest::call case-prepare !!!")
        self.set_default_context()
        self.set_context_from_env()
        self.init_parallel()

    def case_cleanup(self):
        logger.info("MindSporeTest::call case-cleanup !!!")
        self.release_parallel()
        context.reset_ps_context()
        context.reset_auto_parallel_context()
        if self.save_graphs_flag:
            self.clean_all_ir_files(self.save_graphs_path)

    def set_context_from_env(self):
        mode_dict = {
            'GRAPH': context.GRAPH_MODE,
            'GRAPH_MODE': context.GRAPH_MODE,
            'CONTEXT.GRAPH_MODE': context.GRAPH_MODE,
            'PYNATIVE': context.PYNATIVE_MODE,
            'PYNATIVE_MODE': context.PYNATIVE_MODE,
            'CONTEXT.PYNATIVE_MODE': context.PYNATIVE_MODE
        }
        if 'CONTEXT_MODE' in os.environ:
            mode_key = os.environ['CONTEXT_MODE']
            if mode_key in mode_dict:
                context.set_context(mode=mode_dict[mode_key])
        if 'CONTEXT_DEVICE_TARGET' in os.environ:
            context.set_context(device_target=os.environ['CONTEXT_DEVICE_TARGET'])
        if 'CONTEXT_ENABLE_SPARSE' in os.environ:
            context.set_context(enable_sparse=True)
        if "CONTEXT_JIT_LEVEL" in os.environ:
            jit_level = os.environ["CONTEXT_JIT_LEVEL"]
            if jit_level in ("O0", "O1", "O2"):
                context.set_context(jit_config={"jit_level": jit_level})
            else:
                raise ValueError(f"CONTEXT_JIT_LEVEL must be O0/O1/O2, got {jit_level}")
        ssl_enable = os.environ.get('ENABLE_SSL', 'false').lower()
        if ssl_enable == "true":
            security_ctx = {
                "config_file_path": os.path.join(os.path.dirname(__file__), "config.json"),
                "enable_ssl": True,
                "client_password": "",
                "server_password": ""
            }
            context.set_ps_context(**security_ctx)
        else:
            context.set_ps_context(enable_ssl=False)

        logger.info("MindSporeTest::set context from env success!!!")

    def set_default_context(self):
        context.reset_auto_parallel_context()
        context.set_context(mode=context.GRAPH_MODE)
        if 'CONTEXT_DEVICE_TARGET' not in os.environ:
            os.environ['CONTEXT_DEVICE_TARGET'] = 'Ascend'
        context.set_context(device_target=os.environ['CONTEXT_DEVICE_TARGET'])
        logger.info("MindSporeTest::set context from default success!!!")

    def init_parallel(self):
        if 'RANK_SIZE' in os.environ and int(os.environ['RANK_SIZE']) > 1:
            device_target = context.get_context("device_target")
            if device_target == 'Ascend':
                init(backend_name='hccl')
            elif device_target == 'GPU':
                init(backend_name='nccl')
            else:
                init()
            logger.info("MindSporeTest::init parallel success!!!")
        else:
            logger.info("MindSporeTest::single device, skip parallel init")

    def release_parallel(self):
        if 'RANK_SIZE' in os.environ and int(os.environ['RANK_SIZE']) > 1:
            release()
            logger.info("MindSporeTest::release parallel success!!!")

    def get_parallel_variable_from_env(self, attr_key):
        if attr_key in os.environ:
            return int(os.environ[attr_key])
        if attr_key == "RANK_ID":
            return get_rank() if 'RANK_SIZE' in os.environ and int(os.environ['RANK_SIZE']) > 1 else 0
        if attr_key == "DEVICE_ID":
            device_target = context.get_context("device_target")
            return get_rank() if device_target == "GPU" else os.environ.get("DEVICE_ID", 0)
        return -1

    def set_parallel_context(self, parallel_mode, dataset_strategy="data_parallel",
                           search_mode="recursive_programming", device_num=None,
                           strategy_ckpt_config=None,
                           enable_parallel_optimizer=False, **kwargs):
        if strategy_ckpt_config is None:
            strategy_ckpt_config = {"save_file": "", "load_file": "", "only_trainable_params": True}
        context.reset_auto_parallel_context()
        if device_num is None:
            context.set_auto_parallel_context(
                parallel_mode=parallel_mode,
                strategy_ckpt_config=strategy_ckpt_config,
                search_mode=search_mode,
                dataset_strategy=dataset_strategy,
                enable_parallel_optimizer=enable_parallel_optimizer,
                **kwargs
            )
        else:
            context.set_auto_parallel_context(
                parallel_mode=parallel_mode,
                device_num=device_num,
                strategy_ckpt_config=strategy_ckpt_config,
                search_mode=search_mode,
                dataset_strategy=dataset_strategy,
                enable_parallel_optimizer=enable_parallel_optimizer,
                **kwargs
            )
        logger.info("MindSporeTest::set parallel context success!!!")

    def clean_all_ir_files(self, path):
        for file in os.listdir(path):
            if file.endswith('.ir') or file.endswith('.dat'):
                os.remove(os.path.join(path, file))


class ModelTrainBase:
    def __init__(self):
        pass

    def create_train_model(self, network, amp_level="O0", metrics=None, loss_scale_manager=None,
                           loss="default", opt=None):
        logger.info(f"MindSporeTest::create a model with amp_level={amp_level}")
        if loss == "default":
            loss = SoftmaxCrossEntropyWithLogits(reduction='mean')
        opt_fn = opt
        if opt_fn is None:
            optimizer = TrainFaultTolerance.get_optimizer_wrapper(Momentum)
            opt_fn = optimizer(learning_rate=0.01, momentum=0.9, params=network.get_parameters())
        model = Model(network=network, loss_fn=loss, optimizer=opt_fn, amp_level=amp_level,
                      metrics=metrics, loss_scale_manager=loss_scale_manager)
        return model


def clean_all_ckpt_files(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        return
    for file in os.listdir(folder_path):
        if file.endswith('.ckpt') or file.endswith('.meta'):
            os.remove(os.path.join(folder_path, file))


def find_newest_ckpt_file(folder_path, fmt="ckpt"):
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder {folder_path} not exists")
    ckpt_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(f'.{fmt}')]
    if not ckpt_files:
        raise FileNotFoundError(f"No {fmt} files in {folder_path}")
    return max(ckpt_files, key=os.path.getmtime)


class GeneratorFakeData:
    def __init__(self, size=1024, batch_size=32, image_size=(3, 224, 224),
                 num_classes=10, random_offset=0, use_parallel=False,
                 fakedata_mode=FakeDataInitMode.ONES_INIT, dtype=np.float32):
        self.size = size
        self.rank_batch_size = batch_size
        self.total_batch_size = self.rank_batch_size
        self.random_offset = random_offset
        self.image_size = image_size
        self.num_classes = num_classes
        self.rank_size = 1
        self.rank_id = 0
        self.batch_index = 0
        self.image_data_type = dtype
        self.label_data_type = dtype
        self.is_onehot = True
        self.fakedata_mode = fakedata_mode
        if use_parallel:
            if 'RANK_SIZE' in os.environ and int(os.environ['RANK_SIZE']) > 1:
                self.rank_size = get_group_size()
                self.rank_id = get_rank()
        self.total_batch_size = self.rank_batch_size * self.rank_size
        assert self.size % self.total_batch_size == 0
        self.total_batch_data_size = (self.rank_size, self.rank_batch_size) + image_size

    def get_dataset_size(self):
        return int(self.size / self.total_batch_size)

    def get_repeat_count(self):
        return 1

    def set_image_data_type(self, data_type):
        self.image_data_type = data_type

    def set_label_data_type(self, data_type):
        self.label_data_type = data_type

    def set_label_onehot(self, is_onehot=True):
        self.is_onehot = is_onehot

    def create_tuple_iterator(self):
        return self

    def __next__(self):
        batch_index = self.batch_index
        self.batch_index += 1
        if batch_index * self.total_batch_size >= self.size:
            raise StopIteration
        rng_state = np.random.get_state()
        np.random.seed(batch_index + self.random_offset)
        if self.fakedata_mode == FakeDataInitMode.ONES_INIT:
            img = np.ones(self.total_batch_data_size)
        elif self.fakedata_mode == FakeDataInitMode.ZEROS_INIT:
            img = np.zeros(self.total_batch_data_size)
        elif self.fakedata_mode == FakeDataInitMode.UNIQUE_INIT:
            total_size = np.prod(self.total_batch_data_size)
            img = np.reshape(np.arange(total_size) * 0.0001, self.total_batch_data_size)
        else:
            img = np.random.randn(*self.total_batch_data_size)
        target = np.random.randint(0, self.num_classes, size=(self.rank_size, self.rank_batch_size))
        np.random.set_state(rng_state)
        img = img[self.rank_id]
        target = target[self.rank_id]
        img_ret = img.astype(self.image_data_type)
        target_ret = target.astype(self.label_data_type)
        if self.is_onehot:
            target_onehot = np.zeros(shape=(self.rank_batch_size, self.num_classes))
            target_onehot[np.arange(self.rank_batch_size), target] = 1
            target_ret = target_onehot.astype(self.label_data_type)
        return img_ret, target_ret

    def __len__(self):
        return self.size // self.total_batch_size

    def __iter__(self):
        self.batch_index = 0
        return self

    def reset(self):
        self.batch_index = 0


def _get_loss_output(net_outputs):
    if isinstance(net_outputs, tuple):
        loss = net_outputs[0] if len(net_outputs) > 0 else None
        return loss, None, None, None, None
    return net_outputs, None, None, None, None


contextbase = ContextBase()
modeltrainbase = ModelTrainBase()


def setup_function():
    contextbase.case_prepare()


def teardown_function():
    contextbase.case_cleanup()


class PrecisionSpike(Callback):
    def __init__(self, error_steps, dataset_size):
        super().__init__()
        self.error_steps = error_steps
        self.dataset_size = dataset_size
        self.remove = []

    def on_train_step_end(self, run_context):
        cb_params = run_context.original_args()
        cur_step_num = (cb_params.cur_step_num - 1) % self.dataset_size + 1
        loss, _, _, _, _ = _get_loss_output(cb_params.net_outputs)
        cur_epoch_num = (cb_params.cur_step_num - 1) // self.dataset_size + 1
        logger.warning(f"epoch: {cur_epoch_num} step: {cur_step_num}, loss: {loss}")
        if cur_step_num in self.error_steps and cur_step_num not in self.remove:
            if 'RANK_SIZE' in os.environ and int(os.environ['RANK_SIZE']) > 1:
                barrier()
            self.remove.append(cur_step_num)
            raise RuntimeError("TREError occurred")


def load_newest_ckpt_from_model_train(model, epoch, dataset, callback, dataset_sink_mode=True,
                                      ckpt_path="./", ckpt_prefix="ckpt_ms", async_save=False,
                                      save_checkpoint_steps=10, sink_size=1,
                                      integrated_save=True):
    logger.info("MindSporeTest::configure Config to save Checkpoint")
    ckpt_config = CheckpointConfig(keep_checkpoint_max=5, integrated_save=integrated_save,
                                   save_checkpoint_steps=save_checkpoint_steps,
                                   async_save=async_save)
    ckpt_callback = ModelCheckpoint(prefix=ckpt_prefix, directory=ckpt_path, config=ckpt_config)
    trainfaulttolerance = TrainFaultTolerance(
        "./rank_{}_ckpt".format(contextbase.get_parallel_variable_from_env("RANK_ID")))
    logger.info(f"MindSporeTest::clean all Checkpoint file under {ckpt_path}")
    clean_all_ckpt_files(ckpt_path)
    logger.info(f"MindSporeTest::Model train and save checkpoint under {ckpt_path}")
    model.train(epoch=epoch, train_dataset=dataset, dataset_sink_mode=dataset_sink_mode,
                callbacks=[ckpt_callback, callback, trainfaulttolerance], sink_size=sink_size)
    logger.info("MindSporeTest::load the newest checkpoint file and return")
    newest_ckpt_file = find_newest_ckpt_file(ckpt_path)
    return load_checkpoint(newest_ckpt_file)


class Network1(nn.Cell):
    def __init__(self, strategy=None):
        super().__init__()
        weight_init = HeUniform(math.sqrt(5))
        bias1 = Tensor(np.full([512], 0.01, np.float32))
        bias2 = Tensor(np.full([512], 0.01, np.float32))
        bias3 = Tensor(np.full([10], 0.01, np.float32))
        self.flatten = nn.Flatten()
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.layer1 = nn.Dense(28 * 28, 512, weight_init=weight_init, bias_init=bias1)
        if strategy is not None:
            self.layer1.matmul.shard(in_strategy=strategy)
        self.layer2 = nn.Dense(512, 512, weight_init=weight_init, bias_init=bias2)
        self.layer3 = nn.Dense(512, 10, weight_init=weight_init, bias_init=bias3)

    def construct(self, x):
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        logits = self.layer3(x)
        return logits


def create_dataset(batch_size, image_size=(28, 28), num_classes=10, size=32, use_parallel=False):
    fake_dataset = GeneratorFakeData(size=size * batch_size, batch_size=batch_size,
                                     image_size=image_size, num_classes=num_classes, dtype=np.float32,
                                     use_parallel=use_parallel)
    ds_train = GeneratorDataset(fake_dataset, ["data", "label"])
    return ds_train


def test_precision_spike_step_recovery_01():
    contextbase.set_parallel_context(parallel_mode="semi_auto_parallel", device_num=8,
                                    dataset_strategy="full_batch")
    in_strategy = ((2, 1), (1, 1))
    parallel_net = Network1(strategy=in_strategy)
    parallel_dataset = create_dataset(batch_size=32, size=32)
    optimizer = TrainFaultTolerance.get_optimizer_wrapper(Momentum)
    opt_fn = optimizer(learning_rate=0.01, momentum=0.9, params=parallel_net.get_parameters())
    parallel_model1 = modeltrainbase.create_train_model(parallel_net, opt=opt_fn)
    callback = PrecisionSpike([6], 32)
    load_newest_ckpt_from_model_train(
        parallel_model1, epoch=32, dataset=parallel_dataset, callback=callback,
        ckpt_path=f"./rank_{contextbase.get_parallel_variable_from_env('RANK_ID')}_ckpt",
        ckpt_prefix="ckpt_parallel", integrated_save=True)


def test_precision_spike_step_recovery_02():
    contextbase.set_parallel_context(parallel_mode="semi_auto_parallel", device_num=8,
                                    dataset_strategy="full_batch")
    in_strategy = ((2, 4), (1, 4))
    parallel_net = Network1(strategy=in_strategy)
    parallel_dataset = create_dataset(batch_size=32, size=32)
    optimizer = TrainFaultTolerance.get_optimizer_wrapper(Momentum)
    opt_fn = optimizer(learning_rate=0.01, momentum=0.9, params=parallel_net.get_parameters())
    parallel_model1 = modeltrainbase.create_train_model(parallel_net, opt=opt_fn)
    callback = PrecisionSpike([6, 7, 8], 32)
    load_newest_ckpt_from_model_train(
        parallel_model1, epoch=32, dataset=parallel_dataset, callback=callback,
        ckpt_path=f"./rank_{contextbase.get_parallel_variable_from_env('RANK_ID')}_ckpt",
        ckpt_prefix="ckpt_parallel", integrated_save=True)


def test_precision_spike_step_recovery_03():
    contextbase.set_parallel_context(parallel_mode="semi_auto_parallel", device_num=8,
                                     pipeline_stages=2, dataset_strategy="full_batch", enable_parallel_optimizer=True,
                                     parallel_optimizer_config={"parallel_optimizer_threshold": 0,
                                                                "optimizer_weight_shard_size": 2})
    os.environ["MS_ENABLE_TFT"] = "TRE:2,TRE_SNAPSHOT_STEPS:1"
    in_strategy = ((2, 2), (1, 2))
    parallel_net = Network1(strategy=in_strategy)
    parallel_net.flatten.pipeline_stage = 0
    parallel_net.layer1.pipeline_stage = 0
    parallel_net.relu1.pipeline_stage = 0
    parallel_net.layer2.pipeline_stage = 0
    parallel_net.relu2.pipeline_stage = 1
    parallel_net.layer3.pipeline_stage = 1
    parallel_dataset = create_dataset(batch_size=32, size=32)
    parallel_net1 = nn.PipelineCell(
        nn.WithLossCell(parallel_net, nn.SoftmaxCrossEntropyWithLogits(reduction='mean')), 4)
    parallel_model1 = modeltrainbase.create_train_model(parallel_net1, loss=None)
    callback = PrecisionSpike([3, 4, 5], 32)
    load_newest_ckpt_from_model_train(
        parallel_model1, epoch=32, dataset=parallel_dataset, callback=callback,
        ckpt_path="./rank_{}_ckpt".format(contextbase.get_parallel_variable_from_env("RANK_ID")),
        ckpt_prefix="ckpt_parallel", integrated_save=False)


def test_precision_spike_step_recovery_04():
    contextbase.set_parallel_context(parallel_mode="semi_auto_parallel", device_num=8,
                                    dataset_strategy="full_batch")
    in_strategy = ((4, 2), (1, 2))
    parallel_net = Network1(strategy=in_strategy)
    parallel_dataset = create_dataset(batch_size=32, size=32)
    optimizer = TrainFaultTolerance.get_optimizer_wrapper(Momentum)
    opt_fn = optimizer(learning_rate=0.01, momentum=0.9, params=parallel_net.get_parameters())
    parallel_model1 = modeltrainbase.create_train_model(parallel_net, opt=opt_fn)
    callback = PrecisionSpike([4], 32)
    with pytest.raises(TypeError):
        load_newest_ckpt_from_model_train(
            parallel_model1, epoch=32, dataset=parallel_dataset, callback=callback,
            ckpt_path=f"./rank_{contextbase.get_parallel_variable_from_env('RANK_ID')}_ckpt",
            ckpt_prefix="ckpt_parallel", integrated_save=True)


def test_precision_spike_step_recovery_05():
    contextbase.set_parallel_context(parallel_mode="semi_auto_parallel", device_num=8,
                                    dataset_strategy="full_batch")
    in_strategy = ((4, 2), (1, 2))
    parallel_net = Network1(strategy=in_strategy)
    parallel_dataset = create_dataset(batch_size=32, size=32)
    optimizer = TrainFaultTolerance.get_optimizer_wrapper(Momentum)
    opt_fn = optimizer(learning_rate=0.01, momentum=0.9, params=parallel_net.get_parameters())
    parallel_model1 = modeltrainbase.create_train_model(parallel_net, opt=opt_fn)
    callback = PrecisionSpike([10], 32)
    load_newest_ckpt_from_model_train(
        parallel_model1, epoch=32, dataset=parallel_dataset, callback=callback,
        ckpt_path=f"./rank_{contextbase.get_parallel_variable_from_env('RANK_ID')}_ckpt",
        ckpt_prefix="ckpt_parallel", integrated_save=True)


def test_precision_spike_step_recovery_06():
    stra_ckpt_dict1 = {"save_file": "./strategy.ckpt", "only_trainable_params": False}
    contextbase.set_parallel_context(parallel_mode="semi_auto_parallel", device_num=8,
                                    dataset_strategy="full_batch", strategy_ckpt_config=stra_ckpt_dict1)
    parallel_net = Network1(strategy=None)
    parallel_dataset = create_dataset(batch_size=32, size=32)
    parallel_model1 = modeltrainbase.create_train_model(parallel_net)
    callback = PrecisionSpike([16], 32)
    load_newest_ckpt_from_model_train(
        parallel_model1, epoch=3, dataset=parallel_dataset, callback=callback, save_checkpoint_steps=1,
        ckpt_path=f"./rank_{contextbase.get_parallel_variable_from_env('RANK_ID')}_ckpt",
        ckpt_prefix="ckpt_parallel", integrated_save=True)

    contextbase.set_parallel_context(parallel_mode="semi_auto_parallel", device_num=8,
                                    dataset_strategy="full_batch")
    parallel_net = Network1()
    parallel_dataset = create_dataset(batch_size=32, size=32)
    ckpt_path = f"./rank_{contextbase.get_parallel_variable_from_env('RANK_ID')}_ckpt"
    newest_ckpt = find_newest_ckpt_file(ckpt_path)
    param_dict = load_checkpoint(newest_ckpt)
    load_param_into_net(parallel_net, param_dict)
    parallel_model1 = modeltrainbase.create_train_model(parallel_net)
    callback = PrecisionSpike([16], 32)
    load_newest_ckpt_from_model_train(
        parallel_model1, epoch=20, dataset=parallel_dataset, callback=callback,
        ckpt_path=ckpt_path,
        ckpt_prefix="ckpt_parallel", integrated_save=True)


def test_precision_spike_step_recovery_07():
    contextbase.set_parallel_context(parallel_mode="semi_auto_parallel", device_num=8,
                                    dataset_strategy="full_batch")
    in_strategy = ((4, 2), (1, 2))
    parallel_net = Network1(strategy=in_strategy)
    parallel_dataset = create_dataset(batch_size=32, size=128)
    optimizer = TrainFaultTolerance.get_optimizer_wrapper(Momentum)
    opt_fn = optimizer(learning_rate=0.01, momentum=0.9, params=parallel_net.get_parameters())
    parallel_model1 = modeltrainbase.create_train_model(parallel_net, opt=opt_fn)
    error_list = list(range(2, 129))
    callback = PrecisionSpike(error_list, 128)
    load_newest_ckpt_from_model_train(
        parallel_model1, epoch=128, dataset=parallel_dataset, callback=callback,
        ckpt_path=f"./rank_{contextbase.get_parallel_variable_from_env('RANK_ID')}_ckpt",
        ckpt_prefix="ckpt_parallel", integrated_save=True)
    