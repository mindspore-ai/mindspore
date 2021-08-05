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
"""Train api."""
import os
import time
import numpy as np

import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.nn import Momentum
from mindspore.nn.optim import Lamb
from mindspore.train.model import Model
from mindspore.train.loss_scale_manager import DynamicLossScaleManager
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, SummaryCollector, TimeMonitor
from mindspore import context, Parameter
from mindspore.context import ParallelMode
from mindspore.communication import management as MultiDevice
from mindspore.train.serialization import load_checkpoint
from mindspore.common import set_seed

from src.dataset import load_dataset
from src.gnmt_model import GNMTNetworkWithLoss, GNMTTrainOneStepWithLossScaleCell
from src.utils import LossCallBack
from src.utils import one_weight, weight_variable
from src.utils.lr_scheduler import square_root_schedule, polynomial_decay_scheduler, Warmup_MultiStepLR_scheduler
from src.utils.optimizer import Adam
from src.utils.get_config import get_config

from model_utils.config import config as default_config
from model_utils.moxing_adapter import moxing_wrapper
from model_utils.device_adapter import get_device_id, get_device_num

def _train(model, config,
           pre_training_dataset=None, fine_tune_dataset=None, test_dataset=None,
           callbacks: list = None):
    """
    Train model.

    Args:
        model (Model): MindSpore model instance.
        config: Config of mass model.
        pre_training_dataset (Dataset): Pre-training dataset.
        fine_tune_dataset (Dataset): Fine-tune dataset.
        test_dataset (Dataset): Test dataset.
        callbacks (list): A list of callbacks.
    """
    callbacks = callbacks if callbacks else []

    if pre_training_dataset is not None:
        print(" | Start pre-training job.")
        epoch_size = pre_training_dataset.get_repeat_count()
        print("epoch size ", epoch_size)
        if os.getenv("RANK_SIZE") is not None and int(os.getenv("RANK_SIZE")) > 1:
            print(f" | Rank {MultiDevice.get_rank()} Call model train.")
        model.train(config.epochs, pre_training_dataset,
                    callbacks=callbacks, dataset_sink_mode=config.dataset_sink_mode)

    if fine_tune_dataset is not None:
        print(" | Start fine-tuning job.")
        epoch_size = fine_tune_dataset.get_repeat_count()

        model.train(config.epochs, fine_tune_dataset,
                    callbacks=callbacks, dataset_sink_mode=config.dataset_sink_mode)


def _load_checkpoint_to_net(config, network):
    """load parameters to network from checkpoint."""
    if config.existed_ckpt:
        if config.existed_ckpt.endswith(".npz"):
            weights = np.load(config.existed_ckpt)
        else:
            weights = load_checkpoint(config.existed_ckpt)
        for param in network.trainable_params():
            weights_name = param.name
            if weights_name not in weights:
                raise ValueError(f"Param {weights_name} is not found in ckpt file.")

            if isinstance(weights[weights_name], Parameter):
                param.set_data(weights[weights_name].data)
            elif isinstance(weights[weights_name], Tensor):
                param.set_data(Tensor(weights[weights_name].asnumpy(), config.dtype))
            elif isinstance(weights[weights_name], np.ndarray):
                param.set_data(Tensor(weights[weights_name], config.dtype))
            else:
                param.set_data(weights[weights_name])
    else:
        for param in network.trainable_params():
            name = param.name
            value = param.data
            if isinstance(value, Tensor):
                if name.endswith(".gamma"):
                    param.set_data(one_weight(value.asnumpy().shape))
                elif name.endswith(".beta") or name.endswith(".bias"):
                    if param.data.dtype == "Float32":
                        param.set_data((weight_variable(value.asnumpy().shape).astype(np.float32)))
                    elif param.data.dtype == "Float16":
                        param.set_data((weight_variable(value.asnumpy().shape).astype(np.float16)))
                else:
                    if param.data.dtype == "Float32":
                        param.set_data(Tensor(weight_variable(value.asnumpy().shape).astype(np.float32)))
                    elif param.data.dtype == "Float16":
                        param.set_data(Tensor(weight_variable(value.asnumpy().shape).astype(np.float16)))


def _get_lr(config, update_steps):
    """generate learning rate."""
    if config.lr_scheduler == "isr":
        lr = Tensor(square_root_schedule(lr=config.lr,
                                         update_num=update_steps,
                                         decay_start_step=config.decay_start_step,
                                         warmup_steps=config.warmup_steps,
                                         min_lr=config.min_lr), dtype=mstype.float32)
    elif config.lr_scheduler == "poly":
        lr = Tensor(polynomial_decay_scheduler(lr=config.lr,
                                               min_lr=config.min_lr,
                                               decay_steps=config.decay_steps,
                                               total_update_num=update_steps,
                                               warmup_steps=config.warmup_steps,
                                               power=config.lr_scheduler_power), dtype=mstype.float32)
    elif config.lr_scheduler == "WarmupMultiStepLR":
        lr = Tensor(Warmup_MultiStepLR_scheduler(base_lr=config.lr,
                                                 total_update_num=update_steps,
                                                 warmup_steps=config.warmup_steps,
                                                 remain_steps=config.warmup_lr_remain_steps,
                                                 decay_interval=config.warmup_lr_decay_interval,
                                                 decay_steps=config.decay_steps,
                                                 decay_factor=config.lr_scheduler_power), dtype=mstype.float32)
    else:
        lr = config.lr
    return lr


def _get_optimizer(config, network, lr):
    """get gnmt optimizer, support Adam, Lamb, Momentum."""
    if config.optimizer.lower() == "adam":
        optimizer = Adam(network.trainable_params(), lr, beta1=0.9, beta2=0.98)
    elif config.optimizer.lower() == "lamb":
        optimizer = Lamb(network.trainable_params(), learning_rate=lr,
                         eps=1e-6)
    elif config.optimizer.lower() == "momentum":
        optimizer = Momentum(network.trainable_params(), lr, momentum=0.9)
    else:
        raise ValueError(f"optimizer only support `adam` and `momentum` now.")

    return optimizer


def _build_training_pipeline(config,
                             pre_training_dataset=None,
                             fine_tune_dataset=None,
                             test_dataset=None):
    """
    Build training pipeline.

    Args:
        config: Config of mass model.
        pre_training_dataset (Dataset): Pre-training dataset.
        fine_tune_dataset (Dataset): Fine-tune dataset.
        test_dataset (Dataset): Test dataset.
    """
    net_with_loss = GNMTNetworkWithLoss(config, is_training=True, use_one_hot_embeddings=True)
    net_with_loss.init_parameters_data()
    _load_checkpoint_to_net(config, net_with_loss)

    dataset = pre_training_dataset if pre_training_dataset is not None \
        else fine_tune_dataset

    if dataset is None:
        raise ValueError("pre-training dataset or fine-tuning dataset must be provided one.")

    update_steps = config.epochs * dataset.get_dataset_size()

    lr = _get_lr(config, update_steps)
    optimizer = _get_optimizer(config, net_with_loss, lr)

    # Dynamic loss scale.
    scale_manager = DynamicLossScaleManager(init_loss_scale=config.init_loss_scale,
                                            scale_factor=config.loss_scale_factor,
                                            scale_window=config.scale_window)
    net_with_grads = GNMTTrainOneStepWithLossScaleCell(
        network=net_with_loss, optimizer=optimizer,
        scale_update_cell=scale_manager.get_update_cell()
    )
    net_with_grads.set_train(True)
    model = Model(net_with_grads)
    loss_monitor = LossCallBack(config)
    dataset_size = dataset.get_dataset_size()
    time_cb = TimeMonitor(data_size=dataset_size)
    ckpt_config = CheckpointConfig(save_checkpoint_steps=config.save_ckpt_steps,
                                   keep_checkpoint_max=config.keep_ckpt_max)

    rank_size = os.getenv('RANK_SIZE')
    callbacks = [time_cb, loss_monitor]
    if rank_size is not None and int(rank_size) > 1 and MultiDevice.get_rank() % 8 == 0:
        ckpt_callback = ModelCheckpoint(
            prefix=config.ckpt_prefix,
            directory=os.path.join(config.ckpt_path, 'ckpt_{}'.format(MultiDevice.get_rank())),
            config=ckpt_config)
        callbacks.append(ckpt_callback)
        summary_callback = SummaryCollector(summary_dir="./summary", collect_freq=50)
        callbacks.append(summary_callback)

    if rank_size is None or int(rank_size) == 1:
        ckpt_callback = ModelCheckpoint(
            prefix=config.ckpt_prefix,
            directory=os.path.join(config.ckpt_path, 'ckpt_{}'.format(config.device_id)),
            config=ckpt_config)
        callbacks.append(ckpt_callback)
        summary_callback = SummaryCollector(summary_dir="./summary", collect_freq=50)
        callbacks.append(summary_callback)

    print(f" | ALL SET, PREPARE TO TRAIN.")
    _train(model=model, config=config,
           pre_training_dataset=pre_training_dataset,
           fine_tune_dataset=fine_tune_dataset,
           test_dataset=test_dataset,
           callbacks=callbacks)


def _setup_parallel_env():
    context.reset_auto_parallel_context()
    MultiDevice.init()
    context.set_auto_parallel_context(
        parallel_mode=ParallelMode.DATA_PARALLEL,
        device_num=MultiDevice.get_group_size(),
        gradients_mean=True
    )


def train_parallel(config):
    """
    Train model with multi ascend chips.

    Args:
        config: Config for MASS model.
    """
    _setup_parallel_env()
    print(f" | Starting training on {os.getenv('RANK_SIZE', None)} devices.")

    pre_train_dataset = load_dataset(
        data_files=config.pre_train_dataset,
        batch_size=config.batch_size,
        sink_mode=config.dataset_sink_mode,
        rank_size=MultiDevice.get_group_size(),
        rank_id=MultiDevice.get_rank()
    ) if config.pre_train_dataset else None
    fine_tune_dataset = load_dataset(
        data_files=config.fine_tune_dataset,
        batch_size=config.batch_size,
        sink_mode=config.dataset_sink_mode,
        rank_size=MultiDevice.get_group_size(),
        rank_id=MultiDevice.get_rank()
    ) if config.fine_tune_dataset else None
    test_dataset = load_dataset(
        data_files=config.test_dataset,
        batch_size=config.batch_size,
        sink_mode=config.dataset_sink_mode,
        rank_size=MultiDevice.get_group_size(),
        rank_id=MultiDevice.get_rank()
    ) if config.test_dataset else None

    _build_training_pipeline(config=config,
                             pre_training_dataset=pre_train_dataset,
                             fine_tune_dataset=fine_tune_dataset,
                             test_dataset=test_dataset)


def train_single(config):
    """
    Train model on single device.

    Args:
        config: Config for model.
    """
    print(" | Starting training on single device.")

    pre_train_dataset = load_dataset(data_files=config.pre_train_dataset,
                                     batch_size=config.batch_size,
                                     sink_mode=config.dataset_sink_mode) if config.pre_train_dataset else None
    fine_tune_dataset = load_dataset(data_files=config.fine_tune_dataset,
                                     batch_size=config.batch_size,
                                     sink_mode=config.dataset_sink_mode) if config.fine_tune_dataset else None
    test_dataset = load_dataset(data_files=config.test_dataset,
                                batch_size=config.batch_size,
                                sink_mode=config.dataset_sink_mode) if config.test_dataset else None

    _build_training_pipeline(config=config,
                             pre_training_dataset=pre_train_dataset,
                             fine_tune_dataset=fine_tune_dataset,
                             test_dataset=test_dataset)


def modelarts_pre_process():
    '''modelarts pre process function.'''
    def unzip(zip_file, save_dir):
        import zipfile
        s_time = time.time()
        if not os.path.exists(os.path.join(save_dir, default_config.modelarts_dataset_unzip_name)):
            zip_isexist = zipfile.is_zipfile(zip_file)
            if zip_isexist:
                fz = zipfile.ZipFile(zip_file, 'r')
                data_num = len(fz.namelist())
                print("Extract Start...")
                print("unzip file num: {}".format(data_num))
                data_print = int(data_num / 100) if data_num > 100 else 1
                i = 0
                for file in fz.namelist():
                    if i % data_print == 0:
                        print("unzip percent: {}%".format(int(i * 100 / data_num)), flush=True)
                    i += 1
                    fz.extract(file, save_dir)
                print("cost time: {}min:{}s.".format(int((time.time() - s_time) / 60),
                                                     int(int(time.time() - s_time) % 60)))
                print("Extract Done.")
            else:
                print("This is not zip.")
        else:
            print("Zip has been extracted.")

    if default_config.need_modelarts_dataset_unzip:
        zip_file_1 = os.path.join(default_config.data_path, default_config.modelarts_dataset_unzip_name + ".zip")
        save_dir_1 = os.path.join(default_config.data_path)

        sync_lock = "/tmp/unzip_sync.lock"

        # Each server contains 8 devices as most.
        if get_device_id() % min(get_device_num(), 8) == 0 and not os.path.exists(sync_lock):
            print("Zip file path: ", zip_file_1)
            print("Unzip file save dir: ", save_dir_1)
            unzip(zip_file_1, save_dir_1)
            print("===Finish extract data synchronization===")
            try:
                os.mknod(sync_lock)
            except IOError:
                pass

        while True:
            if os.path.exists(sync_lock):
                break
            time.sleep(1)

        print("Device: {}, Finish sync unzip data from {} to {}.".format(get_device_id(), zip_file_1, save_dir_1))

    default_config.ckpt_path = os.path.join(default_config.output_path, default_config.ckpt_path)


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_train():
    '''run train.'''
    _config = get_config(default_config)
    _config.pre_train_dataset = default_config.pre_train_dataset

    context.set_context(mode=context.GRAPH_MODE, save_graphs=False, device_target=_config.device_target,
                        reserve_class_name_in_scope=True, device_id=_config.device_id)
    _rank_size = os.getenv('RANK_SIZE')
    set_seed(_config.random_seed)
    if _rank_size is not None and int(_rank_size) > 1:
        train_parallel(_config)
    else:
        train_single(_config)

if __name__ == '__main__':
    run_train()
