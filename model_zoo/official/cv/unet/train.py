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
import os
import logging

import mindspore
import mindspore.nn as nn
from mindspore import Model, context
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint
from mindspore.context import ParallelMode
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.unet_medical import UNetMedical
from src.unet_nested import NestedUNet, UNet
from src.data_loader import create_dataset, create_multi_class_dataset
from src.loss import CrossEntropyWithLogits, MultiCrossEntropyWithLogits
from src.utils import StepLossTimeMonitor, UnetEval, TempLoss, apply_eval, filter_checkpoint_parameter_by_list, dice_coeff
from src.eval_callback import EvalCallBack

from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.device_adapter import get_device_id

mindspore.set_seed(1)

@moxing_wrapper()
def train_net(cross_valid_ind=1,
              epochs=400,
              batch_size=16,
              lr=0.0001):
    rank = 0
    group_size = 1
    data_dir = config.data_path
    run_distribute = config.run_distribute
    if run_distribute:
        init()
        group_size = get_group_size()
        rank = get_rank()
        parallel_mode = ParallelMode.DATA_PARALLEL
        context.set_auto_parallel_context(parallel_mode=parallel_mode,
                                          device_num=group_size,
                                          gradients_mean=False)
    need_slice = False
    if config.model_name == 'unet_medical':
        net = UNetMedical(n_channels=config.num_channels, n_classes=config.num_classes)
    elif config.model_name == 'unet_nested':
        net = NestedUNet(in_channel=config.num_channels, n_class=config.num_classes, use_deconv=config.use_deconv,
                         use_bn=config.use_bn, use_ds=config.use_ds)
        need_slice = config.use_ds
    elif config.model_name == 'unet_simple':
        net = UNet(in_channel=config.num_channels, n_class=config.num_classes)
    else:
        raise ValueError("Unsupported model: {}".format(config.model_name))

    if config.resume:
        param_dict = load_checkpoint(config.resume_ckpt)
        if config.transfer_training:
            filter_checkpoint_parameter_by_list(param_dict, config.filter_weight)
        load_param_into_net(net, param_dict)

    if hasattr(config, "use_ds") and config.use_ds:
        criterion = MultiCrossEntropyWithLogits()
    else:
        criterion = CrossEntropyWithLogits()
    if hasattr(config, "dataset") and config.dataset != "ISBI":
        dataset_sink_mode = True
        per_print_times = 0
        repeat = config.repeat if hasattr(config, "repeat") else 1
        split = config.split if hasattr(config, "split") else 0.8
        python_multiprocessing = not (config.device_target == "GPU" and run_distribute)
        train_dataset = create_multi_class_dataset(data_dir, config.image_size, repeat, batch_size,
                                                   num_classes=config.num_classes, is_train=True, augment=True,
                                                   split=split, rank=rank, group_size=group_size, shuffle=True,
                                                   python_multiprocessing=python_multiprocessing)
        valid_dataset = create_multi_class_dataset(data_dir, config.image_size, 1, 1,
                                                   num_classes=config.num_classes, is_train=False,
                                                   eval_resize=config.eval_resize, split=split,
                                                   python_multiprocessing=False, shuffle=False)
    else:
        repeat = config.repeat
        dataset_sink_mode = False
        if config.device_target == "GPU":
            dataset_sink_mode = True
        per_print_times = 1
        train_dataset, valid_dataset = create_dataset(data_dir, repeat, batch_size, True, cross_valid_ind,
                                                      run_distribute, config.crop, config.image_size)
    train_data_size = train_dataset.get_dataset_size()
    print("dataset length is:", train_data_size)
    ckpt_save_dir = os.path.join(config.output_path, config.checkpoint_path)
    save_ck_steps = train_data_size
    if config.device_target == "GPU":
        save_ck_steps = train_data_size * epochs
    ckpt_config = CheckpointConfig(save_checkpoint_steps=save_ck_steps,
                                   keep_checkpoint_max=config.keep_checkpoint_max)
    ckpoint_cb = ModelCheckpoint(prefix='ckpt_{}_adam'.format(config.model_name),
                                 directory=ckpt_save_dir+'./ckpt_{}/'.format(rank),
                                 config=ckpt_config)

    optimizer = nn.Adam(params=net.trainable_params(), learning_rate=lr, weight_decay=config.weight_decay,
                        loss_scale=config.loss_scale)

    loss_scale_manager = mindspore.train.loss_scale_manager.FixedLossScaleManager(config.FixedLossScaleManager, False)
    amp_level = "O0" if config.device_target == "GPU" else "O3"
    model = Model(net, loss_fn=criterion, loss_scale_manager=loss_scale_manager, optimizer=optimizer,
                  amp_level=amp_level)
    print("============== Starting Training ==============")
    callbacks = [StepLossTimeMonitor(batch_size=batch_size, per_print_times=per_print_times), ckpoint_cb]
    if config.run_eval:
        eval_model = Model(UnetEval(net, need_slice=need_slice, eval_activate=config.eval_activate.lower()),
                           loss_fn=TempLoss(), metrics={"dice_coeff": dice_coeff(False, config.show_eval)})
        eval_param_dict = {"model": eval_model, "dataset": valid_dataset, "metrics_name": config.eval_metrics}
        eval_cb = EvalCallBack(apply_eval, eval_param_dict, interval=config.eval_interval,
                               eval_start_epoch=config.eval_start_epoch, save_best_ckpt=True,
                               ckpt_directory=ckpt_save_dir+'./ckpt_{}/'.format(rank), besk_ckpt_name="best.ckpt",
                               metrics_name=config.eval_metrics)
        callbacks.append(eval_cb)
    model.train(int(epochs / repeat), train_dataset, callbacks=callbacks, dataset_sink_mode=dataset_sink_mode)
    print("============== End Training ==============")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, save_graphs=False)
    if config.device_target == "Ascend":
        device_id = get_device_id()
        context.set_context(device_id=device_id)
    epoch_size = config.epochs if not config.run_distribute else config.distribute_epochs
    batchsize = config.batch_size
    if config.device_target == 'GPU' and config.run_distribute:
        batchsize = config.distribute_batchsize
    train_net(cross_valid_ind=config.cross_valid_ind,
              epochs=epoch_size,
              batch_size=batchsize,
              lr=config.lr)
