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
"""Face Recognition train."""
import os
import time

import mindspore
from mindspore.nn import Cell
from mindspore import context
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_group_size, get_rank
from mindspore.nn.optim import Momentum
from mindspore.train.model import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.train.loss_scale_manager import DynamicLossScaleManager
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.my_logging import get_logger
from src.init_network import init_net
from src.dataset_factory import get_de_dataset
from src.backbone.resnet import get_backbone
from src.metric_factory import get_metric_fc
from src.loss_factory import get_loss
from src.lrsche_factory import warmup_step_list, list_to_gen
from src.callback_factory import ProgressMonitor

from model_utils.moxing_adapter import moxing_wrapper
from model_utils.config import config
from model_utils.device_adapter import get_device_id, get_device_num, get_rank_id

mindspore.common.seed.set_seed(1)
context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, save_graphs=False,
                    reserve_class_name_in_scope=False, enable_graph_kernel=config.device_target == "GPU")


if config.device_target != 'GPU' or not config.is_distributed:
    context.set_context(device_id=get_device_id())

class DistributedHelper(Cell):
    '''DistributedHelper'''
    def __init__(self, backbone, margin_fc):
        super(DistributedHelper, self).__init__()
        self.backbone = backbone
        self.margin_fc = margin_fc
        if margin_fc is not None:
            self.has_margin_fc = 1
        else:
            self.has_margin_fc = 0

    def construct(self, x, label):
        embeddings = self.backbone(x)
        if self.has_margin_fc == 1:
            return embeddings, self.margin_fc(embeddings, label)
        return embeddings


class BuildTrainNetwork(Cell):
    '''BuildTrainNetwork'''
    def __init__(self, network, criterion, args_1):
        super(BuildTrainNetwork, self).__init__()
        self.network = network
        self.criterion = criterion
        self.args = args_1

        if int(args_1.model_parallel) == 0:
            self.is_model_parallel = 0
        else:
            self.is_model_parallel = 1

    def construct(self, input_data, label):

        if self.is_model_parallel == 0:
            _, output = self.network(input_data, label)
            loss = self.criterion(output, label)
        else:
            _ = self.network(input_data, label)
            loss = self.criterion(None, label)

        return loss


def load_pretrain(cfg, net):
    '''load pretrain function.'''
    if os.path.isfile(cfg.pretrained):
        param_dict = load_checkpoint(cfg.pretrained)
        param_dict_new = {}
        if cfg.train_stage.lower() == 'base':
            for key, value in param_dict.items():
                if key.startswith('moments.'):
                    continue
                elif key.startswith('network.'):
                    param_dict_new[key[8:]] = value
        else:
            for key, value in param_dict.items():
                if key.startswith('moments.'):
                    continue
                elif key.startswith('network.'):
                    if 'layers.' in key and 'bn1' in key:
                        continue
                    elif 'se' in key:
                        continue
                    elif 'head' in key:
                        continue
                    elif 'margin_fc.weight' in key:
                        continue
                    else:
                        param_dict_new[key[8:]] = value
        load_param_into_net(net, param_dict_new)
        cfg.logger.info('load model {} success'.format(cfg.pretrained))
    else:
        if cfg.train_stage.lower() == 'beta':
            raise ValueError("Train beta mode load pretrain model fail from: {}".format(cfg.pretrained))
        init_net(cfg, net)
        cfg.logger.info('init model success')
    return net


def modelarts_pre_process():
    '''modelarts pre process function.'''
    def unzip(zip_file, save_dir):
        import zipfile
        s_time = time.time()
        if not os.path.exists(os.path.join(save_dir, "face_recognition_dataset")):
            zip_isexist = zipfile.is_zipfile(zip_file)
            if zip_isexist:
                fz = zipfile.ZipFile(zip_file, 'r')
                data_num = len(fz.namelist())
                print("Extract Start...")
                print("unzip file num: {}".format(data_num))
                i = 0
                for file in fz.namelist():
                    if i % int(data_num / 100) == 0:
                        print("unzip percent: {}%".format(i / int(data_num / 100)), flush=True)
                    i += 1
                    fz.extract(file, save_dir)
                print("cost time: {}min:{}s.".format(int((time.time() - s_time) / 60),
                                                     int(int(time.time() - s_time) % 60)))
                print("Extract Done.")
            else:
                print("This is not zip.")
        else:
            print("Zip has been extracted.")

    if config.need_modelarts_dataset_unzip:
        zip_file_1 = os.path.join(config.data_path, "face_recognition_dataset.zip")
        save_dir_1 = os.path.join(config.data_path)

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

    config.ckpt_path = os.path.join(config.output_path, str(get_rank_id()), config.ckpt_path)

def model_context():
    """set context for facerecognition"""
    if config.is_distributed:
        parallel_mode = ParallelMode.HYBRID_PARALLEL if config.device_target == 'Ascend' else ParallelMode.DATA_PARALLEL
    else:
        parallel_mode = ParallelMode.STAND_ALONE
    if config.is_distributed:
        if config.device_target == 'Ascend':
            context.set_auto_parallel_context(parallel_mode=parallel_mode,
                                              device_num=config.world_size, gradients_mean=True)
            init()
            config.local_rank = get_rank_id()
            config.world_size = get_device_num()
        elif config.device_target == 'GPU':
            init()
            device_num = get_group_size()
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=device_num,
                                              parallel_mode=parallel_mode,
                                              gradients_mean=True)
            config.world_size = get_group_size()
            config.local_rank = get_rank()
        else:
            pass


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_train():
    '''run train function.'''
    model_context()
    log_path = os.path.join(config.ckpt_path, 'logs')
    config.logger = get_logger(log_path, config.local_rank)
    support_train_stage = ['base', 'beta']
    if config.train_stage.lower() not in support_train_stage:
        config.logger.info('your train stage is not support.')
        raise ValueError('train stage not support.')

    if not os.path.exists(config.data_dir):
        config.logger.info('ERROR, data_dir is not exists, please set data_dir in config.py')
        raise ValueError('ERROR, data_dir is not exists, please set data_dir in config.py')
    if config.local_rank % 8 == 0:
        if not os.path.exists(config.ckpt_path):
            os.makedirs(config.ckpt_path)

    de_dataset, steps_per_epoch, num_classes = get_de_dataset(config)
    config.logger.info('de_dataset: %d', de_dataset.get_dataset_size())

    config.steps_per_epoch = steps_per_epoch
    config.num_classes = num_classes
    config.lr_epochs = list(map(int, config.lr_epochs.split(',')))
    config.logger.info('config.num_classes: %d', config.num_classes)
    config.logger.info('config.world_size: %d', config.world_size)
    config.logger.info('config.local_rank: %d', config.local_rank)
    config.logger.info('config.lr: %f', config.lr)

    if config.nc_16 == 1:
        if config.model_parallel == 0:
            if config.num_classes % 16 == 0:
                config.logger.info('data parallel aleardy 16, nums: %d', config.num_classes)
            else:
                config.num_classes = (config.num_classes // 16 + 1) * 16
        else:
            if config.num_classes % (config.world_size * 16) == 0:
                config.logger.info('model parallel aleardy 16, nums: %d', config.num_classes)
            else:
                config.num_classes = (config.num_classes // (config.world_size * 16) + 1) * config.world_size * 16

    config.logger.info('for D, loaded, class nums: %d', config.num_classes)
    config.logger.info('steps_per_epoch: %d', config.steps_per_epoch)
    config.logger.info('img_total_num: %d', config.steps_per_epoch * config.per_batch_size)

    config.logger.info('get_backbone----in----')
    _backbone = get_backbone(config)
    config.logger.info('get_backbone----out----')
    config.logger.info('get_metric_fc----in----')
    margin_fc_1 = get_metric_fc(config)
    config.logger.info('get_metric_fc----out----')
    config.logger.info('DistributedHelper----in----')
    network_1 = DistributedHelper(_backbone, margin_fc_1)
    config.logger.info('DistributedHelper----out----')
    config.logger.info('network fp16----in----')
    if config.fp16 == 1:
        network_1.add_flags_recursive(fp16=True)
    config.logger.info('network fp16----out----')

    criterion_1 = get_loss(config)
    if config.fp16 == 1 and config.model_parallel == 0:
        criterion_1.add_flags_recursive(fp32=True)

    network_1 = load_pretrain(config, network_1)
    train_net = BuildTrainNetwork(network_1, criterion_1, config)

    # call warmup_step should behind the config steps_per_epoch
    config.lrs = warmup_step_list(config, gamma=0.1)
    lrs_gen = list_to_gen(config.lrs)
    opt = Momentum(params=train_net.trainable_params(), learning_rate=lrs_gen, momentum=config.momentum,
                   weight_decay=config.weight_decay)
    scale_manager = DynamicLossScaleManager(init_loss_scale=config.dynamic_init_loss_scale, scale_factor=2,
                                            scale_window=2000)
    if config.device_target in ("Ascend", "GPU"):
        model = Model(train_net, optimizer=opt, metrics=None, loss_scale_manager=scale_manager)
    elif config.device_target == "CPU":
        model = Model(train_net, optimizer=opt, metrics=None, loss_scale_manager=None)

    save_checkpoint_steps = config.ckpt_steps
    config.logger.info('save_checkpoint_steps: %d', save_checkpoint_steps)
    if config.max_ckpts == -1:
        keep_checkpoint_max = int(config.steps_per_epoch * config.max_epoch / save_checkpoint_steps) + 5
    else:
        keep_checkpoint_max = config.max_ckpts
    config.logger.info('keep_checkpoint_max: %d', keep_checkpoint_max)

    callback_list = []
    config.epoch_cnt = 0
    progress_cb = ProgressMonitor(config)
    callback_list.append(progress_cb)
    if config.local_rank % 8 == 0:
        ckpt_config = CheckpointConfig(save_checkpoint_steps=save_checkpoint_steps,
                                       keep_checkpoint_max=keep_checkpoint_max)
        config.logger.info('max_epoch_train: %d', config.max_epoch)
        ckpt_cb = ModelCheckpoint(config=ckpt_config, directory=config.ckpt_path, prefix='{}'.format(config.local_rank))
        callback_list.append(ckpt_cb)

    new_epoch_train = config.max_epoch * steps_per_epoch // config.log_interval
    model.train(new_epoch_train, de_dataset, callbacks=callback_list, sink_size=config.log_interval)


if __name__ == "__main__":
    run_train()
