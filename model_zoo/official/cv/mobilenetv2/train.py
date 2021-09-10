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
"""Train mobilenetV2 on ImageNet."""

import os
import time
import random
import numpy as np

from mindspore import context
from mindspore import Tensor
from mindspore.nn import WithLossCell, TrainOneStepCell
from mindspore.nn.optim.momentum import Momentum
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.train.model import Model
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.serialization import save_checkpoint
from mindspore.common import set_seed

from src.dataset import create_dataset, extract_features
from src.lr_generator import get_lr
from src.utils import context_device_init, config_ckpoint
from src.models import CrossEntropyWithLabelSmooth, define_net, load_ckpt
from src.metric import DistAccuracy, ClassifyCorrectCell
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.device_adapter import get_device_id, get_device_num


set_seed(1)

def modelarts_pre_process():
    def unzip(zip_file, save_dir):
        import zipfile
        s_time = time.time()
        if not os.path.exists(os.path.join(save_dir, config.modelarts_dataset_unzip_name)):
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
                print("cost time: {}min:{}s.".format(int((time.time() - s_time) / 60),\
                    int(int(time.time() - s_time) % 60)))
                print("Extract Done")
            else:
                print("This is not zip.")
        else:
            print("Zip has been extracted.")

    if config.need_modelarts_dataset_unzip:
        zip_file_1 = os.path.join(config.data_path, config.modelarts_dataset_unzip_name + ".zip")
        save_dir_1 = os.path.join(config.data_path)

        sync_lock = "/tmp/unzip_sync.lock"

        # Each server contains 8 devices as most
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
        print("#" * 200, os.listdir(save_dir_1))
        print("#" * 200, os.listdir(os.path.join(config.data_path, config.modelarts_dataset_unzip_name)))

        config.dataset_path = os.path.join(config.data_path, config.modelarts_dataset_unzip_name)
    config.pretrain_ckpt = os.path.join(config.output_path, config.pretrain_ckpt)

def build_params_groups(net):
    decayed_params = []
    no_decayed_params = []
    for param in net.trainable_params():
        if 'beta' not in param.name and 'gamma' not in param.name and 'bias' not in param.name:
            decayed_params.append(param)
        else:
            no_decayed_params.append(param)

    group_params = [{'params': decayed_params, 'weight_decay': config.weight_decay},
                    {'params': no_decayed_params},
                    {'order_params': net.trainable_params()}]
    return group_params


@moxing_wrapper(pre_process=modelarts_pre_process)
def train_mobilenetv2():
    config.train_dataset_path = os.path.join(config.dataset_path, 'train')
    config.eval_dataset_path = os.path.join(config.dataset_path, 'validation_preprocess')
    if not config.device_id:
        config.device_id = get_device_id()
    start = time.time()
    # set context and device init
    context_device_init(config)
    print('\nconfig: {} \n'.format(config))
    # define network
    backbone_net, head_net, net = define_net(config, config.is_training)
    dataset = create_dataset(dataset_path=config.train_dataset_path, do_train=True, config=config,
                             enable_cache=config.enable_cache, cache_session_id=config.cache_session_id)
    step_size = dataset.get_dataset_size()
    if config.platform == "GPU":
        context.set_context(enable_graph_kernel=True)
    if config.pretrain_ckpt:
        if config.freeze_layer == "backbone":
            load_ckpt(backbone_net, config.pretrain_ckpt, trainable=False)
            step_size = extract_features(backbone_net, config.train_dataset_path, config)
        elif config.filter_head:
            load_ckpt(backbone_net, config.pretrain_ckpt)
        else:
            load_ckpt(net, config.pretrain_ckpt)
    if step_size == 0:
        raise ValueError("The step_size of dataset is zero. Check if the images' count of train dataset is more \
            than batch_size in config.py")

    # define loss
    if config.label_smooth > 0:
        loss = CrossEntropyWithLabelSmooth(
            smooth_factor=config.label_smooth, num_classes=config.num_classes)
    else:
        loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    epoch_size = config.epoch_size

    # get learning rate
    lr = Tensor(get_lr(global_step=0,
                       lr_init=config.lr_init,
                       lr_end=config.lr_end,
                       lr_max=config.lr_max,
                       warmup_epochs=config.warmup_epochs,
                       total_epochs=epoch_size,
                       steps_per_epoch=step_size))
    metrics = {"acc"}
    dist_eval_network = None
    eval_dataset = None
    if config.run_eval:
        metrics = {'acc': DistAccuracy(batch_size=config.batch_size, device_num=config.rank_size)}
        dist_eval_network = ClassifyCorrectCell(net, config.run_distribute)
        eval_dataset = create_dataset(dataset_path=config.eval_dataset_path, do_train=False, config=config)
    if config.pretrain_ckpt == "" or config.freeze_layer != "backbone":
        if config.platform == "Ascend":
            loss_scale = FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)
            group_params = build_params_groups(net)
            opt = Momentum(group_params, lr, config.momentum, loss_scale=config.loss_scale)
            model = Model(net, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale,
                          metrics=metrics, eval_network=dist_eval_network,
                          amp_level="O2", keep_batchnorm_fp32=False,
                          boost_level=config.boost_mode)

        else:
            opt = Momentum(net.trainable_params(), lr, config.momentum, config.weight_decay)
            model = Model(net, loss_fn=loss, optimizer=opt, metrics=metrics, eval_network=dist_eval_network,
                          boost_level=config.boost_mode)
        cb = config_ckpoint(config, lr, step_size, model, eval_dataset)
        print("============== Starting Training ==============")
        model.train(epoch_size, dataset, callbacks=cb)
        print("============== End Training ==============")

    else:
        opt = Momentum(filter(lambda x: x.requires_grad, head_net.get_parameters()),
                       lr, config.momentum, config.weight_decay)

        network = WithLossCell(head_net, loss)
        network = TrainOneStepCell(network, opt)
        network.set_train()

        features_path = config.train_dataset_path + '_features'
        idx_list = list(range(step_size))
        rank = config.rank_id
        save_ckpt_path = os.path.join(config.save_checkpoint_path, 'ckpt_' + str(rank) + '/')
        if not os.path.isdir(save_ckpt_path):
            os.mkdir(save_ckpt_path)

        for epoch in range(epoch_size):
            random.shuffle(idx_list)
            epoch_start = time.time()
            losses = []
            for j in idx_list:
                feature = Tensor(np.load(os.path.join(features_path, f"feature_{j}.npy")))
                label = Tensor(np.load(os.path.join(features_path, f"label_{j}.npy")))
                losses.append(network(feature, label).asnumpy())
            epoch_mseconds = (time.time()-epoch_start) * 1000
            per_step_mseconds = epoch_mseconds / step_size
            print("epoch[{}/{}], iter[{}] cost: {:5.3f}, per step time: {:5.3f}, avg loss: {:5.3f}"\
            .format(epoch + 1, epoch_size, step_size, epoch_mseconds, per_step_mseconds, np.mean(np.array(losses))))
            if (epoch + 1) % config.save_checkpoint_epochs == 0:
                save_checkpoint(net, os.path.join(save_ckpt_path, f"mobilenetv2_{epoch+1}.ckpt"))
        print("total cost {:5.4f} s".format(time.time() - start))

    if config.enable_cache:
        print("Remember to shut down the cache server via \"cache_admin --stop\"")


if __name__ == '__main__':
    train_mobilenetv2()
