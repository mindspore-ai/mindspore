# Copyright 2021 Huawei Technologies Co., Ltd
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

"""train DCGAN and get checkpoint files."""
import argparse
import ast
import os
import datetime
import numpy as np

from mindspore import context
from mindspore import nn, Tensor
from mindspore.train.callback import CheckpointConfig
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_group_size
from src.dataset import create_dataset_imagenet
from src.config import dcgan_imagenet_cfg as cfg
from src.generator import Generator
from src.discriminator import Discriminator
from src.cell import WithLossCellD, WithLossCellG, DcganModelCheckpoint
from src.dcgan import DCGAN

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MindSpore dcgan training')
    parser.add_argument("--run_distribute", type=ast.literal_eval, default=False,
                        help="Run distribute, default is false.")
    parser.add_argument('--device_id', type=int, default=0, help='device id of Ascend (Default: 0)')
    parser.add_argument('--dataset_path', type=str, default=None, help='dataset path')
    parser.add_argument('--save_path', type=str, default=None, help='checkpoint save path')
    args = parser.parse_args()

    if args.run_distribute:
        device_id = args.device_id
        context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=False)
        context.set_context(device_id=device_id)
        init()
        device_num = get_group_size()
        context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
    else:
        device_id = args.device_id
        context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=False)
        context.set_context(device_id=device_id)

    # Load Dataset
    ds = create_dataset_imagenet(os.path.join(args.dataset_path), num_parallel_workers=2)

    steps_per_epoch = ds.get_dataset_size()

    # Define Network
    netD = Discriminator()
    netG = Generator()

    criterion = nn.BCELoss(reduction='mean')

    netD_with_criterion = WithLossCellD(netD, netG, criterion)
    netG_with_criterion = WithLossCellG(netD, netG, criterion)

    optimizerD = nn.Adam(netD.trainable_params(), learning_rate=cfg.learning_rate, beta1=cfg.beta1)
    optimizerG = nn.Adam(netG.trainable_params(), learning_rate=cfg.learning_rate, beta1=cfg.beta1)

    myTrainOneStepCellForD = nn.TrainOneStepCell(netD_with_criterion, optimizerD)
    myTrainOneStepCellForG = nn.TrainOneStepCell(netG_with_criterion, optimizerG)

    dcgan = DCGAN(myTrainOneStepCellForD, myTrainOneStepCellForG)
    dcgan.set_train()

    # checkpoint save
    ckpt_config = CheckpointConfig(save_checkpoint_steps=steps_per_epoch,
                                   keep_checkpoint_max=cfg.epoch_size)
    ckpt_cb = DcganModelCheckpoint(config=ckpt_config, directory=args.save_path, prefix='dcgan')

    class CallbackParam(dict):
        """Internal callback object's parameters."""

        def __getattr__(self, key):
            return self[key]

        def __setattr__(self, key, value):
            self[key] = value

    cb_params = CallbackParam()
    cb_params.train_network = dcgan
    cb_params.batch_num = steps_per_epoch
    cb_params.epoch_num = cfg.epoch_size
    # For each epoch
    cb_params.cur_epoch_num = 0
    cb_params.cur_step_num = 0

    np.random.seed(1)
    fixed_noise = Tensor(np.random.normal(size=(16, cfg.latent_size, 1, 1)).astype("float32"))

    data_loader = ds.create_dict_iterator(output_numpy=True, num_epochs=cfg.epoch_size)
    G_losses = []
    D_losses = []
    # Start Training Loop
    print("Starting Training Loop...")
    for epoch in range(cfg.epoch_size):
        # For each batch in the dataloader
        for i, data in enumerate(data_loader):
            real_data = Tensor(data['image'])
            latent_code = Tensor(data["latent_code"])
            netD_loss, netG_loss = dcgan(real_data, latent_code)
            if i % 50 == 0:
                time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print("Date time: ", time, "\tepoch: ", epoch, "/", cfg.epoch_size, "\tstep: ", i,
                      "/", steps_per_epoch, "\tDloss: ", netD_loss, "\tGloss: ", netG_loss)
            D_losses.append(netD_loss.asnumpy())
            G_losses.append(netG_loss.asnumpy())
            cb_params.cur_step_num = cb_params.cur_step_num + 1
        cb_params.cur_epoch_num = cb_params.cur_epoch_num + 1
        print("================saving model===================")
        if args.device_id == 0 or not args.run_distribute:
            ckpt_cb.save_ckpt(cb_params, True)
        print("================success================")
