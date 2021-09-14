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
# ===========================================================================

'''
    Train Pix2Pix model including distributed training
'''

import os
import datetime
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.train.serialization import save_checkpoint
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore import context
from mindspore.context import ParallelMode
from src.models.loss import D_Loss, D_WithLossCell, G_Loss, G_WithLossCell, TrainOneStepCell
from src.models.pix2pix import Pix2Pix, get_generator, get_discriminator
from src.dataset.pix2pix_dataset import pix2pixDataset, create_train_dataset
from src.utils.config import get_args
from src.utils.tools import save_losses, save_image, get_lr

if __name__ == '__main__':

    args = get_args()

    # Preprocess the data for training
    dataset = pix2pixDataset(root_dir=args.train_data_dir)
    ds = create_train_dataset(dataset)
    print("ds:", ds.get_dataset_size())
    print("ds:", ds.get_col_names())
    print("ds.shape:", ds.output_shapes())

    steps_per_epoch = ds.get_dataset_size()
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    if args.device_target == 'Ascend':
        if args.run_distribute:
            print("Ascend distribute")
            context.set_context(device_id=int(os.getenv("DEVICE_ID")))
            device_num = args.device_num
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                              device_num=device_num)
            init()

            rank = get_rank()
        else:
            context.set_context(device_id=args.device_id)
    elif args.device_target == 'GPU':
        if args.run_distribute:
            print("GPU distribute")
            init()
            device_num = args.device_num
            context.set_auto_parallel_context(device_num=get_group_size(),
                                              parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
        else:
            context.set_context(device_id=args.device_id)
    netG = get_generator()
    netD = get_discriminator()

    pix2pix = Pix2Pix(generator=netG, discriminator=netD)

    d_loss_fn = D_Loss()
    g_loss_fn = G_Loss()
    d_loss_net = D_WithLossCell(backbone=pix2pix, loss_fn=d_loss_fn)
    g_loss_net = G_WithLossCell(backbone=pix2pix, loss_fn=g_loss_fn)

    d_opt = nn.Adam(pix2pix.netD.trainable_params(), learning_rate=get_lr(),
                    beta1=args.beta1, beta2=args.beta2, loss_scale=1)
    g_opt = nn.Adam(pix2pix.netG.trainable_params(), learning_rate=get_lr(),
                    beta1=args.beta1, beta2=args.beta2, loss_scale=1)

    train_net = TrainOneStepCell(loss_netD=d_loss_net, loss_netG=g_loss_net, optimizerD=d_opt, optimizerG=g_opt, sens=1)
    train_net.set_train()

    if not os.path.isdir(args.train_fakeimg_dir):
        os.makedirs(args.train_fakeimg_dir)
    if not os.path.isdir(args.loss_show_dir):
        os.makedirs(args.loss_show_dir)
    if not os.path.isdir(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)

    # Training loop
    G_losses = []
    D_losses = []

    data_loader = ds.create_dict_iterator(output_numpy=True, num_epochs=args.epoch_num)
    print("Starting Training Loop...")
    if args.run_distribute:
        rank = get_rank()
    for epoch in range(args.epoch_num):
        for i, data in enumerate(data_loader):
            input_image = Tensor(data["input_images"])
            target_image = Tensor(data["target_images"])

            dis_loss, gen_loss = train_net(input_image, target_image)

            if i % 100 == 0:
                print("================start===================")
                print("Date time: ", datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                print("epoch: ", epoch + 1, "/", args.epoch_num)
                print("step: ", i, "/", steps_per_epoch)
                print("Dloss: ", dis_loss)
                print("Gloss: ", gen_loss)
                print("=================end====================")

            # Save fake_imgs
            if i == steps_per_epoch - 1:
                fake_image = netG(input_image)
                if args.run_distribute:
                    fakeimg_path = args.train_fakeimg_dir + str(rank) + '/'
                    if not os.path.isdir(fakeimg_path):
                        os.makedirs(fakeimg_path)
                    save_image(fake_image, fakeimg_path + str(epoch + 1))
                else:
                    save_image(fake_image, args.train_fakeimg_dir + str(epoch + 1))
                print("image generated from epoch", epoch + 1, "saved")
                print("The learning rate at this point is：", get_lr()[epoch*i])

            D_losses.append(dis_loss.asnumpy())
            G_losses.append(gen_loss.asnumpy())

        print("epoch", epoch + 1, "saved")
        # Save losses
        save_losses(G_losses, D_losses, epoch + 1)
        print("epoch", epoch + 1, "D&G_Losses saved")
        print("epoch", epoch + 1, "finished")
        # Save checkpoint
        if (epoch+1) % 50 == 0:
            if args.run_distribute:
                save_checkpoint_path = args.ckpt_dir + str(rank) + '/'
                if not os.path.isdir(save_checkpoint_path):
                    os.makedirs(save_checkpoint_path)
                save_checkpoint(netG, os.path.join(save_checkpoint_path, f"Generator_{epoch+1}.ckpt"))
            else:
                save_checkpoint(netG, os.path.join(args.ckpt_dir, f"Generator_{epoch+1}.ckpt"))
            print("ckpt generated from epoch", epoch + 1, "saved")
