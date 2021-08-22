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
    Train Pix2Pix model
'''

import os
import datetime
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.train.serialization import save_checkpoint
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

    # Training loop
    G_losses = []
    D_losses = []

    data_loader = ds.create_dict_iterator(output_numpy=True, num_epochs=args.epoch_num)
    print("Starting Training Loop...")
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
            save_checkpoint(netG, os.path.join(args.ckpt_dir, f"Generator_{epoch+1}.ckpt"))
            print("ckpt generated from epoch", epoch + 1, "saved")
