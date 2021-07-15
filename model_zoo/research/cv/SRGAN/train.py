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

"""train scripts"""

import os
import argparse
import time
import numpy as np
from skimage.color import rgb2ycbcr
from skimage.metrics import peak_signal_noise_ratio
import mindspore.nn as nn
from mindspore.communication.management import init, get_rank
from mindspore import context
from mindspore import load_checkpoint, save_checkpoint, load_param_into_net
from mindspore.context import ParallelMode
import mindspore.ops as ops
from src.model.generator import get_generator
from src.model.discriminator import get_discriminator
from src.dataset.traindataset import create_traindataset
from src.dataset.testdataset import create_testdataset
from src.loss.psnr_loss import PSNRLoss
from src.loss.gan_loss import DiscriminatorLoss, GeneratorLoss
from src.trainonestep.train_psnr import TrainOnestepPSNR
from src.trainonestep.train_gan import TrainOneStepD
from src.trainonestep.train_gan import TrainOnestepG


parser = argparse.ArgumentParser(description="SRGAN train")
parser.add_argument("--train_LR_path", type=str, default='/data/DIV2K/LR')
parser.add_argument("--train_GT_path", type=str, default='/data/DIV2K/HR')
parser.add_argument("--val_LR_path", type=str, default='/data/Set5/LR')
parser.add_argument("--val_GT_path", type=str, default='/data/Set5/HR')
parser.add_argument("--vgg_ckpt", type=str, default='/data/pre-models/vgg19/vgg19.ckpt')
parser.add_argument('--upscale_factor', default=4, type=int, choices=[2, 4, 8],
                    help='super resolution upscale factor')
parser.add_argument("--image_size", type=int, default=96,
                    help="Image size of high resolution image. (default: 96)")
parser.add_argument("--train_batch_size", default=16, type=int,
                    metavar="N",
                    help="batch size for training")
parser.add_argument("--val_batch_size", default=1, type=int,
                    metavar="N",
                    help="batch size for tesing")
parser.add_argument("--psnr_epochs", default=2000, type=int, metavar="N",
                    help="Number of total psnr epochs to run. (default: 2000)")
parser.add_argument("--start_psnr_epoch", default=0, type=int, metavar='N',
                    help="Manual psnr epoch number (useful on restarts). (default: 0)")
parser.add_argument("--gan_epochs", default=1000, type=int, metavar="N",
                    help="Number of total gan epochs to run. (default: 1000)")
parser.add_argument("--start_gan_epoch", default=0, type=int, metavar='N',
                    help="Manual gan epoch number (useful on restarts). (default: 0)")
parser.add_argument('--init_type', type=str, default='normal', choices=("normal", "xavier"), \
                    help='network initialization, default is normal.')
parser.add_argument("--scale", type=int, default=4)
# distribute
parser.add_argument("--run_distribute", type=int, default=0, help="Run distribute, default: false.")
parser.add_argument("--device_id", type=int, default=0, help="device id, default: 0.")
parser.add_argument("--device_num", type=int, default=1, help="number of device, default: 0.")
parser.add_argument("--rank_id", type=int, default=0, help="Rank id, default: 0.")

if __name__ == '__main__':
    args = parser.parse_args()
    context.set_context(mode=context.GRAPH_MODE, device_id=args.device_id, save_graphs=False)
    #distribute
    if args.run_distribute:
        print("distribute")
        context.set_context(device_id=int(os.getenv("DEVICE_ID")))
        device_num = args.device_num
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                          device_num=device_num)
        init()

        rank = get_rank()
    # for srresnet
    # create dataset
    train_ds = create_traindataset(args.train_batch_size, args.train_LR_path, args.train_GT_path)
    test_ds = create_testdataset(args.val_batch_size, args.val_LR_path, args.val_GT_path)
    train_data_loader = train_ds.create_dict_iterator()
    test_data_loader = test_ds.create_dict_iterator()
    # definition of network
    generator = get_generator(4, 0.02)

    # network with loss

    psnr_loss = PSNRLoss(generator)

    # optimizer
    psnr_optimizer = nn.Adam(generator.trainable_params(), 1e-4)

    # operation for testing
    op = ops.ReduceSum(keep_dims=False)
    # trainonestep

    train_psnr = TrainOnestepPSNR(psnr_loss, psnr_optimizer)
    train_psnr.set_train()

    bestpsnr = 0
    if not os.path.exists("./ckpt"):
        os.makedirs("./ckpt")
    print('start training:')

    print('start training PSNR:')
    # warm up generator
    for epoch in range(args.start_psnr_epoch, args.psnr_epochs):
        print("training {:d} epoch:".format(epoch+1))
        mysince = time.time()
        for data in train_data_loader:
            lr = data['LR']
            hr = data['HR']
            mse_loss = train_psnr(hr, lr)
        steps = train_ds.get_dataset_size()
        time_elapsed = (time.time()-mysince)
        step_time = time_elapsed / steps
        print('per step needs time:{:.0f}ms'.format(step_time * 1000))
        print("mse_loss:")
        print(mse_loss)
        psnr_list = []
        #val for every epoch
        print("start valing:")
        for test in test_data_loader:
            lr = test['LR']
            gt = test['HR']

            bs, c, h, w = lr.shape[:4]
            gt = gt[:, :, : h * args.scale, : w *args.scale]

            output = generator(lr)
            output = op(output, 0)
            output = output.asnumpy()
            output = np.clip(output, -1.0, 1.0)
            gt = op(gt, 0)

            output = (output + 1.0) / 2.0
            gt = (gt + 1.0) / 2.0

            output = output.transpose(1, 2, 0)
            gt = gt.asnumpy()
            gt = gt.transpose(1, 2, 0)

            y_output = rgb2ycbcr(output)[args.scale:-args.scale, args.scale:-args.scale, :1]
            y_gt = rgb2ycbcr(gt)[args.scale:-args.scale, args.scale:-args.scale, :1]
            psnr = peak_signal_noise_ratio(y_output / 255.0, y_gt / 255.0, data_range=1.0)
            psnr_list.append(psnr)

        mean = np.mean(psnr_list)
        print("psnr:")
        print(mean)
        if mean > bestpsnr:
            print("saving ckpt")
            bestpsnr = mean
            if args.run_distribute == 0:
                save_checkpoint(train_psnr, "./ckpt/best.ckpt")
            else:
                if args.device_id == 0:
                    save_checkpoint(train_psnr, "./ckpt/best.ckpt")

        if (epoch+1)%200 == 0:
            if args.run_distribute == 0:
                save_checkpoint(train_psnr, './ckpt/pre_trained_model_%03d.ckpt'%(epoch+1))
            else:
                if args.device_id == 0:
                    save_checkpoint(train_psnr, './ckpt/pre_trained_model_%03d.ckpt'%(epoch+1))

        print("{:d}/2000 epoch finished".format(epoch+1))
    # for srgan
    generator = get_generator(4, 0.02)
    discriminator = get_discriminator(96, 0.02)
    if args.run_distribute == 0:
        ckpt = "./ckpt/best.ckpt"
    else:
        ckpt = '../train_parallel0/ckpt/best.ckpt'
    params = load_checkpoint(ckpt)
    load_param_into_net(generator, params)
    discriminator_loss = DiscriminatorLoss(discriminator, generator)
    generator_loss = GeneratorLoss(discriminator, generator, args.vgg_ckpt)
    generator_optimizer = nn.Adam(generator.trainable_params(), 1e-4)
    discriminator_optimizer = nn.Adam(discriminator.trainable_params(), 1e-4)
    train_discriminator = TrainOneStepD(discriminator_loss, discriminator_optimizer)
    train_generator = TrainOnestepG(generator_loss, generator_optimizer)
    print("========================================")
    print('start training GAN :')
    # trainGAN
    for epoch in range(args.start_gan_epoch, args.gan_epochs):
        print('training {:d} epoch'.format(epoch+1))
        mysince1 = time.time()
        for data in train_data_loader:
            lr = data['LR']
            hr = data['HR']
            D_loss = train_discriminator(hr, lr)
            G_loss = train_generator(hr, lr)
        time_elapsed1 = (time.time()-mysince1)
        steps = train_ds.get_dataset_size()
        step_time1 = time_elapsed1 / steps
        print('per step needs time:{:.0f}ms'.format(step_time1 * 1000))
        print("D_loss:")
        print(D_loss.mean())
        print("G_loss:")
        print(G_loss.mean())

        if (epoch+1)%100 == 0:
            print("saving ckpt")
            if args.run_distribute == 0:
                save_checkpoint(train_generator, './ckpt/G_model_%03d.ckpt'%(epoch+1))
                save_checkpoint(train_discriminator, './ckpt/D_model_%03d.ckpt'%(epoch+1))
            else:
                if args.device_id == 0:
                    save_checkpoint(train_generator, './ckpt/G_model_%03d.ckpt'%(epoch+1))
                    save_checkpoint(train_discriminator, './ckpt/D_model_%03d.ckpt'%(epoch+1))
        print(" {:d}/1000 epoch finished".format(epoch+1))
