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
"""Entry point for training AttGAN network"""

import argparse
import datetime
import json
import math
import os
from os.path import join

import numpy as np

import mindspore.common.dtype as mstype
from mindspore import Tensor, context
from mindspore import nn
from mindspore.common import set_seed
from mindspore.communication.management import init, get_rank
from mindspore.context import ParallelMode
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, _InternalCallbackParam, RunContext
from mindspore.train.serialization import load_param_into_net

from src.attgan import Gen, Dis
from src.cell import TrainOneStepCellGen, TrainOneStepCellDis, init_weights
from src.data import data_loader
from src.helpers import Progressbar
from src.loss import GenLoss, DisLoss
from src.utils import resume_generator, resume_discriminator

attrs_default = [
    'Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows',
    'Eyeglasses', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'No_Beard', 'Pale_Skin', 'Young'
]


def parse(arg=None):
    """Define configuration of Model"""
    parser = argparse.ArgumentParser()

    parser.add_argument('--attrs', dest='attrs', default=attrs_default, nargs='+', help='attributes to learn')
    parser.add_argument('--data', dest='data', type=str, choices=['CelebA'], default='CelebA')
    parser.add_argument('--data_path', dest='data_path', type=str, default='./data/img_align_celeba')
    parser.add_argument('--attr_path', dest='attr_path', type=str, default='./data/list_attr_celeba.txt')

    parser.add_argument('--img_size', dest='img_size', type=int, default=128)
    parser.add_argument('--shortcut_layers', dest='shortcut_layers', type=int, default=1)
    parser.add_argument('--inject_layers', dest='inject_layers', type=int, default=1)

    parser.add_argument('--enc_dim', dest='enc_dim', type=int, default=64)
    parser.add_argument('--dec_dim', dest='dec_dim', type=int, default=64)
    parser.add_argument('--dis_dim', dest='dis_dim', type=int, default=64)
    parser.add_argument('--dis_fc_dim', dest='dis_fc_dim', type=int, default=1024)
    parser.add_argument('--enc_layers', dest='enc_layers', type=int, default=5)
    parser.add_argument('--dec_layers', dest='dec_layers', type=int, default=5)
    parser.add_argument('--dis_layers', dest='dis_layers', type=int, default=5)
    parser.add_argument('--enc_norm', dest='enc_norm', type=str, default='batchnorm')
    parser.add_argument('--dec_norm', dest='dec_norm', type=str, default='batchnorm')
    parser.add_argument('--dis_norm', dest='dis_norm', type=str, default='instancenorm')
    parser.add_argument('--dis_fc_norm', dest='dis_fc_norm', type=str, default='none')
    parser.add_argument('--enc_acti', dest='enc_acti', type=str, default='lrelu')
    parser.add_argument('--dec_acti', dest='dec_acti', type=str, default='relu')
    parser.add_argument('--dis_acti', dest='dis_acti', type=str, default='lrelu')
    parser.add_argument('--dis_fc_acti', dest='dis_fc_acti', type=str, default='relu')
    parser.add_argument('--lambda_1', dest='lambda_1', type=float, default=100.0)
    parser.add_argument('--lambda_2', dest='lambda_2', type=float, default=10.0)
    parser.add_argument('--lambda_3', dest='lambda_3', type=float, default=1.0)
    parser.add_argument('--lambda_gp', dest='lambda_gp', type=float, default=10.0)

    parser.add_argument('--epochs', dest='epochs', type=int, default=200, help='# of epochs')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32)
    parser.add_argument('--num_workers', dest='num_workers', type=int, default=16)
    parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--beta1', dest='beta1', type=float, default=0.5)
    parser.add_argument('--beta2', dest='beta2', type=float, default=0.999)
    parser.add_argument('--n_d', dest='n_d', type=int, default=5, help='# of d updates per g update')
    parser.add_argument('--split_point', dest='split_point', type=int, default=182000, help='# of dataset split point')

    parser.add_argument('--thres_int', dest='thres_int', type=float, default=0.5)
    parser.add_argument('--test_int', dest='test_int', type=float, default=1.0)
    parser.add_argument('--save_interval', dest='save_interval', type=int, default=500)
    parser.add_argument('--experiment_name', dest='experiment_name',
                        default=datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y"))
    parser.add_argument("--run_distribute", type=int, default=0, help="Run distribute, default: false.")
    parser.add_argument('--resume_model', action='store_true')
    parser.add_argument('--gen_ckpt_name', type=str, default='')
    parser.add_argument('--dis_ckpt_name', type=str, default='')

    return parser.parse_args(arg)


args = parse()
print(args)

args.lr_base = args.lr
args.n_attrs = len(args.attrs)

# initialize environment
set_seed(1)
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=False)

if args.run_distribute:
    if os.getenv("DEVICE_ID", "not_set").isdigit():
        context.set_context(device_id=int(os.getenv("DEVICE_ID")))
    device_num = int(os.getenv('RANK_SIZE'))
    print(device_num)
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                      device_num=device_num)
    init()
    rank = get_rank()
else:
    if os.getenv("DEVICE_ID", "not_set").isdigit():
        context.set_context(device_id=int(os.getenv("DEVICE_ID")))
    device_num = int(os.getenv('RANK_SIZE'))
    rank = 0

print("Initialize successful!")

os.makedirs(join('output', args.experiment_name), exist_ok=True)
os.makedirs(join('output', args.experiment_name, 'checkpoint'), exist_ok=True)
with open(join('output', args.experiment_name, 'setting.txt'), 'w') as f:
    f.write(json.dumps(vars(args), indent=4, separators=(',', ':')))

if __name__ == '__main__':

    # Define dataloader
    train_dataset, train_length = data_loader(img_path=args.data_path,
                                              attr_path=args.attr_path,
                                              selected_attrs=args.attrs,
                                              mode="train",
                                              batch_size=args.batch_size,
                                              device_num=device_num,
                                              shuffle=True,
                                              split_point=args.split_point)
    train_loader = train_dataset.create_dict_iterator()

    print('Training images:', train_length)

    # Define network
    gen = Gen(args.enc_dim, args.enc_layers, args.enc_norm, args.enc_acti, args.dec_dim, args.dec_layers, args.dec_norm,
              args.dec_acti, args.n_attrs, args.shortcut_layers, args.inject_layers, args.img_size, mode='train')
    dis = Dis(args.dis_dim, args.dis_norm, args.dis_acti, args.dis_fc_dim, args.dis_fc_norm, args.dis_fc_acti,
              args.dis_layers, args.img_size, mode='train')

    # Initialize network
    init_weights(gen, 'KaimingUniform', math.sqrt(5))
    init_weights(dis, 'KaimingUniform', math.sqrt(5))

    # Resume from checkpoint
    if args.resume_model:
        para_gen = resume_generator(args, gen, args.gen_ckpt_name)
        para_dis = resume_discriminator(args, dis, args.dis_ckpt_name)
        load_param_into_net(gen, para_gen)
        load_param_into_net(dis, para_dis)

    # Define network with loss
    G_loss_cell = GenLoss(args, gen, dis)
    D_loss_cell = DisLoss(args, gen, dis)

    # Define Optimizer
    optimizer_G = nn.Adam(params=gen.trainable_params(), learning_rate=args.lr, beta1=args.beta1, beta2=args.beta2)
    optimizer_D = nn.Adam(params=dis.trainable_params(), learning_rate=args.lr, beta1=args.beta1, beta2=args.beta2)

    # Define One Step Train
    G_trainOneStep = TrainOneStepCellGen(G_loss_cell, optimizer_G)
    D_trainOneStep = TrainOneStepCellDis(D_loss_cell, optimizer_D)

    # Train
    G_trainOneStep.set_train(True)
    D_trainOneStep.set_train(True)

    print("Start Training")

    train_iter = train_length // args.batch_size
    ckpt_config = CheckpointConfig(save_checkpoint_steps=args.save_interval)

    if rank == 0:
        local_train_url = os.path.join('output', args.experiment_name, 'checkpoint/rank{}'.format(rank))
        ckpt_cb_gen = ModelCheckpoint(config=ckpt_config, directory=local_train_url, prefix='generator')
        ckpt_cb_dis = ModelCheckpoint(config=ckpt_config, directory=local_train_url, prefix='discriminator')

        cb_params_gen = _InternalCallbackParam()
        cb_params_gen.train_network = gen
        cb_params_gen.cur_epoch_num = 0
        gen_run_context = RunContext(cb_params_gen)
        ckpt_cb_gen.begin(gen_run_context)

        cb_params_dis = _InternalCallbackParam()
        cb_params_dis.train_network = dis
        cb_params_dis.cur_epoch_num = 0
        dis_run_context = RunContext(cb_params_dis)
        ckpt_cb_dis.begin(dis_run_context)

    # Initialize Progressbar
    progressbar = Progressbar()

    it = 0
    for epoch in range(args.epochs):

        for data in progressbar(train_loader, train_iter):
            img_a = data["image"]
            att_a = data["attr"]
            att_a = att_a.asnumpy()
            att_b = np.random.permutation(att_a)

            att_a_ = (att_a * 2 - 1) * args.thres_int
            att_b_ = (att_b * 2 - 1) * args.thres_int

            att_a = Tensor(att_a, mstype.float32)
            att_a_ = Tensor(att_a_, mstype.float32)
            att_b = Tensor(att_b, mstype.float32)
            att_b_ = Tensor(att_b_, mstype.float32)

            if (it + 1) % (args.n_d + 1) != 0:
                d_out, d_real_loss, d_fake_loss, dc_loss, df_gp = D_trainOneStep(img_a, att_a, att_a_, att_b, att_b_)
            else:
                g_out, gf_loss, gc_loss, gr_loss = G_trainOneStep(img_a, att_a, att_a_, att_b, att_b_)
                progressbar.say(epoch=epoch, iter=it + 1, d_loss=d_out, g_loss=g_out, gf_loss=gf_loss, gc_loss=gc_loss,
                                gr_loss=gr_loss, dc_loss=dc_loss, df_gp=df_gp)

            if (epoch + 1) % 5 == 0 and (it + 1) % args.save_interval == 0 and rank == 0:
                cb_params_gen.cur_epoch_num = epoch + 1
                cb_params_dis.cur_epoch_num = epoch + 1
                cb_params_gen.cur_step_num = it + 1
                cb_params_dis.cur_step_num = it + 1
                cb_params_gen.batch_num = it + 2
                cb_params_dis.batch_num = it + 2
                ckpt_cb_gen.step_end(gen_run_context)
                ckpt_cb_dis.step_end(dis_run_context)
            it += 1
