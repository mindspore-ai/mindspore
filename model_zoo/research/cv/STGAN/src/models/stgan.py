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
""" STGAN Models """
import os
import math
import random
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.common.dtype as mstype
from mindspore import Tensor
import numpy as np

from .networks import init_weights, Discriminator, Generator, TrainOneStepGenerator, TrainOneStepDiscriminator
from .losses import GeneratorLoss, DiscriminatorLoss
from .base_model import BaseModel


class STGANModel(BaseModel):
    """ STGANModel """
    def __init__(self, args):
        """ """
        BaseModel.__init__(self, args)
        self.rand_int = ops.UniformInt()
        self.concat = ops.operations.Concat(axis=1)
        self.use_stu = args.use_stu
        self.args = args
        self.n_attrs = len(args.attrs)
        self.mode = args.mode
        self.loss_names = [
            'D', 'G', 'adv_D', 'cls_D', 'real_D', 'fake_D', 'gp_D', 'adv_G',
            'cls_G', 'rec_G'
        ]

        if self.isTrain:
            self.model_names = ['G', 'D']
        else:
            self.model_names = ['G']

        self.netG = Generator(self.n_attrs,
                              args.enc_dim,
                              args.dec_dim,
                              args.enc_layers,
                              args.dec_layers,
                              args.shortcut_layers,
                              args.stu_kernel_size,
                              use_stu=args.use_stu,
                              one_more_conv=args.one_more_conv,
                              stu_norm=args.stu_norm)
        print('Generator: ', self.netG)
        if self.isTrain:
            self.netD = Discriminator(args.image_size, self.n_attrs,
                                      args.dis_dim, args.dis_fc_dim,
                                      args.dis_layers)
            print('Discriminator: ', self.netD)

        if self.args.continue_train:
            continue_iter = self.args.continue_iter if self.args.continue_iter != -1 else 'latest'
            self.load_networks(continue_iter)

        if self.netG is not None:
            num_params = 0
            for p in self.netG.trainable_params():
                num_params += np.prod(p.shape)
            print(
                '\n\n\nGenerator trainable parameters: {}'.format(num_params))
        if self.netD is not None:
            num_params = 0
            for p in self.netD.trainable_params():
                num_params += np.prod(p.shape)
            print('Discriminator trainable parameters: {}\n\n\n'.format(
                num_params))

        if self.isTrain:
            if not self.args.continue_train:
                init_weights(self.netG, 'KaimingUniform', math.sqrt(5))
                init_weights(self.netD, 'KaimingUniform', math.sqrt(5))
            self.loss_D_model = DiscriminatorLoss(self.netD, self.netG,
                                                  self.args, self.mode)
            self.loss_G_model = GeneratorLoss(self.netG, self.netD, self.args,
                                              self.mode)
            self.optimizer_G = nn.Adam(self.netG.trainable_params(),
                                       self.get_learning_rate(),
                                       beta1=args.beta1,
                                       beta2=args.beta2)
            self.optimizer_D = nn.Adam(self.netD.trainable_params(),
                                       self.get_learning_rate(),
                                       beta1=args.beta1,
                                       beta2=args.beta2)
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.train_G = TrainOneStepGenerator(self.loss_G_model,
                                                 self.optimizer_G, self.args)
            self.train_D = TrainOneStepDiscriminator(self.loss_D_model,
                                                     self.optimizer_D)

            self.train_G.set_train()
            self.train_D.set_train()

        if self.args.phase == 'test':
            if self.args.ckpt_path is not None:
                self.load_generator_from_path(self.args.ckpt_path)
            else:
                self.load_networks('latest' if args.test == -1 else args.test)

    def set_input(self, input_data):
        self.label_org = Tensor(input_data['label'], mstype.float32)
        (shape_0, _) = self.label_org.shape
        rand_idx = random.sample(range(0, shape_0), shape_0)
        self.label_trg = self.label_org[rand_idx]
        self.real_x = Tensor(input_data['image'], mstype.float32)

    def test(self, data, filename=''):
        """ Test Function """
        data_image = data['image']
        data_label = data['label']
        c_trg = self.create_test_label(data_label, self.args.attrs)
        attr_diff = c_trg - data_label if self.args.attr_mode == 'diff' else c_trg
        attr_diff = attr_diff * self.args.thres_int
        fake_x = self.netG(data_image, attr_diff)
        self.save_image(
            fake_x[0],
            os.path.join(self.test_result_save_path, '{}'.format(filename)))

    def optimize_parameters(self):
        """ Optimizing Model's Trainable Parameters
        """
        attr_diff = self.label_trg - self.label_org if self.args.attr_mode == 'diff' else self.label_trg
        (h, w) = attr_diff.shape
        rand_attr = Tensor(np.random.rand(h, w), mstype.float32)
        attr_diff = attr_diff * rand_attr * (2 * self.args.thres_int)
        alpha = Tensor(np.random.randn(self.real_x.shape[0], 1, 1, 1),
                       mstype.float32)
        # train D
        _, loss_D, loss_real_D, loss_fake_D, loss_cls_D, loss_gp_D, loss_adv_D, attr_diff =\
            self.train_D(self.real_x, self.label_org, self.label_trg, attr_diff, alpha)
        if self.current_iteration % self.args.n_critic == 0:
            # train G
            _, _, loss_G, loss_fake_G, loss_cls_G, loss_rec_G, loss_adv_G =\
                self.train_G(self.real_x, self.label_org, self.label_trg, attr_diff)
            # saving losses
            if (self.current_iteration / 5) % self.args.print_freq == 0:
                with open(os.path.join(self.train_log_path, 'loss.log'),
                          'a+') as f:
                    f.write('Iter: %s\n' % self.current_iteration)
                    f.write(
                        'loss D: %s, loss D_real: %s, loss D_fake: %s, loss D_gp: %s, loss D_adv: %s, loss D_cls: %s \n'
                        % (loss_D, loss_real_D, loss_fake_D, loss_gp_D,
                           loss_adv_D, loss_cls_D))
                    f.write(
                        'loss G: %s, loss G_rec: %s, loss G_fake: %s, loss G_adv: %s, loss G_cls: %s \n\n'
                        % (loss_G, loss_rec_G, loss_fake_G, loss_adv_G,
                           loss_cls_G))

    def eval(self, data_loader):
        """ Eval function of STGAN
        """
        val_loader = data_loader.val_loader
        concat_3d = ops.Concat(axis=3)
        concat_1d = ops.Concat(axis=1)
        data = next(val_loader)
        data_image = data['image']
        data_label = data['label']
        sample_list = self.create_labels(data_label, self.args.attrs)
        sample_list.insert(0, data_label)
        x_concat = data_image
        for c_trg_sample in sample_list:
            attr_diff = c_trg_sample - data_label if self.args.attr_mode == 'diff' else c_trg_sample
            attr_diff = attr_diff * self.args.thres_int
            fake_x = self.netG(data_image, attr_diff)
            x_concat = concat_3d((x_concat, fake_x))
        sample_result = x_concat[0]
        for i in range(1, x_concat.shape[0]):
            sample_result = concat_1d((sample_result, x_concat[i]))
        self.save_image(
            sample_result,
            os.path.join(self.sample_save_path,
                         'samples_{}.jpg'.format(self.current_iteration + 1)))
