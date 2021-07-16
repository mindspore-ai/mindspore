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
""" Base Model """
import os
import json
from abc import ABC, abstractmethod
import numpy as np
import mindspore.numpy as numpy
import mindspore.common.dtype as mstype

from mindspore import load_checkpoint, load_param_into_net, save_checkpoint
from mindspore.ops import composite as C
from mindspore import Tensor
from PIL import Image

from src.utils import mkdirs


class BaseModel(ABC):
    """ BaseModel """
    def __init__(self, args):
        self.isTrain = args.isTrain
        self.save_dir = os.path.join(args.outputs_dir, args.experiment_name)
        self.model_names = []
        self.loss_names = []
        self.args = args
        self.optimizers = []
        self.current_iteration = 0
        self.netG = None
        self.netD = None

        # continue train
        if self.isTrain:
            if self.args.continue_train:
                assert (os.path.exists(self.save_dir)
                        ), 'Checkpoint path not found at %s' % self.save_dir
                self.current_iteration = self.args.continue_iter
            else:
                if not os.path.exists(self.save_dir):
                    mkdirs(self.save_dir)

        # save config
        self.config_save_path = os.path.join(self.save_dir, 'config')
        if not os.path.exists(self.config_save_path):
            mkdirs(self.config_save_path)
        if self.isTrain:
            with open(os.path.join(self.config_save_path, 'train.conf'),
                      'w') as f:
                f.write(json.dumps(vars(self.args)))
            if self.current_iteration == -1:
                with open(os.path.join(self.config_save_path, 'latest.conf'),
                          'r') as f:
                    self.current_iteration = int(f.read())

        # sample save path
        if self.isTrain:
            self.sample_save_path = os.path.join(self.save_dir, 'sample')
            if not os.path.exists(self.sample_save_path):
                mkdirs(self.sample_save_path)

        # test result save path
        if self.args.phase == 'test':
            self.test_result_save_path = os.path.join(self.save_dir, 'test')
            if not os.path.exists(self.test_result_save_path):
                mkdirs(self.test_result_save_path)

        # train log save path
        if self.isTrain:
            self.train_log_path = os.path.join(self.save_dir, 'logs')
            if not os.path.exists(self.train_log_path):
                mkdirs(self.train_log_path)

    @abstractmethod
    def set_input(self, input_data):
        pass

    @abstractmethod
    def optimize_parameters(self):
        pass

    @abstractmethod
    def test(self):
        pass

    @abstractmethod
    def eval(self):
        pass

    def load_config(self):
        print('loading config from {}\n\n'.format(
            os.path.join(self.config_save_path, 'train.conf')))
        with open(os.path.join(self.config_save_path, 'train.conf'), 'r') as f:
            config = json.loads(f.read())
            print('config: ', config)
            return config

    def save_networks(self):
        """ saving networks """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_%s.ckpt' % (self.current_iteration, name)
                save_filename_latest = 'latest_%s.ckpt' % name
                save_path = os.path.join(self.save_dir, 'ckpt')
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                save_path_latest = os.path.join(save_path,
                                                save_filename_latest)
                save_path = os.path.join(save_path, save_filename)
                net = getattr(self, 'net' + name)

                print('saving the model to %s' % save_path)
                print('saving the model to %s' % save_path_latest)
                save_checkpoint(net, save_path)
                save_checkpoint(net, save_path_latest)

        with open(os.path.join(self.config_save_path, 'latest.conf'),
                  'w') as f:
            f.write("{}".format(self.current_iteration))

    def load_networks(self, epoch='latest'):
        """ Load model checkpoint file

            Parameters:
                epoch: epoch saved, default is latest
        """
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_%s.ckpt' % (epoch, name)
                load_path = os.path.join(self.save_dir, 'ckpt', load_filename)
                net = getattr(self, 'net' + name)

                print('loading the model from %s' % load_path)
                if name == 'G':
                    net.encoder.update_parameters_name(
                        'network.network.netG.encoder.')
                    net.decoder.update_parameters_name(
                        'network.network.netG.decoder.')
                    net.stu.update_parameters_name('network.network.netG.stu.')
                params = load_checkpoint(load_path, net, strict_load=True)
                load_param_into_net(net, params, strict_load=True)
        if epoch == 'latest':
            assert os.path.exists(
                os.path.join(self.config_save_path, 'latest.conf')
            ), 'Missing iteration information of latest checkpoint file.'
            with open(os.path.join(self.config_save_path, 'latest.conf'),
                      'r') as f:
                self.current_iteration = f.read()

    def load_generator_from_path(self, path=None):
        """ Load generator checkpoint file from given path

            Parameters:
                path: path of checkpoint file, required
        """
        assert path is not None, 'Path of checkpoint can not be None'
        print('loading the model from %s' % path)
        net = getattr(self, 'netG')
        net.encoder.update_parameters_name('network.network.netG.encoder.')
        net.decoder.update_parameters_name('network.network.netG.decoder.')
        net.stu.update_parameters_name('network.network.netG.stu.')
        params = load_checkpoint(path, net, strict_load=True)
        load_param_into_net(net, params, strict_load=True)

    def get_learning_rate(self):
        """Learning rate generator."""
        lrs = [self.args.lr] * self.args.dataset_size * self.args.init_epoch
        lrs += [self.args.lr * 0.1] * self.args.dataset_size * (
            self.args.n_epochs - self.args.init_epoch)
        return Tensor(np.array(lrs).astype(
            np.float32))[self.current_iteration:]

    def save_image(self, img, img_path):
        """Save a numpy image to the disk

        Parameters:
            img (numpy array / Tensor): image to save.
            image_path (str): the path of the image.
        """
        if isinstance(img, Tensor):
            img = self.decode_image(img)
        elif not isinstance(img, np.ndarray):
            raise ValueError(
                "img should be Tensor or numpy array, but get {}".format(
                    type(img)))
        img_pil = Image.fromarray(img)
        img_pil.save(img_path)

    def decode_image(self, img):
        """Decode a [1, C, H, W] Tensor to image numpy array."""
        mean = 0.5 * 255
        std = 0.5 * 255
        return (img.asnumpy() * std + mean).astype(np.uint8).transpose(
            (1, 2, 0))

    def create_labels(self, c_org, selected_attrs=None):
        """Generate target domain labels for debugging and testing."""
        hair_color_indices = []
        for i, attr_name in enumerate(selected_attrs):
            if attr_name in [
                    'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair'
            ]:
                hair_color_indices.append(i)

        c_trg_list = []
        for i in range(len(selected_attrs)):
            c_trg = numpy.copy(c_org)
            if i in hair_color_indices:
                c_trg[:, i] = Tensor(1, mstype.float32)
                for j in hair_color_indices:
                    if j != i:
                        c_trg[:, j] = Tensor(0, mstype.float32)
            else:
                c_trg[:, i] = Tensor((c_trg[:, i] == 0).asnumpy(),
                                     mstype.float32)

            c_trg_list.append(c_trg)
        return c_trg_list

    def create_test_label(self, c_org, selected_attrs=None):
        """Generate target domain labels for debugging and testing."""
        hair_color_indices = []
        for i, attr_name in enumerate(selected_attrs):
            if attr_name in [
                    'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair'
            ]:
                hair_color_indices.append(i)

        c_trg = numpy.copy(c_org)
        for i in range(len(selected_attrs)):
            if i in hair_color_indices:
                c_trg[:, i] = 1.
                for j in hair_color_indices:
                    if j != i:
                        c_trg[:, j] = Tensor(0, mstype.float32)
            else:
                c_trg[:, i] = Tensor((c_trg[:, i] == 0).asnumpy(),
                                     mstype.float32)

        return c_trg

    def denorm(self, x):
        """ Denormalization """
        out = (x + 1) / 2
        return C.clip_by_value(out, 0, 1)
