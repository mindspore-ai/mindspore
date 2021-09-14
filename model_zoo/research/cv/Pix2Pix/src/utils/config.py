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

"""
    Define the common options that are used in both training and test.
"""

import argparse
import ast


def get_args():
    '''
        get args.
    '''
    parser = argparse.ArgumentParser(description='Pix2Pix Model')

    # parameters
    parser.add_argument('--device_target', type=str, default='Ascend', choices=('Ascend', 'GPU'),
                        help='device where the code will be implemented (default: Ascend)')
    parser.add_argument('--run_distribute', type=int, default=0, help='distributed training, default is 0.')
    parser.add_argument('--device_num', type=int, default=1, help='device num, default is 1.')
    parser.add_argument('--device_id', type=int, default=6, help='device id, default is 0.')
    parser.add_argument('--save_graphs', type=ast.literal_eval, default=False,
                        help='whether save graphs, default is False.')
    parser.add_argument('--init_type', type=str, default='normal', help='network initialization, default is normal.')
    parser.add_argument('--init_gain', type=float, default=0.02,
                        help='scaling factor for normal, xavier and orthogonal, default is 0.02.')
    parser.add_argument('--pad_mode', type=str, default='CONSTANT', choices=('CONSTANT', 'REFLECT', 'SYMMETRIC'),
                        help='scale images to this size, default is CONSTANT.')
    parser.add_argument('--load_size', type=int, default=286, help='scale images to this size, default is 286.')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size, default is 1.')
    parser.add_argument('--LAMBDA_Dis', type=float, default=0.5, help='weight for Discriminator Loss, default is 0.5.')
    parser.add_argument('--LAMBDA_GAN', type=int, default=1, help='weight for GAN Loss, default is 1.')
    parser.add_argument('--LAMBDA_L1', type=int, default=100, help='weight for L1 Loss, default is 100.')
    parser.add_argument('--beta1', type=float, default=0.5, help='adam beta1, default is 0.5.')
    parser.add_argument('--beta2', type=float, default=0.999, help='adam beta2, default is 0.999.')
    parser.add_argument('--lr', type=float, default=0.0002, help='the initial learning rate, default is 0.0002.')
    parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy, default is linear.')
    parser.add_argument('--epoch_num', type=int, default=200, help='epoch number for training, default is 200.')
    parser.add_argument('--n_epochs', type=int, default=100,
                        help='number of epochs with the initial learning rate, default is 100.')
    parser.add_argument('--n_epochs_decay', type=int, default=100,
                        help='number of epochs with the dynamic learning rate, default is 100.')
    parser.add_argument('--dataset_size', type=int, default=400, choices=(400, 1096),
                        help='for Facade_dataset,the number is 400; for Maps_dataset,the number is 1096.')

    # The location of input and output data
    parser.add_argument('--train_data_dir', type=str, default=None, help='the file path of input data during training.')
    parser.add_argument('--val_data_dir', type=str, default=None, help='the file path of input data during validating.')
    parser.add_argument('--train_fakeimg_dir', type=str, default='./results/fake_img/',
                        help='during training, the file path of stored fake img.')
    parser.add_argument('--loss_show_dir', type=str, default='./results/loss_show',
                        help='during training, the file path of stored loss img.')
    parser.add_argument('--ckpt_dir', type=str, default='./results/ckpt/',
                        help='during training, the file path of stored CKPT.')
    parser.add_argument('--ckpt', type=str, default=None, help='during validating, the file path of the CKPT used.')
    parser.add_argument('--predict_dir', type=str, default='./results/predict/',
                        help='during validating, the file path of Generated image.')
    args = parser.parse_args()
    return args
