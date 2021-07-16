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
"""arguments"""
import os
import argparse
import ast
import datetime
from mindspore.context import ParallelMode
from mindspore import context
from mindspore.communication.management import init

def add_basic_parameters(parser):
    """ add basic parameters """
    parser.add_argument("--platform",
                        type=str,
                        default="Ascend",
                        choices=("Ascend", "GPU", "CPU"),
                        help="running platform, support Ascend, GPU and CPU")
    parser.add_argument("--device_id",
                        type=int,
                        default=0,
                        help="device id, default is 0")
    parser.add_argument('--device_num',
                        type=int,
                        default=1,
                        help='device num, default is 1.')
    parser.add_argument('--ms_version',
                        type=str,
                        default='1.2.0',
                        help="Mindspore's Version, default is 1.2.0")
    return parser

def add_model_parameters(parser):
    """ add model parameters """
    att_dict = {
        '5_o_Clock_Shadow': 0,
        'Arched_Eyebrows': 1,
        'Attractive': 2,
        'Bags_Under_Eyes': 3,
        'Bald': 4,
        'Bangs': 5,
        'Big_Lips': 6,
        'Big_Nose': 7,
        'Black_Hair': 8,
        'Blond_Hair': 9,
        'Blurry': 10,
        'Brown_Hair': 11,
        'Bushy_Eyebrows': 12,
        'Chubby': 13,
        'Double_Chin': 14,
        'Eyeglasses': 15,
        'Goatee': 16,
        'Gray_Hair': 17,
        'Heavy_Makeup': 18,
        'High_Cheekbones': 19,
        'Male': 20,
        'Mouth_Slightly_Open': 21,
        'Mustache': 22,
        'Narrow_Eyes': 23,
        'No_Beard': 24,
        'Oval_Face': 25,
        'Pale_Skin': 26,
        'Pointy_Nose': 27,
        'Receding_Hairline': 28,
        'Rosy_Cheeks': 29,
        'Sideburns': 30,
        'Smiling': 31,
        'Straight_Hair': 32,
        'Wavy_Hair': 33,
        'Wearing_Earrings': 34,
        'Wearing_Hat': 35,
        'Wearing_Lipstick': 36,
        'Wearing_Necklace': 37,
        'Wearing_Necktie': 38,
        'Young': 39
    }
    attr_default = ['Bangs', 'Blond_Hair', 'Mustache', 'Young']
    parser.add_argument("--attrs",
                        default=attr_default,
                        choices=att_dict,
                        nargs='+',
                        help='Attributes to modify by the model')
    parser.add_argument('--image_size',
                        type=int,
                        default=128,
                        help='input image size')
    parser.add_argument(
        '--shortcut_layers',
        type=int,
        default=3,
        help='# of skip connections between the encoder and the decoder')
    parser.add_argument('--enc_dim', type=int, default=64)
    parser.add_argument('--dec_dim', type=int, default=64)
    parser.add_argument('--dis_dim', type=int, default=64)
    parser.add_argument('--dis_fc_dim',
                        type=int,
                        default=1024,
                        help='# of discriminator fc channels')
    parser.add_argument('--enc_layers', type=int, default=5)
    parser.add_argument('--dec_layers', type=int, default=5)
    parser.add_argument('--dis_layers', type=int, default=5)
    # STGAN & STU
    parser.add_argument('--attr_mode',
                        type=str,
                        default='diff',
                        choices=['diff', 'target'])
    parser.add_argument('--use_stu', type=bool, default=True)
    parser.add_argument('--stu_dim', type=int, default=64)
    parser.add_argument('--stu_kernel_size', type=int, default=3)
    parser.add_argument('--stu_norm',
                        type=str,
                        default='bn',
                        choices=['bn', 'in'])
    parser.add_argument(
        '--stu_state',
        type=str,
        default='stu',
        choices=['stu', 'gru', 'direct'],
        help=
        'gru: gru arch.; stu: stu arch.; direct: directly pass the inner state to the outer layer'
    )
    parser.add_argument(
        '--multi_inputs',
        type=int,
        default=1,
        help='# of hierarchical inputs (in the first several encoder layers')
    parser.add_argument(
        '--one_more_conv',
        type=int,
        default=1,
        choices=[0, 1, 3],
        help='0: no further conv after the decoder; 1: conv(k=1); 3: conv(k=3)'
    )
    return parser

def add_train_parameters(parser):
    """ add train parameters """
    parser.add_argument('--mode',
                        default='wgan',
                        choices=['wgan', 'lsgan', 'dcgan'])
    parser.add_argument('--continue_train',
                        type=bool,
                        default=False,
                        help='Flag of continue train, default is false')
    parser.add_argument(
        '--continue_iter',
        type=int,
        default=-1,
        help='Continue point of continue training, -1 means latest')
    parser.add_argument('--test_iter',
                        type=int,
                        default=-1,
                        help='Checkpoint of model testing, -1 means latest')
    parser.add_argument('--n_epochs',
                        type=int,
                        default=100,
                        help='# of epochs')
    parser.add_argument('--n_critic',
                        type=int,
                        default=5,
                        help='number of D updates per each G update')
    parser.add_argument('--max_epoch',
                        type=int,
                        default=100,
                        help='# of epochs')
    parser.add_argument('--init_epoch',
                        type=int,
                        default=50,
                        help='# of epochs with init lr.')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument("--beta1",
                        type=float,
                        default=0.5,
                        help="Adam beta1, default is 0.5.")
    parser.add_argument("--beta2",
                        type=float,
                        default=0.999,
                        help="Adam beta2, default is 0.999.")
    parser.add_argument("--lambda_gp",
                        type=int,
                        default=10,
                        help="Lambda gp, default is 10")
    parser.add_argument("--lambda1",
                        type=int,
                        default=1,
                        help="Lambda1, default is 1")
    parser.add_argument("--lambda2",
                        type=int,
                        default=10,
                        help="Lambda2, default is 10")
    parser.add_argument("--lambda3",
                        type=int,
                        default=100,
                        help="Lambda3, default is 100")
    parser.add_argument('--lr',
                        type=float,
                        default=0.0002,
                        help='learning rate')
    parser.add_argument('--thres_int', type=float, default=0.5)
    parser.add_argument('--test_int', type=float, default=1.0)
    parser.add_argument('--n_sample',
                        type=int,
                        default=64,
                        help='# of sample images')
    parser.add_argument('--print_freq',
                        type=int,
                        default=1,
                        help='print log freq (per critic), default is 1')
    parser.add_argument(
        '--save_freq',
        type=int,
        default=5000,
        help='save model evary save_freq iters, 0 means to save evary epoch.')
    parser.add_argument(
        '--sample_freq',
        type=int,
        default=1000,
        help=
        'eval on validation set every sample_freq iters, 0 means to save evary epoch.'
    )
    return parser

def get_args(phase):
    """get args"""
    parser = argparse.ArgumentParser(description="STGAN")
    # basic parameters
    parser = add_basic_parameters(parser)

    #model parameters
    parser = add_model_parameters(parser)

    # training
    parser = add_train_parameters(parser)

    # others
    parser.add_argument('--use_cropped_img', action='store_true')
    default_experiment_name = datetime.datetime.now().strftime(
        "%Y.%m.%d-%H%M%S")
    parser.add_argument('--experiment_name', default=default_experiment_name)
    parser.add_argument('--num_ckpt', type=int, default=1)
    parser.add_argument('--clear', default=False, action='store_true')
    parser.add_argument('--save_graphs', type=ast.literal_eval, default=False, \
                        help='whether save graphs, default is False.')
    parser.add_argument('--outputs_dir', type=str, default='./outputs', \
                        help='models are saved here, default is ./outputs.')
    parser.add_argument("--dataroot", type=str, default='./dataset')
    parser.add_argument('--file_format', type=str, choices=['AIR', 'ONNX', 'MINDIR'], default='AIR', \
                    help='file format')
    parser.add_argument('--file_name', type=str, default='STGAN', help='output file name prefix.')
    parser.add_argument('--ckpt_path', default=None, help='path of checkpoint file.')

    args = parser.parse_args()
    if phase == 'test':
        assert args.experiment_name != default_experiment_name, "--experiment_name should be assigned in test mode"
    if args.continue_train:
        assert args.experiment_name != default_experiment_name, "--experiment_name should be assigned in continue"
    if args.device_num > 1 and args.platform != "CPU":
        context.set_context(mode=context.GRAPH_MODE,
                            device_target=args.platform,
                            save_graphs=args.save_graphs,
                            device_id=int(os.environ["DEVICE_ID"]))
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(
            parallel_mode=ParallelMode.AUTO_PARALLEL,
            gradients_mean=True,
            device_num=args.device_num)
        init()
        args.rank = int(os.environ["DEVICE_ID"])
    else:
        context.set_context(mode=context.GRAPH_MODE,
                            device_target=args.platform,
                            save_graphs=args.save_graphs,
                            device_id=args.device_id)
        args.rank = 0
        args.device_num = 1

    args.n_epochs = min(args.max_epoch, args.n_epochs)
    args.n_epochs_decay = args.max_epoch - args.n_epochs
    if phase == 'train':
        args.isTrain = True
    else:
        args.isTrain = False
    args.phase = phase
    return args
