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
"""
config
"""
import argparse
import ast


class Opts:
    """
    parameter configuration
    """
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        # basic experiment setting
        self.parser.add_argument('--load_model', default="/Fairmot/ckpt/Fairmot_7-30_1595.ckpt",
                                 help='path to pretrained model')
        self.parser.add_argument('--load_pre_model', default="/Fairmot/ckpt/dla34-ba72cf86_ms.ckpt",
                                 help='path to pretrained model')
        self.parser.add_argument('--data_cfg', type=str,
                                 default='/fairmot/data/data.json', help='load data from cfg')
        self.parser.add_argument('--arch', default='dla_34',
                                 help='model architecture. Currently tested'
                                      'resdcn_34 | resdcn_50 | resfpndcn_34 |' 'dla_34 | hrnet_18')
        self.parser.add_argument('--num_epochs', type=int, default=30, help='total training epochs.')
        self.parser.add_argument('--lr', type=float, default=1e-4,
                                 help='learning rate for batch size 12.')
        self.parser.add_argument('--batch_size', type=int, default=4, help='batch size')
        self.parser.add_argument('--input-video', type=str,
                                 default='/videos/MOT16-03.mp4', help='path to the input video')
        self.parser.add_argument('--output-root', type=str, default='./exports', help='expected output root path')
        self.parser.add_argument("--is_modelarts", type=ast.literal_eval, default=False,
                                 help="Run distribute in modelarts, default is false.")
        self.parser.add_argument("--run_distribute", type=ast.literal_eval, default=False,
                                 help="Run distribute, default is false.")
        self.parser.add_argument('--data_url', default=None, help='Location of data.')
        self.parser.add_argument('--train_url', default=None, help='Location of training outputs.')
        self.parser.add_argument('--id', type=int, default=0)
        # model
        self.parser.add_argument('--head_conv', type=int, default=-1,
                                 help='conv layer channels for output head')
        self.parser.add_argument('--down_ratio', type=int, default=4,
                                 help='output stride. Currently only supports 4.')
        # input
        self.parser.add_argument('--input_res', type=int, default=-1,
                                 help='input height and width. -1 for default from '
                                      'dataset. Will be overridden by input_h | input_w')
        self.parser.add_argument('--input_h', type=int, default=-1,
                                 help='input height. -1 for default from dataset.')
        self.parser.add_argument('--input_w', type=int, default=-1,
                                 help='input width. -1 for default from dataset.')
        # test
        self.parser.add_argument('--K', type=int, default=500, help='max number of output objects.')
        self.parser.add_argument('--not_prefetch_test', action='store_true',
                                 help='not use parallal data pre-processing.')
        self.parser.add_argument('--fix_res', action='store_true',
                                 help='fix testing resolution or keep the original resolution')
        self.parser.add_argument('--keep_res', action='store_true',
                                 help='keep the original resolution during validation.')
        # tracking
        self.parser.add_argument('--conf_thres', type=float, default=0.3, help='confidence thresh for tracking')
        self.parser.add_argument('--det_thres', type=float, default=0.3, help='confidence thresh for detection')
        self.parser.add_argument('--nms_thres', type=float, default=0.4, help='iou thresh for nms')
        self.parser.add_argument('--track_buffer', type=int, default=30, help='tracking buffer')
        self.parser.add_argument('--min-box-area', type=float, default=100, help='filter out tiny boxes')
        self.parser.add_argument('--output-format', type=str, default='video', help='video or text')
        # mot
        self.parser.add_argument('--data_dir', default='/opt_data/xidian_wks/kasor/Fairmot/dataset')
        # loss
        self.parser.add_argument('--mse_loss', action='store_true',
                                 help='use mse loss or focal loss to train keypoint heatmaps.')
        self.parser.add_argument('--reg_loss', default='l1',
                                 help='regression loss: sl1 | l1 | l2')
        self.parser.add_argument('--hm_weight', type=float, default=1,
                                 help='loss weight for keypoint heatmaps.')
        self.parser.add_argument('--off_weight', type=float, default=1,
                                 help='loss weight for keypoint local offsets.')
        self.parser.add_argument('--wh_weight', type=float, default=0.1,
                                 help='loss weight for bounding box size.')
        self.parser.add_argument('--id_loss', default='ce',
                                 help='reid loss: ce | triplet')
        self.parser.add_argument('--id_weight', type=float, default=1,
                                 help='loss weight for id')
        self.parser.add_argument('--reid_dim', type=int, default=128,
                                 help='feature dim for reid')
        self.parser.add_argument('--ltrb', default=True,
                                 help='regress left, top, right, bottom of bbox')
        self.parser.add_argument('--norm_wh', action='store_true',
                                 help='L1(\\hat(y) / y, 1) or L1(\\hat(y), y)')
        self.parser.add_argument('--dense_wh', action='store_true',
                                 help='apply weighted regression near center or '
                                      'just apply regression on center point.')
        self.parser.add_argument('--cat_spec_wh', action='store_true',
                                 help='category specific bounding box size.')
        self.parser.add_argument('--not_reg_offset', action='store_true',
                                 help='not regress local offset.')

    def parse(self, args=''):
        """parameter parse"""
        if args == '':
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)

        opt.fix_res = not opt.keep_res
        print('Fix size testing.' if opt.fix_res else 'Keep resolution testing.')
        opt.reg_offset = not opt.not_reg_offset

        if opt.head_conv == -1:  # init default head_conv
            opt.head_conv = 256 if 'dla' in opt.arch else 256
        opt.pad = 31
        opt.num_stacks = 1


        return opt

    def update_dataset_info_and_set_heads(self, opt, dataset):
        """update dataset info and set heads"""
        input_h, input_w = dataset.default_resolution
        opt.mean, opt.std = dataset.mean, dataset.std
        opt.num_classes = dataset.num_classes

        # input_h(w): opt.input_h overrides opt.input_res overrides dataset default
        input_h = opt.input_res if opt.input_res > 0 else input_h
        input_w = opt.input_res if opt.input_res > 0 else input_w
        opt.input_h = opt.input_h if opt.input_h > 0 else input_h
        opt.input_w = opt.input_w if opt.input_w > 0 else input_w
        opt.output_h = opt.input_h // opt.down_ratio
        opt.output_w = opt.input_w // opt.down_ratio
        opt.input_res = max(opt.input_h, opt.input_w)
        opt.output_res = max(opt.output_h, opt.output_w)

        opt.heads = {'hm': opt.num_classes,
                     'wh': 2 if not opt.ltrb else 4,
                     'id': opt.reid_dim}
        if opt.reg_offset:
            opt.heads.update({'reg': 2})
        opt.nID = dataset.nID
        opt.img_size = (1088, 608)
        # opt.img_size = (864, 480)
        # opt.img_size = (576, 320)
        print('heads', opt.heads)
        return opt

    def init(self, args=''):
        """opt init"""
        default_dataset_info = {
            'mot': {'default_resolution': [608, 1088], 'num_classes': 1,
                    'mean': [0.408, 0.447, 0.470], 'std': [0.289, 0.274, 0.278],
                    'dataset': 'jde', 'nID': 14455},
        }

        class Struct:
            """opt struct"""
            def __init__(self, entries):
                for k, v in entries.items():
                    self.__setattr__(k, v)

        opt = self.parse(args)
        dataset = Struct(default_dataset_info['mot'])
        opt.dataset = dataset.dataset
        opt = self.update_dataset_info_and_set_heads(opt, dataset)
        return opt
