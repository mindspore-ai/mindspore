# Copyright 2020 Huawei Technologies Co., Ltd
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
Test centerface example
"""
import os
import time
import argparse
import datetime
import scipy.io as sio

from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.utils import get_logger
from src.var_init import default_recurisive_init
from src.centerface import CenterfaceMobilev2, CenterFaceWithNms
from src.config import ConfigCenterface

from dependency.centernet.src.lib.detectors.base_detector import CenterFaceDetector
from dependency.evaluate.eval import evaluation

dev_id = int(os.getenv('DEVICE_ID'))
context.set_context(mode=context.GRAPH_MODE, enable_auto_mixed_precision=False,
                    device_target="Ascend", save_graphs=False, device_id=dev_id)

parser = argparse.ArgumentParser('mindspore coco training')
parser.add_argument('--data_dir', type=str, default='', help='train data dir')
parser.add_argument('--test_model', type=str, default='', help='test model dir')
parser.add_argument('--ground_truth_mat', type=str, default='', help='ground_truth, mat type')
parser.add_argument('--save_dir', type=str, default='', help='save_path for evaluate')
parser.add_argument('--ground_truth_path', type=str, default='', help='ground_truth path, contain all mat file')
parser.add_argument('--eval', type=int, default=0, help='if do eval after test')
parser.add_argument('--eval_script_path', type=str, default='', help='evaluate script path')
parser.add_argument('--rank', type=int, default=0, help='local rank of distributed')
parser.add_argument('--ckpt_path', type=str, default='outputs/', help='checkpoint save location')
parser.add_argument('--ckpt_name', type=str, default="", help='input model name')
parser.add_argument('--device_num', type=int, default=1, help='device num for testing')
parser.add_argument('--steps_per_epoch', type=int, default=198, help='steps for each epoch')
parser.add_argument('--start', type=int, default=0, help='start loop number, used to calculate first epoch number')
parser.add_argument('--end', type=int, default=18, help='end loop number, used to calculate last epoch number')

args, _ = parser.parse_known_args()

if __name__ == "__main__":
    # logger
    args.outputs_dir = os.path.join(args.ckpt_path,
                                    datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))
    args.logger = get_logger(args.outputs_dir, args.rank)
    args.logger.save_args(args)

    if args.ckpt_name != "":
        args.start = 0
        args.end = 1

    for loop in range(args.start, args.end, 1):
        network = CenterfaceMobilev2()
        default_recurisive_init(network)

        if args.ckpt_name == "":
            ckpt_num = loop * args.device_num + args.rank + 1
            ckpt_name = "0-" + str(ckpt_num) + "_" + str(args.steps_per_epoch * ckpt_num) + ".ckpt"
        else:
            ckpt_name = args.ckpt_name

        test_model = args.test_model + ckpt_name
        if not test_model:
            args.logger.info('load_model {} none'.format(test_model))
            continue

        if os.path.isfile(test_model):
            param_dict = load_checkpoint(test_model)
            param_dict_new = {}
            for key, values in param_dict.items():
                if key.startswith('moments.') or key.startswith('moment1.') or key.startswith('moment2.'):
                    continue
                elif key.startswith('centerface_network.'):
                    param_dict_new[key[19:]] = values
                else:
                    param_dict_new[key] = values

            load_param_into_net(network, param_dict_new)
            args.logger.info('load_model {} success'.format(test_model))
        else:
            args.logger.info('{} not exists or not a pre-trained file'.format(test_model))
            continue

        train_network_type_nms = 1 # default with num
        if train_network_type_nms:
            network = CenterFaceWithNms(network)
            args.logger.info('train network type with nms')
        network.set_train(False)
        args.logger.info('finish get network')

        config = ConfigCenterface()

        # test network -----------
        start = time.time()

        ground_truth_mat = sio.loadmat(args.ground_truth_mat)
        event_list = ground_truth_mat['event_list']
        file_list = ground_truth_mat['file_list']
        if args.ckpt_name == "":
            save_path = args.save_dir + str(ckpt_num) + '/'
        else:
            save_path = args.save_dir+ '/'
        detector = CenterFaceDetector(config, network)

        for index, event in enumerate(event_list):
            file_list_item = file_list[index][0]
            im_dir = event[0][0]
            if not os.path.exists(save_path + im_dir):
                os.makedirs(save_path + im_dir)
                args.logger.info('save_path + im_dir={}'.format(save_path + im_dir))
            for num, file in enumerate(file_list_item):
                im_name = file[0][0]
                zip_name = '%s/%s.jpg' % (im_dir, im_name)
                img_path = os.path.join(args.data_dir, zip_name)
                args.logger.info('img_path={}'.format(img_path))

                dets = detector.run(img_path)['results']

                f = open(save_path + im_dir + '/' + im_name + '.txt', 'w')
                f.write('{:s}\n'.format('%s/%s.jpg' % (im_dir, im_name)))
                f.write('{:d}\n'.format(len(dets)))
                for b in dets[1]:
                    x1, y1, x2, y2, s = b[0], b[1], b[2], b[3], b[4]
                    f.write('{:.1f} {:.1f} {:.1f} {:.1f} {:.3f}\n'.format(x1, y1, (x2 - x1 + 1), (y2 - y1 + 1), s))
                f.close()
                args.logger.info('event:{}, num:{}'.format(index + 1, num + 1))

                end = time.time()
                args.logger.info("============num {} time {}".format(num, (end-start)*1000))
                start = end

        if args.eval:
            args.logger.info('==========start eval===============')
            args.logger.info("test output path = {}".format(save_path))
            if os.path.isdir(save_path):
                evaluation(save_path, args.ground_truth_path)
            else:
                args.logger.info('no test output path')
            args.logger.info('==========end eval===============')

        if args.ckpt_name != "":
            break

    args.logger.info('==========end testing===============')
