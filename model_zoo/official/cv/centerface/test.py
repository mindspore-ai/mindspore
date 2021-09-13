# Copyright 2020-21 Huawei Technologies Co., Ltd
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
import datetime
import scipy.io as sio

from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.var_init import default_recurisive_init
from src.centerface import CenterfaceMobilev2, CenterFaceWithNms

from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.device_adapter import get_device_id

from dependency.centernet.src.lib.detectors.base_detector import CenterFaceDetector
from dependency.evaluate.eval import evaluation

dev_id = get_device_id()
context.set_context(mode=context.GRAPH_MODE,
                    device_target=config.device_target, save_graphs=False)

if config.device_target == "Ascend":
    context.set_context(device_id=dev_id)


def modelarts_process():
    config.data_dir = config.data_path
    config.save_dir = config.output_path
    config.ckpt_path = os.path.join(config.output_path, config.ckpt_path)


@moxing_wrapper(pre_process=modelarts_process)
def test_centerface():
    """" test_centerface """
    config.outputs_dir = os.path.join(config.ckpt_path, datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))

    if config.ckpt_name != "":
        config.start = 0
        config.end = 1

    for loop in range(config.start, config.end, 1):
        network = CenterfaceMobilev2()
        default_recurisive_init(network)

        if config.ckpt_name == "":
            ckpt_num = loop * config.device_num + config.rank + 1
            ckpt_name = "0-" + str(ckpt_num) + "_" + str(config.steps_per_epoch * ckpt_num) + ".ckpt"
        else:
            ckpt_name = config.ckpt_name

        test_model = config.test_model + "/" + ckpt_name
        if not test_model:
            print('load_model {} none'.format(test_model))
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
            print('load_model {} success'.format(test_model))
        else:
            print('{} not exists or not a pre-trained file'.format(test_model))
            continue

        train_network_type_nms = 1 # default with num
        if train_network_type_nms:
            network = CenterFaceWithNms(network)
            print('train network type with nms')
        network.set_train(False)
        print('finish get network')

        # test network -----------
        start = time.time()

        ground_truth_mat = sio.loadmat(config.ground_truth_mat)
        event_list = ground_truth_mat['event_list']
        file_list = ground_truth_mat['file_list']
        if config.ckpt_name == "":
            save_path = config.save_dir + str(ckpt_num) + '/'
        else:
            save_path = config.save_dir+ '/'
        detector = CenterFaceDetector(config, network)

        for index, event in enumerate(event_list):
            file_list_item = file_list[index][0]
            im_dir = event[0][0]
            if not os.path.exists(save_path + im_dir):
                os.makedirs(save_path + im_dir)
                print('save_path + im_dir={}'.format(save_path + im_dir))
            for num, file_obj in enumerate(file_list_item):
                im_name = file_obj[0][0]
                zip_name = '%s/%s.jpg' % (im_dir, im_name)
                img_path = os.path.join(config.data_dir, zip_name)
                print('img_path={}'.format(img_path))

                dets = detector.run(img_path)['results']

                f = open(save_path + im_dir + '/' + im_name + '.txt', 'w')
                f.write('{:s}\n'.format('%s/%s.jpg' % (im_dir, im_name)))
                f.write('{:d}\n'.format(len(dets)))
                for b in dets[1]:
                    x1, y1, x2, y2, s = b[0], b[1], b[2], b[3], b[4]
                    f.write('{:.1f} {:.1f} {:.1f} {:.1f} {:.3f}\n'.format(x1, y1, (x2 - x1 + 1), (y2 - y1 + 1), s))
                f.close()
                print('event:{}, num:{}'.format(index + 1, num + 1))

                end = time.time()
                print("============num {} time {}".format(num, (end-start)*1000))
                start = end

        if config.eval:
            print('==========start eval===============')
            print("test output path = {}".format(save_path))
            if os.path.isdir(save_path):
                evaluation(save_path, config.ground_truth_path)
            else:
                print('no test output path')
            print('==========end eval===============')

        if config.ckpt_name != "":
            break

    print('==========end testing===============')

if __name__ == "__main__":
    test_centerface()
