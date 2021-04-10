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
"""Face Quality Assessment eval."""
import os
import warnings
import argparse
import numpy as np
import cv2
from tqdm import tqdm

import mindspore.nn as nn
from mindspore import Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.ops import operations as P
from mindspore import context

from src.face_qa import FaceQABackbone

warnings.filterwarnings('ignore')
devid = int(os.getenv('DEVICE_ID'))
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=False, device_id=devid)


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=1)

def get_md_output(out):
    '''get md output'''
    out_eul = out[0].asnumpy().astype(np.float32)[0]
    heatmap = out[1].asnumpy().astype(np.float32)[0]
    eulers = out_eul * 90

    kps_score_sum = 0
    kp_scores = list()
    kp_coord_ori = list()

    for i, _ in enumerate(heatmap):
        map_1 = heatmap[i].reshape(1, 48*48)
        map_1 = softmax(map_1)

        kp_coor = map_1.argmax()
        max_response = map_1.max()
        kp_scores.append(max_response)
        kps_score_sum += min(max_response, 0.25)
        kp_coor = int((kp_coor % 48) * 2.0), int((kp_coor / 48) * 2.0)
        kp_coord_ori.append(kp_coor)

    return kp_scores, kps_score_sum, kp_coord_ori, eulers, 1


def read_gt(txt_path, x_length, y_length):
    '''read gt'''
    txt_line = open(txt_path).readline()
    eulers_txt = txt_line.strip().split(" ")[:3]
    kp_list = [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]]
    box_cur = txt_line.strip().split(" ")[3:]
    bndbox = []
    for index in range(len(box_cur) // 2):
        bndbox.append([box_cur[index * 2], box_cur[index * 2 + 1]])
    kp_id = -1
    for box in bndbox:
        kp_id = kp_id + 1
        x_coord = float(box[0])
        y_coord = float(box[1])
        if x_coord < 0 or y_coord < 0:
            continue

        kp_list[kp_id][0] = int(float(x_coord) / x_length * 96)

        kp_list[kp_id][1] = int(float(y_coord) / y_length * 96)

    return eulers_txt, kp_list


def read_img(img_path):
    img_ori = cv2.imread(img_path)
    img = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (96, 96))
    img = img.transpose(2, 0, 1)
    img = np.array([img]).astype(np.float32)/255.
    img = Tensor(img)
    return img, img_ori


blur_soft = nn.Softmax(0)
kps_soft = nn.Softmax(-1)
reshape = P.Reshape()
argmax = P.ArgMaxWithValue()


def test_trains(args):
    '''test trains'''
    print('----eval----begin----')

    model_path = args.pretrained
    result_file = model_path.replace('.ckpt', '.txt')
    if os.path.exists(result_file):
        os.remove(result_file)
    epoch_result = open(result_file, 'a')
    epoch_result.write(model_path + '\n')

    network = FaceQABackbone()
    ckpt_path = model_path

    if os.path.isfile(ckpt_path):
        param_dict = load_checkpoint(ckpt_path)

        param_dict_new = {}
        for key, values in param_dict.items():
            if key.startswith('moments.'):
                continue
            elif key.startswith('network.'):
                param_dict_new[key[8:]] = values
            else:
                param_dict_new[key] = values
        load_param_into_net(network, param_dict_new)

    else:
        print('wrong model path')
        return 1

    path = args.eval_dir
    kp_error_all = [[], [], [], [], []]
    eulers_error_all = [[], [], []]
    kp_ipn = []

    file_list = os.listdir(path)
    for file_name in tqdm(file_list):
        if file_name.endswith('jpg'):
            img_path = os.path.join(path, file_name)
            img, img_ori = read_img(img_path)

            txt_path = img_path.replace('jpg', 'txt')

            if os.path.exists(txt_path):
                euler_kps_do = True
                x_length = img_ori.shape[1]
                y_length = img_ori.shape[0]
                eulers_gt, kp_list = read_gt(txt_path, x_length, y_length)
            else:
                euler_kps_do = False
                continue

            out = network(img)

            _, _, kp_coord_ori, eulers_ori, _ = get_md_output(out)

            if euler_kps_do:
                eulgt = list(eulers_gt)
                for euler_id, _ in enumerate(eulers_ori):
                    eulori = eulers_ori[euler_id]
                    eulers_error_all[euler_id].append(abs(eulori-float(eulgt[euler_id])))

                eye01 = kp_list[0]
                eye02 = kp_list[1]
                eye_dis = 1
                cur_flag = True
                if eye01[0] < 0 or eye01[1] < 0 or eye02[0] < 0 or eye02[1] < 0:
                    cur_flag = False
                else:
                    eye_dis = np.sqrt(np.square(abs(eye01[0]-eye02[0]))+np.square(abs(eye01[1]-eye02[1])))
                cur_error_list = []
                for i in range(5):
                    kp_coord_gt = kp_list[i]
                    kp_coord_model = kp_coord_ori[i]
                    if kp_coord_gt[0] != -1:
                        dis = np.sqrt(np.square(
                            kp_coord_gt[0] - kp_coord_model[0]) + np.square(kp_coord_gt[1] - kp_coord_model[1]))
                        kp_error_all[i].append(dis)
                        cur_error_list.append(dis)
                if cur_flag:
                    kp_ipn.append(sum(cur_error_list)/len(cur_error_list)/eye_dis)

    kp_ave_error = []
    for kps, _ in enumerate(kp_error_all):
        kp_ave_error.append("%.3f" % (sum(kp_error_all[kps])/len(kp_error_all[kps])))

    euler_ave_error = []
    elur_mae = []
    for eulers, _ in enumerate(eulers_error_all):
        euler_ave_error.append("%.3f" % (sum(eulers_error_all[eulers])/len(eulers_error_all[eulers])))
        elur_mae.append((sum(eulers_error_all[eulers])/len(eulers_error_all[eulers])))

    print(r'5 keypoints average err:'+str(kp_ave_error))
    print(r'3 eulers average err:'+str(euler_ave_error))
    print('IPN of 5 keypoints:'+str(sum(kp_ipn)/len(kp_ipn)*100))
    print('MAE of elur:'+str(sum(elur_mae)/len(elur_mae)))

    epoch_result.write(str(sum(kp_ipn)/len(kp_ipn)*100)+'\t'+str(sum(elur_mae)/len(elur_mae))+'\t'
                       + str(kp_ave_error)+'\t'+str(euler_ave_error)+'\n')

    print('----eval----end----')
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Face Quality Assessment')
    parser.add_argument('--eval_dir', type=str, default='', help='eval image dir, e.g. /home/test')
    parser.add_argument('--pretrained', type=str, default='', help='pretrained model to load')

    arg = parser.parse_args()

    test_trains(arg)
