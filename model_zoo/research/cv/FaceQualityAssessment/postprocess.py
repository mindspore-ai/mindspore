# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
"""Face Quality Assessment cal acc."""
import os
import warnings
import argparse
import numpy as np
import cv2
from tqdm import tqdm

from mindspore import context

warnings.filterwarnings('ignore')


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=1)

def get_md_output(result_path, file_name):
    '''get md output'''
    eul_result_path = os.path.join(result_path, file_name + "_0.bin")
    heatmap_result_path = os.path.join(result_path, file_name + "_1.bin")
    out_eul = np.fromfile(eul_result_path, dtype=np.float32)
    heatmap = np.fromfile(heatmap_result_path, dtype=np.float32).reshape([1, 5, 48, 48])
    heatmap = heatmap[0]
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
    return img_ori

def test_infer(args):
    '''test infer starts'''
    print('----infer----begin----')

    result_file = './result_file.txt'
    if os.path.exists(result_file):
        os.remove(result_file)
    epoch_result = open(result_file, 'a')
    epoch_result.write('./FaceQualityAssessment' + '\n')

    path = args.result_path
    kp_error_all = [[], [], [], [], []]
    eulers_error_all = [[], [], []]
    kp_ipn = []

    file_list = os.listdir(path)
    for file in tqdm(file_list):
        file_name = file.split('_')[0]
        img_path = os.path.join(args.data_path, file_name + '.jpg')
        label_path = os.path.join(args.label_path, file_name + '.txt')
        img_ori = read_img(img_path)
        x_length = img_ori.shape[1]
        y_length = img_ori.shape[0]
        eulers_gt, kp_list = read_gt(label_path, x_length, y_length)
        _, _, kp_coord_ori, eulers_ori, _ = get_md_output(args.result_path, file_name)
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

    print('----infer----end----')
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Face Quality Assessment')
    parser.add_argument('--result_path', type=str, default='', help='infer results, e.g. /result_Files')
    parser.add_argument('--data_path', type=str, default='', help='original imagess')
    parser.add_argument('--label_path', type=str, default='', help='original txt folder after preprocess')
    parser.add_argument('--device_target', type=str, choices=['Ascend', 'GPU', 'CPU'], default='Ascend',
                        help='device target')

    arg = parser.parse_args()

    context.set_context(mode=context.GRAPH_MODE, device_target=arg.device_target, save_graphs=False)
    if arg.device_target == 'Ascend':
        devid = 0
        context.set_context(device_id=devid)

    test_infer(arg)
