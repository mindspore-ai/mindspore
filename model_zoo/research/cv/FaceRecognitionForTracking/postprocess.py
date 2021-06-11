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
"""post process for 310 inference"""
import os
import re
import warnings
import argparse
import numpy as np
from tqdm import tqdm

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='FaceRecognitionForTracking calcul Recall')
parser.add_argument("--result_path", type=str, required=True, default='', help="result file path")
parser.add_argument("--data_dir", type=str, required=True, default='', help="data dir")
args = parser.parse_args()


def inclass_likehood(ims_info, types='cos'):
    '''Inclass likehood.'''
    obj_feas = {}
    likehoods = []
    for name, _, fea in ims_info:
        if re.split('_\\d\\d\\d\\d', name)[0] not in obj_feas:
            obj_feas[re.split('_\\d\\d\\d\\d', name)[0]] = []
        obj_feas[re.split('_\\d\\d\\d\\d', name)[0]].append(fea)  # pylint: "_\d\d\d\d" -> "_\\d\\d\\d\\d"
    for _, feas in tqdm(obj_feas.items()):
        feas = np.array(feas)
        if types == 'cos':
            likehood_mat = np.dot(feas, np.transpose(feas)).tolist()
            for row in likehood_mat:
                likehoods += row
        else:
            for fea in feas.tolist():
                likehoods += np.sum(-(fea - feas) ** 2, axis=1).tolist()

    likehoods = np.array(likehoods)
    return likehoods


def btclass_likehood(ims_info, types='cos'):
    '''Btclass likehood.'''
    likehoods = []
    count = 0
    for name1, _, fea1 in tqdm(ims_info):
        count += 1
        # pylint: "_\d\d\d\d" -> "_\\d\\d\\d\\d"
        frame_id1, _ = re.split('_\\d\\d\\d\\d', name1)[0], name1.split('_')[-1]
        fea1 = np.array(fea1)
        for name2, _, fea2 in ims_info:
            # pylint: "_\d\d\d\d" -> "_\\d\\d\\d\\d"
            frame_id2, _ = re.split('_\\d\\d\\d\\d', name2)[0], name2.split('_')[-1]
            if frame_id1 == frame_id2:
                continue
            fea2 = np.array(fea2)
            if types == 'cos':
                likehoods.append(np.sum(fea1 * fea2))
            else:
                likehoods.append(np.sum(-(fea1 - fea2) ** 2))

    likehoods = np.array(likehoods)
    return likehoods


def tar_at_far(inlikehoods, btlikehoods):
    test_point = [0.5, 0.3, 0.1, 0.01, 0.001, 0.0001, 0.00001]
    tar_far = []
    for point in test_point:
        thre = btlikehoods[int(btlikehoods.size * point)]
        n_ta = np.sum(inlikehoods > thre)
        tar_far.append((point, float(n_ta) / inlikehoods.size, thre))

    return tar_far


def main():
    with open("result.txt", 'a+') as result_fw:
        root_path = args.data_dir
        root_file_list = os.listdir(root_path)
        ims_info = []
        for sub_path in root_file_list:
            for im_path in os.listdir(os.path.join(root_path, sub_path)):
                ims_info.append((im_path.split('.')[0], os.path.join(root_path, sub_path, im_path)))

        paths = [path for name, path in ims_info]
        names = [name for name, path in ims_info]
        print("exact feature...")
        result_shape = (1, 128)
        result_path = args.result_path
        l_t = []
        for file in [name + "_0.bin" for name in names]:
            full_file_path = os.path.join(result_path, file)
            if os.path.isfile(full_file_path):
                result = np.fromfile(full_file_path, dtype=np.float32).reshape(result_shape).astype(np.float16)
                l_t.append(result)
        feas = np.concatenate(l_t, axis=0)
        ims_info = list(zip(names, paths, feas.tolist()))

        print("exact inclass likehood...")
        inlikehoods = inclass_likehood(ims_info)
        inlikehoods[::-1].sort()

        print("exact btclass likehood...")
        btlikehoods = btclass_likehood(ims_info)
        btlikehoods[::-1].sort()
        tar_far = tar_at_far(inlikehoods, btlikehoods)

        for far, tar, thre in tar_far:
            print('---{}: {}@{}'.format(far, tar, thre))

        for far, tar, thre in tar_far:
            result_fw.write('{}: {}@{} \n'.format(far, tar, thre))


if __name__ == '__main__':
    main()
