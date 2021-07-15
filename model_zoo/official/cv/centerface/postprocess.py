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
import numpy as np
from src.model_utils.config import config

from dependency.centernet.src.lib.detectors.base_detector import CenterFaceDetector
from dependency.evaluate.eval import evaluation

dct_map = {'16': '16--Award_Ceremony', '26': '26--Soldier_Drilling', '29': '29--Students_Schoolkids',
           '30': '30--Surgeons', '52': '52--Photographers', '59': '59--people--driving--car', '44': '44--Aerobics',
           '50': '50--Celebration_Or_Party', '19': '19--Couple', '38': '38--Tennis', '37': '37--Soccer',
           '48': '48--Parachutist_Paratrooper', '53': '53--Raid', '6': '6--Funeral', '40': '40--Gymnastics',
           '5': '5--Car_Accident', '39': '39--Ice_Skating', '47': '47--Matador_Bullfighter', '61': '61--Street_Battle',
           '56': '56--Voter', '18': '18--Concerts', '1': '1--Handshaking', '2': '2--Demonstration',
           '28': '28--Sports_Fan', '4': '4--Dancing', '43': '43--Row_Boat', '49': '49--Greeting', '12': '12--Group',
           '24': '24--Soldier_Firing', '33': '33--Running', '11': '11--Meeting', '36': '36--Football',
           '45': '45--Balloonist', '15': '15--Stock_Market', '51': '51--Dresses', '7': '7--Cheering',
           '32': '32--Worker_Laborer', '58': '58--Hockey', '35': '35--Basketball', '22': '22--Picnic',
           '55': '55--Sports_Coach_Trainer', '3': '3--Riot', '23': '23--Shoppers', '34': '34--Baseball',
           '8': '8--Election_Campain', '9': '9--Press_Conference', '17': '17--Ceremony', '13': '13--Interview',
           '20': '20--Family_Group', '25': '25--Soldier_Patrol', '42': '42--Car_Racing', '0': '0--Parade',
           '14': '14--Traffic', '41': '41--Swimming', '46': '46--Jockey', '10': '10--People_Marching',
           '54': '54--Rescue', '57': '57--Angler', '31': '31--Waiter_Waitress', '27': '27--Spa', '21': '21--Festival'}


def cal_acc(result_path, label_path, meta_path, save_path):
    detector = CenterFaceDetector(config, None)
    if not os.path.exists(save_path):
        for im_dir in dct_map.values():
            out_path = os.path.join(save_path, im_dir)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
    name_list = np.load(os.path.join(meta_path, "name_list.npy"), allow_pickle=True)
    meta_list = np.load(os.path.join(meta_path, "meta_list.npy"), allow_pickle=True)

    for num, im_name in enumerate(name_list):
        meta = meta_list[num]
        output_hm = np.fromfile(os.path.join(result_path, im_name) + "_0.bin", dtype=np.float32).reshape((1, 200))
        output_wh = np.fromfile(os.path.join(result_path, im_name) + "_1.bin", dtype=np.float32).reshape(
            (1, 2, 208, 208))
        output_off = np.fromfile(os.path.join(result_path, im_name) + "_2.bin", dtype=np.float32).reshape(
            (1, 2, 208, 208))
        output_kps = np.fromfile(os.path.join(result_path, im_name) + "_3.bin", dtype=np.float32).reshape(
            (1, 10, 208, 208))
        topk_inds = np.fromfile(os.path.join(result_path, im_name) + "_4.bin", dtype=np.int32).reshape((1, 200))

        reg = output_off if config.reg_offset else None
        detections = []
        for scale in config.test_scales:
            dets = detector.centerface_decode(output_hm, output_wh, output_kps, reg=reg, opt_k=config.K,
                                              topk_inds=topk_inds)
            dets = detector.post_process(dets, meta, scale)
            detections.append(dets)
            dets = detector.merge_outputs(detections)
            index = im_name.split('_')[0]
            im_dir = dct_map.get(index)
            with open(save_path + '/' + im_dir + '/' + im_name + '.txt', 'w') as f:
                f.write('{:s}\n'.format('%s/%s.jpg' % (im_dir, im_name)))
                f.write('{:d}\n'.format(len(dets)))
                for b in dets[1]:
                    x1, y1, x2, y2, s = b[0], b[1], b[2], b[3], b[4]
                    f.write('{:.1f} {:.1f} {:.1f} {:.1f} {:.3f}\n'.format(x1, y1, (x2 - x1 + 1), (y2 - y1 + 1), s))
        print(f"no.[{num}], image_nameL {im_name}")
    evaluation(save_path, label_path)


if __name__ == '__main__':
    cal_acc(config.result_path, config.label_path, config.meta_path, config.save_path)
