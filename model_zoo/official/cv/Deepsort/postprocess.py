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
import os
import argparse
import numpy as np

parser = argparse.ArgumentParser('mindspore deepsort infer')
# Path for data
parser.add_argument('--det_dir', type=str, default='', help='det directory.')
parser.add_argument('--result_dir', type=str, default="./result_Files", help='infer result dir.')
parser.add_argument('--output_dir', type=str, default="./", help='output dir.')

args, _ = parser.parse_known_args()

if __name__ == "__main__":
    rst_path = args.result_dir
    start = end = 0

    for sequence in os.listdir(args.det_dir):
        #sequence_dir = os.path.join(mot_dir, sequence)
        start = end
        detection_dir = os.path.join(args.det_dir, sequence)
        detection_file = os.path.join(detection_dir, "det/det.txt")

        detections_in = np.loadtxt(detection_file, delimiter=',')
        detections_out = []
        raws = []
        features = []

        frame_indices = detections_in[:, 0].astype(np.int)
        min_frame_idx = frame_indices.astype(np.int).min()
        max_frame_idx = frame_indices.astype(np.int).max()

        for frame_idx in range(min_frame_idx, max_frame_idx + 1):
            mask = frame_indices == frame_idx
            rows = detections_in[mask]

            for box in rows:
                raws.append(box)
                end += 1

        raws = np.array(raws)
        for i in range(start, end):
            file_name = os.path.join(rst_path, "DeepSort_data_bs" + str(1) + '_' + str(i) + '_0.bin')
            output = np.fromfile(file_name, np.float32)
            features.append(output)
        features = np.array(features)
        detections_out += [np.r_[(row, feature)] for row, feature in zip(raws, features)]

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        output_filename = os.path.join(args.output_dir, "%s.npy" % sequence)
        print(output_filename)
        np.save(output_filename, np.asarray(detections_out), allow_pickle=False)
