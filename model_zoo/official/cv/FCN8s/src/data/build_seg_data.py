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
from mindspore.mindrecord import FileWriter


seg_schema = {"file_name": {"type": "string"}, "label": {"type": "bytes"}, "data": {"type": "bytes"}}


def parse_args():
    parser = argparse.ArgumentParser('mindrecord')

    parser.add_argument('--data_root', type=str, default='', help='root path of data')
    parser.add_argument('--data_lst', type=str, default='', help='list of data')
    parser.add_argument('--dst_path', type=str, default='', help='save path of mindrecords')
    parser.add_argument('--num_shards', type=int, default=8, help='number of shards')
    parser.add_argument('--shuffle', type=bool, default=True, help='shuffle or not')

    parser_args, _ = parser.parse_known_args()
    return parser_args


if __name__ == '__main__':
    args = parse_args()

    datas = []
    with open(args.data_lst) as f:
        lines = f.readlines()
    if args.shuffle:
        np.random.shuffle(lines)

    dst_dir = '/'.join(args.dst_path.split('/')[:-1])
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    print('number of samples:', len(lines))
    writer = FileWriter(file_name=args.dst_path, shard_num=args.num_shards)
    writer.add_schema(seg_schema, "seg_schema")
    cnt = 0

    for l in lines:
        img_name = l.strip('\n')

        img_path = 'img/' + str(img_name) + '.jpg'
        label_path = 'cls_png/' + str(img_name) + '.png'

        sample_ = {"file_name": img_path.split('/')[-1]}

        with open(os.path.join(args.data_root, img_path), 'rb') as f:
            sample_['data'] = f.read()
        with open(os.path.join(args.data_root, label_path), 'rb') as f:
            sample_['label'] = f.read()
        datas.append(sample_)
        cnt += 1
        if cnt % 1000 == 0:
            writer.write_raw_data(datas)
            print('number of samples written:', cnt)
            datas = []

    if datas:
        writer.write_raw_data(datas)
    writer.commit()
    print('number of samples written:', cnt)
