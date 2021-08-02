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
import json,cv2
from collections import defaultdict

ban_path = '/opt/npu/dataset/coco/coco2014/5k.txt'
with open(ban_path, 'r')as f:
    ban_list = f.read().split('\n')[:-1]
    ban_list = [i.split('/')[-1] for i in ban_list]

name_box_id = defaultdict(list)
id_name = dict()

f = open(
    "/opt/npu/dataset/coco/coco2014/annotations/instances_train2014.json",
    encoding='utf-8')
data = json.load(f)
annotations = data['annotations']
for ant in annotations:
    id = ant['image_id']
    name = '/opt/npu/dataset/coco/coco2014/train2014/COCO_train2014_%012d.jpg' % id
    cat = ant['category_id']

    if cat >= 1 and cat <= 11:
        cat = cat - 1
    elif cat >= 13 and cat <= 25:
        cat = cat - 2
    elif cat >= 27 and cat <= 28:
        cat = cat - 3
    elif cat >= 31 and cat <= 44:
        cat = cat - 5
    elif cat >= 46 and cat <= 65:
        cat = cat - 6
    elif cat == 67:
        cat = cat - 7
    elif cat == 70:
        cat = cat - 9
    elif cat >= 72 and cat <= 82:
        cat = cat - 10
    elif cat >= 84 and cat <= 90:
        cat = cat - 11

    name_box_id[name].append([ant['bbox'], cat])




f = open(
    "/opt/npu/dataset/coco/coco2014/annotations/instances_val2014.json",
    encoding='utf-8')
data = json.load(f)
annotations = data['annotations']
for ant in annotations:
    id = ant['image_id']
    name = '/opt/npu/dataset/coco/coco2014/val2014/COCO_val2014_%012d.jpg' % id
    cat = ant['category_id']

    if cat >= 1 and cat <= 11:
        cat = cat - 1
    elif cat >= 13 and cat <= 25:
        cat = cat - 2
    elif cat >= 27 and cat <= 28:
        cat = cat - 3
    elif cat >= 31 and cat <= 44:
        cat = cat - 5
    elif cat >= 46 and cat <= 65:
        cat = cat - 6
    elif cat == 67:
        cat = cat - 7
    elif cat == 70:
        cat = cat - 9
    elif cat >= 72 and cat <= 82:
        cat = cat - 10
    elif cat >= 84 and cat <= 90:
        cat = cat - 11

    name_box_id[name].append([ant['bbox'], cat])
    





f = open('data/coco2014_trainval.txt', 'w')
ii = 0
for idx, key in enumerate(name_box_id.keys()):
    if key.split('/')[-1] in ban_list:
        continue

    print('trainval', key.split('/')[-1])

    f.write('%d '%ii)
    ii += 1
    f.write(key)

    img = cv2.imread(key)
    h,w,c = img.shape

    f.write(' %d %d'%(w,h))

    box_infos = name_box_id[key]
    for info in box_infos:
        x_min = int(info[0][0])
        y_min = int(info[0][1])
        x_max = x_min + int(info[0][2])
        y_max = y_min + int(info[0][3])

        box_info = " %d %d %d %d %d" % (
            int(info[1]), x_min, y_min, x_max, y_max
        )
        f.write(box_info)
    f.write('\n')
f.close()
