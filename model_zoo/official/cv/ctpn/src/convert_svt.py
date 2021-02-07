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
"""convert svt dataset label"""
import os
import argparse
from xml.etree import ElementTree as ET
import numpy as np

def init_args():
    parser = argparse.ArgumentParser('')
    parser.add_argument('-d', '--dataset_dir', type=str, default='./',
                        help='Directory containing images')
    parser.add_argument('-x', '--xml_file', type=str, default='test.xml',
                        help='Directory where character dictionaries for the dataset were stored')
    parser.add_argument('-o', '--location_dir', type=str, default='./location',
                        help='Directory where ord map dictionaries for the dataset were stored')
    return parser.parse_args()

def xml_to_dict(xml_file, save_file=False):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    imgs_labels = []

    for ch in root:
        im_label = {}
        for ch01 in ch:
            if ch01.tag in "address":
                continue
            elif ch01.tag in 'taggedRectangles':
                # multiple children
                rect_list = []
                for ch02 in ch01:
                    rect = {}
                    rect['location'] = ch02.attrib
                    rect['label'] = ch02[0].text
                    rect_list.append(rect)
                im_label['rect'] = rect_list
            else:
                im_label[ch01.tag] = ch01.text
        imgs_labels.append(im_label)

    if save_file:
        np.save("annotation_train.npy", imgs_labels)

    return imgs_labels

def convert():
    args = init_args()
    if not os.path.exists(args.dataset_dir):
        raise ValueError("dataset_dir :{} does not exist".format(args.dataset_dir))

    if not os.path.exists(args.xml_file):
        raise ValueError("xml_file :{} does not exist".format(args.xml_file))

    if not os.path.exists(args.location_dir):
        os.makedirs(args.location_dir)

    ims_labels_dict = xml_to_dict(args.xml_file, True)
    num_images = len(ims_labels_dict)
    print("Converting annotation, {} images in total ".format(num_images))
    for i in range(num_images):
        img_label = ims_labels_dict[i]
        image_name = img_label['imageName']
        rects = img_label['rect']
        print("processing image: {}".format(image_name))
        location_file_name = os.path.join(args.location_dir, os.path.basename(image_name).replace(".jpg", ".txt"))
        f = open(location_file_name, 'w')
        for j, rect in enumerate(rects):
            rect = rects[j]
            location = rect['location']
            h = int(location['height'])
            w = int(location['width'])
            x = int(location['x'])
            y = int(location['y'])
            pos = [x, y, x+w, y+h]
            str_pos = str(pos[0]) + "," + str(pos[1]) + "," + str(pos[2]) + "," + str(pos[3])
            f.write(str_pos)
            f.write("\n")
        f.close()

if __name__ == "__main__":
    convert()
