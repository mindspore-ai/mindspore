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

import os
import argparse
from xml.etree import ElementTree as ET
from PIL import Image
import numpy as np


def init_args():
    parser = argparse.ArgumentParser('')
    parser.add_argument('-d', '--dataset_dir', type=str, default='./',
                        help='Directory containing test_features.tfrecords')
    parser.add_argument('-x', '--xml_file', type=str, default='test.xml',
                        help='Directory where character dictionaries for the dataset were stored')
    parser.add_argument('-o', '--output_dir', type=str, default='./processed',
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


def image_crop_save(image, location, output_dir):
    """
    crop image with location (h,w,x,y)
    save cropped image to output directory
    """
    start_x = location[2]
    end_x = start_x + location[1]
    start_y = location[3]
    if start_y < 0:
        start_y = 0
    end_y = start_y + location[0]
    print("image array shape :{}".format(image.shape))
    print("crop region ", start_x, end_x, start_y, end_y)
    if len(image.shape) == 3:
        cropped = image[start_y:end_y, start_x:end_x, :]
    else:
        cropped = image[start_y:end_y, start_x:end_x]
    im = Image.fromarray(np.uint8(cropped))
    im.save(output_dir)


def convert():
    args = init_args()
    if not os.path.exists(args.dataset_dir):
        raise ValueError("dataset_dir :{} does not exist".format(args.dataset_dir))

    if not os.path.exists(args.xml_file):
        raise ValueError("xml_file :{} does not exist".format(args.xml_file))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    ims_labels_dict = xml_to_dict(args.xml_file, True)
    num_images = len(ims_labels_dict)
    lexicon_list = []
    annotation_list = []
    print("Converting annotation, {} images in total ".format(num_images))
    for i in range(num_images):
        img_label = ims_labels_dict[i]
        image_name = img_label['imageName']
        lex = img_label['lex']
        rects = img_label['rect']
        name, ext = image_name.split('.')
        fullpath = os.path.join(args.dataset_dir, image_name)
        im_array = np.asarray(Image.open(fullpath))
        lexicon_list.append(lex)
        print("processing image: {}".format(image_name))
        for j, rect in enumerate(rects):
            rect = rects[j]
            location = rect['location']
            h = int(location['height'])
            w = int(location['width'])
            x = int(location['x'])
            y = int(location['y'])
            label = rect['label']
            loc = [h, w, x, y]
            output_name = name + "_" + str(j) + "_" + label + '.' + ext
            output_file = os.path.join(args.output_dir, output_name)
            image_crop_save(im_array, loc, output_file)
            ann = output_name + "," + label + ',' + str(i)
            annotation_list.append(ann)

    lex_file = './lexicon_ann_train.txt'
    ann_file = './annotation_train.txt'
    with open(lex_file, 'w') as f:
        for line in lexicon_list:
            txt = line + '\n'
            f.write(txt)

    with open(ann_file, 'w') as f:
        for line in annotation_list:
            txt = line + '\n'
            f.write(txt)


if __name__ == "__main__":
    convert()
